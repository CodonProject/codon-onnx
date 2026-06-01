# codon_onnx/service.py
import asyncio
import time
import uuid
import json
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, AsyncGenerator, Tuple, Union

from fastapi import FastAPI
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# 引入我们写好的 codon_onnx 核心组件
from codon_onnx.base import CausalLMOnnx
from codon_onnx.tokenizer import PackedTokenizer


@dataclass
class ModelCard:
    '''
    用于向服务注册模型的卡片。

    Attributes:
        model (CausalLMOnnx): ONNX Runtime 驱动的模型实例。
        tokenizer (PackedTokenizer): 兼容的纯本地分词器。
        model_id (str): 模型的唯一标识符（例如 'motif-a1'）。
        owned (str): 模型所有者信息。
    '''
    model: CausalLMOnnx
    tokenizer: PackedTokenizer
    model_id: str
    owned: str


class ChatMessage(BaseModel):
    role: str
    content: Union[str, List[Dict[str, Any]]]

    model_config = {
        'extra': 'allow'
    }


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: float = 0.7
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    repetition_penalty: float = 1.15
    max_tokens: int = 1024
    stream: bool = False

    model_config = {
        'extra': 'allow'
    }


class Service:
    '''
    兼容 OpenAI 接口规范的 FastAPI 运行时服务。
    支持一键部署任何经过 ONNX 导出的 Motif 系列模型。
    '''
    def __init__(self, models: List[ModelCard]) -> None:
        self.models = {card.model_id: card for card in models}
        # 为每个模型配备一个互斥锁，保护硬件推理通道（尤其是 GPU）
        self.locks = {card.model_id: asyncio.Lock() for card in models}
        self.app = FastAPI(title='Codon ONNX Inference Service')

        # 跨域配置
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=['*'],
            allow_credentials=True,
            allow_methods=['*'],
            allow_headers=['*'],
        )

        self._register_routes()

    def _register_routes(self) -> None:
        self.app.get('/v1/models')(self.list_models)
        self.app.get('/models')(self.list_models)
        self.app.post('/v1/chat/completions')(self.chat_completions)
        self.app.post('/chat/completions')(self.chat_completions)

    @staticmethod
    def _safe_next(iterator: Any) -> Optional[Any]:
        try:
            return next(iterator)
        except StopIteration:
            return None

    async def list_models(self) -> JSONResponse:
        data = []
        for model_id, card in self.models.items():
            data.append({
                'id': model_id,
                'object': 'model',
                'created': int(time.time()),
                'owned_by': card.owned
            })
        return JSONResponse(content={'object': 'list', 'data': data})

    def _make_chunk(
        self,
        request_id: str,
        model_id: str,
        content: str,
        reasoning_content: str,
        created_time: int,
        finish_reason: Optional[str]
    ) -> Dict[str, Any]:
        delta = {}
        if content:
            delta['content'] = content
        if reasoning_content:
            delta['reasoning_content'] = reasoning_content

        return {
            'id': request_id,
            'object': 'chat.completion.chunk',
            'created': created_time,
            'model': model_id,
            'choices': [
                {
                    'index': 0,
                    'delta': delta,
                    'logprobs': None,
                    'finish_reason': finish_reason
                }
            ]
        }

    async def _stream_generator(
        self,
        request_id: str,
        model_id: str,
        model: CausalLMOnnx,
        tokenizer: PackedTokenizer,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float,
        top_k: Optional[int],
        top_p: Optional[float],
        created_time: int
    ) -> AsyncGenerator[str, None]:
        '''
        在异步线程池（Executor）中安全地驱动 NumPy 生成器，防止阻塞 FastAPI 事件循环。
        '''
        async with self.locks[model_id]:
            def _blocking_generator():
                # 动态导入我们写好的轻量级 chat_stream 引擎
                from codon_onnx.generator import chat_stream
                for chunk in chat_stream(
                    model=model,
                    tokenizer=tokenizer,
                    messages=messages,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p
                ):
                    yield chunk

            loop = asyncio.get_running_loop()
            iterator = _blocking_generator()

            while True:
                # 每次迭代都派发到线程池，确保整个 API 服务在单线程推理时依然能够响应其他请求
                chunk = await loop.run_in_executor(None, self._safe_next, iterator)
                if chunk is None:
                    break

                content = '' if chunk.is_cot else chunk.content
                reasoning_content = chunk.content if chunk.is_cot else ''

                if content or reasoning_content:
                    yield f'data: {json.dumps(self._make_chunk(request_id, model_id, content, reasoning_content, created_time, None))}\n\n'

            # 结束标志
            yield f"data: {json.dumps(self._make_chunk(request_id, model_id, '', '', created_time, 'stop'))}\n\n"
            yield 'data: [DONE]\n\n'

    async def chat_completions(self, request: ChatCompletionRequest) -> Any:
        model_id = request.model
        if model_id not in self.models:
            return JSONResponse(
                status_code=404,
                content={
                    'error': {
                        'message': f"Model '{model_id}' not found in codon-onnx library.",
                        'type': 'invalid_request_error',
                        'param': 'model',
                        'code': 'model_not_found'
                    }
                }
            )

        card = self.models[model_id]
        model = card.model
        tokenizer = card.tokenizer

        # 格式化历史消息
        formatted_messages = []
        for msg in request.messages:
            formatted_messages.append({'role': msg.role, 'content': msg.content})

        request_id = f'chatcmpl-{uuid.uuid4()}'
        created_time = int(time.time())

        # 1. 流式响应
        if request.stream:
            return StreamingResponse(
                self._stream_generator(
                    request_id=request_id,
                    model_id=model_id,
                    model=model,
                    tokenizer=tokenizer,
                    messages=formatted_messages,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_k=request.top_k,
                    top_p=request.top_p,
                    created_time=created_time
                ),
                media_type='text/event-stream'
            )

        # 2. 非流式响应（同步聚合结果）
        async with self.locks[model_id]:
            def _blocking_generate() -> Tuple[str, str, int]:
                from codon_onnx.generator import chat_stream
                content_accum = []
                reasoning_accum = []
                total_tokens = 0
                for chunk in chat_stream(
                    model=model,
                    tokenizer=tokenizer,
                    messages=formatted_messages,
                    max_new_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_k=request.top_k,
                    top_p=request.top_p
                ):
                    if chunk.content:
                        if chunk.is_cot:
                            reasoning_accum.append(chunk.content)
                        else:
                            content_accum.append(chunk.content)
                    total_tokens += 1
                return ''.join(content_accum), ''.join(reasoning_accum), total_tokens

            loop = asyncio.get_running_loop()
            content, reasoning, total_generated = await loop.run_in_executor(None, _blocking_generate)

        message_payload = {
            'role': 'assistant',
            'content': content
        }
        if reasoning:
            # 兼容带有思维链展示的前端（如 Cherry Studio, Page Assist 等）
            message_payload['reasoning_content'] = reasoning

        return JSONResponse(
            content={
                'id': request_id,
                'object': 'chat.completion',
                'created': created_time,
                'model': model_id,
                'choices': [
                    {
                        'index': 0,
                        'message': message_payload,
                        'logprobs': None,
                        'finish_reason': 'stop'
                    }
                ],
                'usage': {
                    'prompt_tokens': 0,  # 占位符
                    'completion_tokens': total_generated,
                    'total_tokens': total_generated
                }
            }
        )

    def run(self, host: str = '0.0.0.0', port: int = 11305, **kwargs) -> None:
        '''
        启动轻量化 Uvicorn API 服务。
        '''
        uvicorn.run(self.app, host=host, port=port, **kwargs)