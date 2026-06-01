from dataclasses import dataclass
from typing import Generator, List, Dict, Optional
import numpy as np

from codon_onnx.tokenizer import PackedTokenizer
from codon_onnx.session import Session
from codon_onnx.base import CausalLMOnnx, Sampler, KVCache

@dataclass
class ChatChunk:
    content: str
    is_cot: bool
    cot_ended: bool


def chat_stream(
    model: CausalLMOnnx,
    tokenizer: PackedTokenizer,
    messages: List[Dict[str, str]],
    max_new_tokens: int = 1024,
    temperature: float = 0.3,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    enable_thinking: bool = True,
) -> Generator[ChatChunk, None, None]:
    session = Session(tokenizer)
    session.add_messages(messages)
    
    prompt_ids = session.get_prompt_ids(add_generation_prompt=True, enable_thinking=enable_thinking)
    prompt_len = len(prompt_ids)
    
    sampler = Sampler(temperature=temperature, top_k=top_k, top_p=top_p)
    kv_cache = KVCache(
        num_layers=model.num_layers,
        num_kv_heads=model.num_kv_heads,
        head_dim=model.head_dim,
        dtype=model.dtype
    )
    
    generated = np.array([prompt_ids], dtype=np.int64)

    cot_start_id = tokenizer.token_to_id('[cot_start]')
    cot_end_id = tokenizer.token_to_id('[cot_end]')
    im_end_id = tokenizer.token_to_id('[im_end]')
    pad_id = tokenizer.token_to_id('[pad]')

    is_cot = enable_thinking 
    cot_ended = False

    # A. Prefill
    positions_prefill = np.arange(0, prompt_len, dtype=np.int64).reshape(1, -1)
    logits, new_kvs = model.forward(
        input_ids=generated,
        positions=positions_prefill,
        kv_cache=kv_cache
    )
    kv_cache.update(new_kvs)
    
    next_token_logits = logits[:, -1, :]
    next_token = sampler(next_token_logits, input_ids=generated)
    generated = np.concatenate([generated, next_token], axis=-1)
    
    current_pos = prompt_len

    # B. Decode
    for _ in range(max_new_tokens - 1):
        token_val = int(next_token[0, 0])

        if token_val == im_end_id or token_val == pad_id: break

        if token_val == cot_start_id:
            is_cot = True
            token_str = ''
        elif token_val == cot_end_id:
            is_cot = False
            cot_ended = True
            token_str = ''
        else:
            token_str = tokenizer.decode([token_val], skip_special_tokens=True)
            if token_str in ['[cot_start]', '[cot_end]', '[im_end]', '[im_start]']:
                token_str = ''

        if token_str or cot_ended:
            yield ChatChunk(content=token_str, is_cot=is_cot, cot_ended=cot_ended)
            
        if cot_ended: 
            cot_ended = False

        positions_decode = np.array([[current_pos]], dtype=np.int64)
        
        logits, new_kvs = model.forward(
            input_ids=next_token,
            positions=positions_decode,
            kv_cache=kv_cache
        )
        kv_cache.update(new_kvs)
        
        current_pos += 1
        
        next_token = sampler(logits[:, -1, :], input_ids=generated)
        generated = np.concatenate([generated, next_token], axis=-1)