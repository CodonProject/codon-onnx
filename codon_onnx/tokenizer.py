import json
import os
import zipfile
import copy
import re
from tokenizers import Tokenizer
from jinja2 import Environment

from typing import Optional, List, Dict, Any


core_tokens = ['[pad]', '[unk]', '[sep]', '[cls]']
chat_tokens = [
    '[im_start]', '[im_end]',
    '[system]', '[user]', '[model]', '[tool]',
    '[interruption]', '[fim]',
]
reasoning_tokens = ['[cot_start]', '[cot_end]']
code_tokens = ['[fim_pre]', '[fim_mid]', '[fim_suf]']
tool_tokens = ['[tool_start]', '[tool_name]', '[tool_args]', '[tool_end]']

multimodal_tokens = [
    '[image_start]', '[image_end]', '[audio_start]', '[audio_end]', 
    '[video_start]', '[video_end]'
]

base_special_tokens = (
    core_tokens + 
    chat_tokens + 
    reasoning_tokens + 
    code_tokens + 
    tool_tokens + 
    multimodal_tokens
)
base_special_tokens += [f'[unused_{i}]' for i in range(len(base_special_tokens), 64)]

chat_template = (
    "{% for message in messages %}"
        "{{ '[im_start]' }}"
        "{% if message['role'] == 'fim' %}"
            "{{ '[fim]' }}"
            "{{ '[fim_pre]' + message['prefix'] + '[fim_suf]' + message['suffix'] + '[fim_mid]' }}"
            "{% if message['middle'] %}"
                "{{ message['middle'] + '[im_end]' }}"
            "{% endif %}"
        "{% else %}"
            "{% if message['role'] in ['system', 'instruction', 'developer'] %}"
                "{{ '[system]' }}"
            "{% elif message['role'] == 'user' %}"
                "{{ '[user]' }}"
            "{% elif message['role'] in ['assistant', 'model'] %}"
                "{{ '[model]' }}"
                "{% set thought_content = message.get('thought') or message.get('reasoning_content') %}"
                "{% if thought_content %}"
                    "{{ '[cot_start]' + thought_content + '[cot_end]' }}"
                "{% else %}"
                    "{{ '[cot_start][cot_end]' }}"
                "{% endif %}"
            "{% elif message['role'] == 'tool' %}"
                "{{ '[tool]' }}"
            "{% else %}"
                "{{ message['role'] }}"
            "{% endif %}"
            "{% if message['content'] is defined and message['content'] is not none %}"
                "{% if message['content'] is string %}"
                    "{{ message['content'] }}"
                "{% else %}"
                    "{% for item in message['content'] %}"
                        "{% if item['type'] == 'text' %}"
                            "{{ item['text'] }}"
                        "{% elif item['type'] == 'image' %}"
                            "{{ '[image_start][image_end]' }}"
                        "{% elif item['type'] == 'audio' %}"
                            "{{ '[audio_start][audio_end]' }}"
                        "{% elif item['type'] == 'video' %}"
                            "{{ '[video_start][video_end]' }}"
                        "{% endif %}"
                    "{% endfor %}"
                "{% endif %}"
            "{% endif %}"
            "{% if message['tools'] is defined and message['tools'] %}"
                "{{ message['tools'] }}"
            "{% endif %}"
            "{% if message['tool_calls'] is defined and message['tool_calls'] %}"
                "{% for tool_call in message['tool_calls'] %}"
                    "{{ '[tool_start][tool_name]' + tool_call.function.name + '[tool_args]' + tool_call.function.arguments + '[tool_end]' }}"
                "{% endfor %}"
            "{% endif %}"
            "{{ '[im_end]' }}"
        "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
        "{{ '[im_start][model]' }}"
        "{% if enable_thinking is defined and enable_thinking %}"
            "{{ '[cot_start]' }}"
        "{% elif enable_thinking is defined and not enable_thinking %}"
            "{{ '[cot_start][cot_end]' }}"
        "{% endif %}"
    "{% endif %}"
)


class PackedTokenizer:
    def __init__(self, tokenizer_path: Optional[str] = None):
        self._tokenizer: Optional[Tokenizer] = None
        self.config = {}
        self.template = chat_template

        self.safe_escape = '[unused_42]'
        self.safe_escape_id: Optional[int] = None

        self._jinja_env = Environment(
            trim_blocks=True, 
            lstrip_blocks=True
        )

        self.special_tokens_set = set(base_special_tokens)
        special_escaped = [re.escape(t) for t in base_special_tokens]
        self._special_pattern = re.compile(f"({'|'.join(special_escaped)})")

        if tokenizer_path is not None:
            self.load(tokenizer_path)

    @property
    def tokenizer(self) -> Tokenizer:
        if self._tokenizer is None:
            raise ValueError("Tokenizer is not loaded.")
        return self._tokenizer
    
    def token_to_id(self, token: str) -> Optional[int]:
        return self.tokenizer.token_to_id(token)
    
    def ensure_escape(self) -> int:
        if self.safe_escape_id is None:
            tid = self.tokenizer.token_to_id(self.safe_escape)
            if tid is None:
                raise ValueError(f'Escape token {self.safe_escape} not found in vocab.')
            self.safe_escape_id = tid
        return self.safe_escape_id
    
    def _sanitize_content(self, content: Any) -> Any:
        if isinstance(content, str):
            return content.replace(']', f'{self.safe_escape}]')
        elif isinstance(content, list):
            return [
                {**item, 'text': self._sanitize_content(item['text'])} if item.get('type') == 'text' else item 
                for item in content
            ]
        return content
        
    def apply_chat_template(
        self, 
        messages: List[Dict[str, Any]], 
        add_generation_prompt: bool = True,
        **kwargs
    ) -> List[int]:
        escape_id = self.ensure_escape()
        
        safe_messages = copy.deepcopy(messages)
        for msg in safe_messages:
            if 'content' in msg:
                msg['content'] = self._sanitize_content(msg['content'])
        
        compiled_template = self._jinja_env.from_string(self.template)
        rendered_text = compiled_template.render(
            messages=safe_messages,
            add_generation_prompt=add_generation_prompt,
            **kwargs
        )
        
        raw_ids = self.encode(rendered_text, add_special_tokens=False)
        return [tid for tid in raw_ids if tid != escape_id]

    def encode(self, text: str, add_special_tokens: bool = False, **kwargs) -> List[int]:
        parts = self._special_pattern.split(text)
        
        token_ids = []
        for part in parts:
            if not part:
                continue
            if part in self.special_tokens_set:
                tid = self.token_to_id(part)
                if tid is not None:
                    token_ids.append(tid)
            else:
                safe_part = part.replace(']', f'{self.safe_escape}]')
                encoding = self.tokenizer.encode(safe_part, add_special_tokens=add_special_tokens)
                token_ids.extend(encoding.ids)
                
        return token_ids
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = False) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
    
    def save(self, path: str) -> 'PackedTokenizer':
        if self._tokenizer is None:
            raise ValueError('No tokenizer to save.')

        with zipfile.ZipFile(path, 'w') as z:
            z.writestr('tokenizer.json', self._tokenizer.to_str())
            z.writestr('tokenizer_config.json', json.dumps(self.config, indent=2))
            z.writestr('chat_template.jinja', self.template)
            
        return self
    
    def load(self, path: str) -> 'PackedTokenizer':
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")

        with zipfile.ZipFile(path, 'r') as z:
            file_list = z.namelist()
            
            def find_file(name):
                for f in file_list:
                    if f == name or f.endswith(f'/{name}'):
                        return f
                return None

            tokenizer_file = find_file('tokenizer.json')
            if tokenizer_file:
                tokenizer_json = z.read(tokenizer_file).decode('utf-8')
                self._tokenizer = Tokenizer.from_str(tokenizer_json)
            else:
                raise ValueError("tokenizer.json not found in zip file")

            config_file = find_file('tokenizer_config.json')
            if config_file:
                config_json = z.read(config_file).decode('utf-8')
                self.config = json.loads(config_json)

            template_file = find_file('chat_template.jinja')
            if template_file:
                self.template = z.read(template_file).decode('utf-8')

        return self