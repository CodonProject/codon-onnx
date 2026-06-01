import copy
from typing import List, Dict, Any
from codon_onnx.tokenizer import PackedTokenizer

class Session:
    def __init__(self, tokenizer: PackedTokenizer):
        self.tokenizer = tokenizer
        self.messages: List[Dict[str, Any]] = []

    def add_messages(self, messages: List[Dict[str, Any]]):
        self.messages.extend(copy.deepcopy(messages))

    def add_message(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})

    def get_prompt_ids(self, add_generation_prompt: bool = True, enable_thinking: bool = True) -> List[int]:
        return self.tokenizer.apply_chat_template(
            self.messages,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking
        )

    def clear(self):
        self.messages = []