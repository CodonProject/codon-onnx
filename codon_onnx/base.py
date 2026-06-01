import os
import sys
import numpy as np

if sys.platform == 'win32':
    try:
        import site
        import ctypes
        
        possible_dirs = []
        if hasattr(site, 'getsitepackages'):
            possible_dirs.extend(site.getsitepackages())
        if hasattr(site, 'getusersitepackages'):
            possible_dirs.append(site.getusersitepackages())
        possible_dirs.extend(sys.path)

        found_bin_path = None
        for s_dir in possible_dirs:
            if not s_dir: continue
            cudnn_bin = os.path.join(s_dir, 'nvidia', 'cudnn', 'bin')
            if os.path.exists(cudnn_bin):
                found_bin_path = cudnn_bin
                os.add_dll_directory(cudnn_bin)
                os.environ['PATH'] = cudnn_bin + os.pathsep + os.environ['PATH']
                target_dll = os.path.join(cudnn_bin, 'cudnn64_9.dll')
                if os.path.exists(target_dll):
                    ctypes.CDLL(target_dll)
                break
    except Exception: pass

import onnxruntime as ort
from typing import Optional, List, Tuple, Union


class Sampler:
    def __init__(
        self,
        temperature: float = 0.7,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: float = 1.15
    ) -> None:
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty

    def __call__(self, logits: np.ndarray, input_ids: Optional[np.ndarray] = None) -> np.ndarray:
        logits = logits.copy()

        # Repetition Penalty
        if self.repetition_penalty != 1.0 and input_ids is not None:
            for i in range(logits.shape[0]):
                unique_tokens = np.unique(input_ids[i])
                for token_id in unique_tokens:
                    val = logits[i, token_id]
                    if val > 0:
                        logits[i, token_id] = val / self.repetition_penalty
                    else:
                        logits[i, token_id] = val * self.repetition_penalty

        # Temperature
        if self.temperature != 1.0:
            temp = max(self.temperature, 1e-5)
            logits = logits / temp

        # Top-K
        if self.top_k is not None and self.top_k > 0:
            top_k = min(self.top_k, logits.shape[-1])
            for i in range(logits.shape[0]):
                threshold = -np.partition(-logits[i], top_k - 1)[top_k - 1]
                logits[i, logits[i] < threshold] = -np.inf

        # Top-P
        if self.top_p is not None and 0.0 < self.top_p < 1.0:
            for i in range(logits.shape[0]):
                sorted_indices = np.argsort(-logits[i])
                sorted_logits = logits[i, sorted_indices]
                
                exp_logits = np.exp(sorted_logits - np.max(sorted_logits))
                probs = exp_logits / np.sum(exp_logits)
                cumulative_probs = np.cumsum(probs)

                sorted_indices_to_remove = cumulative_probs > self.top_p
                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].copy()
                sorted_indices_to_remove[0] = False

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[i, indices_to_remove] = -np.inf

        # Multinomial
        next_tokens = []
        for i in range(logits.shape[0]):
            max_logit = np.max(logits[i])
            if max_logit == -np.inf:
                probs = np.ones_like(logits[i]) / logits.shape[-1]
            else:
                exp_logits = np.exp(logits[i] - max_logit)
                probs = exp_logits / np.sum(exp_logits)
            
            token = np.random.choice(len(probs), p=probs)
            next_tokens.append(token)

        return np.array(next_tokens, dtype=np.int64).reshape(-1, 1)


class KVCache:
    def __init__(self, num_layers: int, num_kv_heads: int, head_dim: int, dtype=np.float32, device: str = 'cpu') -> None:
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device  # 'cpu' | 'cuda' | 'dml'
        self.states: List[Tuple[Union[np.ndarray, ort.OrtValue], Union[np.ndarray, ort.OrtValue]]] = []
        self.clear()

    def update(self, next_states: List[Union[np.ndarray, ort.OrtValue]]) -> None:
        self.states = []
        for i in range(self.num_layers):
            self.states.append((next_states[2 * i], next_states[2 * i + 1]))

    @property
    def current_len(self) -> int:
        if not self.states:
            return 0
        state_k = self.states[0][0]
        if isinstance(state_k, ort.OrtValue):
            return state_k.shape()[-2]
        return state_k.shape[-2]

    def clear(self) -> None:
        empty_np = np.zeros((1, self.num_kv_heads, 0, self.head_dim), dtype=self.dtype)
        if self.device == 'cuda':
            self.states = [
                (ort.OrtValue.ortvalue_from_numpy(empty_np, 'cuda', 0),
                 ort.OrtValue.ortvalue_from_numpy(empty_np, 'cuda', 0))
                for _ in range(self.num_layers)
            ]
        else:
            self.states = [(empty_np, empty_np) for _ in range(self.num_layers)]


class CausalLMOnnx:
    def __init__(self, model_path: str, force_cpu: bool = False):
        self.model_path = model_path
        self.sess_options = ort.SessionOptions()
        self.sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        available_providers = ort.get_available_providers()
        
        desired_providers = []
        if not force_cpu:
            if 'CUDAExecutionProvider' in available_providers:
                desired_providers.append('CUDAExecutionProvider')
            if 'DmlExecutionProvider' in available_providers:
                desired_providers.append('DmlExecutionProvider')
        
        desired_providers.append('CPUExecutionProvider')
        
        self.session = ort.InferenceSession(model_path, self.sess_options, providers=desired_providers)
        
        actual_providers = self.session.get_providers()
        if 'CUDAExecutionProvider' in actual_providers:
            self.device = 'cuda'
        elif 'DmlExecutionProvider' in actual_providers:
            self.device = 'dml'
        else:
            self.device = 'cpu'
            
        self.inputs = {node.name: node for node in self.session.get_inputs()}
        self.output_names = [node.name for node in self.session.get_outputs()]
        
        has_long_name = any(name.startswith('past_key_') for name in self.inputs)
        if has_long_name:
            self.k_prefix, self.v_prefix = 'past_key_', 'past_value_'
        else:
            self.k_prefix, self.v_prefix = 'past_k_', 'past_v_'

        first_kv_node = self.inputs[f'{self.k_prefix}0']
        self.num_layers = len([name for name in self.inputs if name.startswith(self.k_prefix)])
        
        self.num_kv_heads = first_kv_node.shape[1] if isinstance(first_kv_node.shape[1], int) else 2
        self.head_dim = first_kv_node.shape[3] if isinstance(first_kv_node.shape[3], int) else 96
        self.dtype = np.float16 if 'float16' in first_kv_node.type else np.float32

    def forward(
        self,
        input_ids: np.ndarray,
        positions: np.ndarray,
        kv_cache: KVCache,
        mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, List[Union[np.ndarray, ort.OrtValue]]]:
        if self.device == 'cuda':
            try:
                io_binding = self.session.io_binding()
                
                ort_input_ids = ort.OrtValue.ortvalue_from_numpy(input_ids.astype(np.int64), 'cuda', 0)
                io_binding.bind_ortvalue_input('input_ids', ort_input_ids)
                
                if 'positions' in self.inputs:
                    ort_pos = ort.OrtValue.ortvalue_from_numpy(positions.astype(np.int64), 'cuda', 0)
                    io_binding.bind_ortvalue_input('positions', ort_pos)
                elif 'start_pos' in self.inputs:
                    ort_sp = ort.OrtValue.ortvalue_from_numpy(np.array([positions[0, -1]], dtype=np.int64), 'cuda', 0)
                    io_binding.bind_ortvalue_input('start_pos', ort_sp)
                    
                if 'mask' in self.inputs:
                    mask_val = np.empty((0,), dtype=self.dtype) if mask is None else mask.astype(self.dtype)
                    ort_mask = ort.OrtValue.ortvalue_from_numpy(mask_val, 'cuda', 0)
                    io_binding.bind_ortvalue_input('mask', ort_mask)
                    
                for i in range(self.num_layers):
                    k_state, v_state = kv_cache.states[i]
                    
                    if isinstance(k_state, np.ndarray):
                        k_state = ort.OrtValue.ortvalue_from_numpy(k_state.astype(self.dtype), 'cuda', 0)
                    if isinstance(v_state, np.ndarray):
                        v_state = ort.OrtValue.ortvalue_from_numpy(v_state.astype(self.dtype), 'cuda', 0)
                    
                    io_binding.bind_ortvalue_input(f'{self.k_prefix}{i}', k_state)
                    io_binding.bind_ortvalue_input(f'{self.v_prefix}{i}', v_state)
                    
                io_binding.bind_output('logits', 'cuda', 0)
                for name in self.output_names[1:]:
                    io_binding.bind_output(name, 'cuda', 0)
                    
                self.session.run_with_iobinding(io_binding)
                
                ort_outputs = io_binding.get_outputs()
                ort_logits = ort_outputs[0]
                new_kvs = ort_outputs[1:]
                
                logits_np = ort_logits.numpy()
                return logits_np, new_kvs
                
            except Exception:
                self.device = 'cpu'
                kv_cache.device = 'cpu'
                for idx in range(len(kv_cache.states)):
                    k_v, v_v = kv_cache.states[idx]
                    k_np = k_v.numpy() if hasattr(k_v, 'numpy') else k_v
                    v_np = v_v.numpy() if hasattr(v_v, 'numpy') else v_v
                    kv_cache.states[idx] = (k_np, v_np)
                return self.forward(input_ids, positions, kv_cache, mask)

        else:
            feeds = {}
            feeds['input_ids'] = input_ids.astype(np.int64)
            
            if 'positions' in self.inputs:
                feeds['positions'] = positions.astype(np.int64)
            elif 'start_pos' in self.inputs:
                feeds['start_pos'] = np.array([positions[0, -1]], dtype=np.int64)
            
            if 'mask' in self.inputs:
                feeds['mask'] = np.empty((0,), dtype=self.dtype) if mask is None else mask.astype(self.dtype)
                    
            for i in range(self.num_layers):
                k_state, v_state = kv_cache.states[i]
                feeds[f'{self.k_prefix}{i}'] = k_state
                feeds[f'{self.v_prefix}{i}'] = v_state

            outputs = self.session.run(None, feeds)
            return outputs[0], outputs[1:]