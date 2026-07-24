"""
NumPy 后端实现
--------------
将 Backend 接口映射到 NumPy 操作。

用于纯 CPU 推理（无需安装 PyTorch）。
不支持梯度计算和训练。
"""

import math
import numpy as np
from typing import Any, Tuple, Optional, Dict
from backends.base import Backend


class NumPyBackend(Backend):
    """NumPy 计算后端 — 纯 CPU 推理"""

    name = "numpy"

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)

    # ============================================================
    # 张量工具
    # ============================================================

    def shape(self, x) -> Tuple[int, ...]:
        return x.shape

    def add(self, a, b):
        return a + b

    def multiply(self, a, b):
        return a * b

    def matmul(self, a, b):
        return np.matmul(a, b)

    def transpose(self, x, dim0: int, dim1: int):
        return np.swapaxes(x, dim0, dim1)

    def reshape(self, x, *shape):
        return x.reshape(*shape)

    def zeros_like(self, x):
        return np.zeros_like(x)

    def ones_like(self, x):
        return np.ones_like(x)

    def unsqueeze(self, x, dim: int):
        return np.expand_dims(x, axis=dim)

    def sqrt(self, x):
        return np.sqrt(x)

    def log(self, x):
        return np.log(np.clip(x, 1e-10, None))

    def exp(self, x):
        return np.exp(np.clip(x, -50, 50))

    def argmax(self, x, dim: int = -1):
        return np.argmax(x, axis=dim)

    def to_numpy(self, x):
        return x

    # ============================================================
    # 神经网络基础
    # ============================================================

    def linear(self, x, weight, bias=None):
        # x: (..., in_features), weight: (out_features, in_features)
        result = x @ weight.T
        if bias is not None:
            result = result + bias
        return result

    def embedding(self, input_ids, weight):
        # input_ids: (batch, seq), weight: (vocab_size, hidden_size)
        return weight[input_ids]

    def layer_norm(self, x, weight, bias, eps: float):
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + eps)
        return x_norm * weight + bias

    def rms_norm(self, x, weight, eps: float):
        rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)
        return (x / rms) * weight

    def dropout(self, x, p: float, training: bool):
        # NumPy 推理模式：不做 dropout
        return x

    # ============================================================
    # 激活函数
    # ============================================================

    def silu(self, x):
        # SiLU(x) = x * sigmoid(x)
        return x * (1.0 / (1.0 + np.exp(-np.clip(x, -20, 20))))

    def gelu(self, x):
        # GELU 近似: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        cdf = 0.5 * (1.0 + np.tanh(
            math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)
        ))
        return x * cdf

    def softmax(self, x, dim: int = -1):
        # 数值稳定 softmax
        x_max = np.max(x, axis=dim, keepdims=True)
        e_x = np.exp(x - x_max)
        return e_x / np.sum(e_x, axis=dim, keepdims=True)

    # ============================================================
    # 注意力
    # ============================================================

    def reshape_for_heads(self, x, batch, seq, num_heads, head_dim):
        # (batch, seq, hidden) → (batch, num_heads, seq, head_dim)
        x = x.reshape(batch, seq, num_heads, head_dim)
        return np.swapaxes(x, 1, 2)

    def reshape_from_heads(self, x, batch, seq, hidden):
        # (batch, num_heads, seq, head_dim) → (batch, seq, hidden)
        x = np.swapaxes(x, 1, 2)
        return x.reshape(batch, seq, hidden)

    def scaled_dot_product_attention(
        self, q, k, v, mask=None, num_kv_heads=None,
        scale=None, dropout_p=0.0, training=True
    ):
        if scale is None:
            scale = q.shape[-1] ** 0.5

        # GQA: 扩展 KV 头
        if num_kv_heads is not None and q.shape[1] != k.shape[1]:
            ratio = q.shape[1] // k.shape[1]
            if ratio > 1:
                k = np.repeat(k, ratio, axis=1)
                v = np.repeat(v, ratio, axis=1)

        # 缩放点积
        attn_scores = np.matmul(q, k.swapaxes(-2, -1)) / scale

        if mask is not None:
            if mask.ndim == 2:
                mask = mask[np.newaxis, np.newaxis, :, :]
            elif mask.ndim == 3:
                mask = mask[:, np.newaxis, :, :]
            attn_scores = np.where(~mask, -1e10, attn_scores)

        attn_weights = self.softmax(attn_scores, dim=-1)
        # NumPy 不做 dropout
        return np.matmul(attn_weights, v)

    def causal_mask(self, seq_len: int) -> Any:
        return np.tril(np.ones((seq_len, seq_len), dtype=bool))

    def concat(self, a, b, dim: int = 0):
        """沿指定维度拼接两个数组（用于 KV-Cache 等场景）"""
        return np.concatenate([a, b], axis=dim)

    # ============================================================
    # 位置编码
    # ============================================================

    def rope_precompute(self, seq_len: int, dim: int, theta: float) -> Tuple[Any, Any]:
        inv_freq = 1.0 / (theta ** (np.arange(0, dim, 2, dtype=np.float32) / dim))
        t = np.arange(seq_len, dtype=np.float32)
        freqs = np.outer(t, inv_freq)
        emb = np.concatenate([freqs, freqs], axis=-1)
        return np.cos(emb), np.sin(emb)

    def rope_apply(self, q, k, cos, sin) -> Tuple[Any, Any]:
        seq_len = q.shape[2]
        cos = cos[:seq_len][np.newaxis, np.newaxis, :, :]
        sin = sin[:seq_len][np.newaxis, np.newaxis, :, :]

        def rotate_half(x):
            half = x.shape[-1] // 2
            x1, x2 = x[..., :half], x[..., half:]
            return np.concatenate([-x2, x1], axis=-1)

        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        return q_embed, k_embed

    # ============================================================
    # 损失函数
    # ============================================================

    def cross_entropy(self, logits, targets, ignore_index: int = -100,
                      label_smoothing: float = 0.0):
        # NumPy 推理不计算损失
        return np.array(0.0)

    # ============================================================
    # 权重管理
    # ============================================================

    def init_weight(self, shape: Tuple[int, ...], method: str = "xavier") -> Any:
        if method == "xavier":
            fan_in = shape[0] if len(shape) >= 2 else shape[0]
            fan_out = shape[1] if len(shape) >= 2 else 1
            limit = math.sqrt(6.0 / (fan_in + fan_out)) if fan_in > 0 else 0.01
            return self.rng.uniform(-limit, limit, shape).astype(np.float32)
        elif method == "normal":
            return (self.rng.randn(*shape) * 0.02).astype(np.float32)
        elif method == "zeros":
            return np.zeros(shape, dtype=np.float32)
        elif method == "ones":
            return np.ones(shape, dtype=np.float32)
        else:
            raise ValueError(f"未知初始化方法: {method}")

    def create_parameter(self, shape: Tuple[int, ...], method: str = "xavier") -> Any:
        return self.init_weight(shape, method)

    def create_optimizer(self, parameters, lr: float, **kwargs) -> Any:
        return None  # NumPy 不支持训练

    def save_checkpoint(self, model, path: str, extra: Dict = None) -> None:
        state = self.get_state_dict(model)
        checkpoint = {"state_dict": state, **(extra or {})}
        np.savez_compressed(path, **checkpoint)

    def load_checkpoint(self, model, path: str) -> Dict:
        if path.endswith('.npz'):
            data = np.load(path, allow_pickle=True)
            state_dict = dict(data)
            # np.savez 会把 dict 扁平化，这里做了简单处理
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict'].item()
        elif path.endswith('.pt'):
            # 从 PyTorch 检查点加载（需要转换）
            import torch
            checkpoint = torch.load(path, map_location='cpu', weights_only=False)
            state_dict = checkpoint.get('state_dict', checkpoint)
            # 转换 torch tensors → numpy
            state_dict = {k: v.numpy() if hasattr(v, 'numpy') else v
                         for k, v in state_dict.items()}
        else:
            raise ValueError(f"不支持的文件格式: {path}")
        self.load_state_dict(model, state_dict)
        return state_dict

    def get_state_dict(self, model) -> Dict[str, Any]:
        from ir.utils import collect_params
        return collect_params(model)

    def load_state_dict(self, model, state_dict: Dict[str, Any]) -> None:
        from ir.utils import load_state_dict

        def _convert_fn(key, value):
            return np.asarray(value, dtype=np.float32)

        load_state_dict(model, state_dict, transform_fn=_convert_fn)

    def multinomial(self, probs, num_samples: int = 1):
        # 从概率分布中采样
        flat_probs = probs.reshape(-1)
        flat_probs = flat_probs / flat_probs.sum()  # 确保和为 1
        indices = self.rng.choice(len(flat_probs), size=num_samples, p=flat_probs)
        return indices.reshape(probs.shape[:-1] + (num_samples,))


def convert_torch_to_numpy(model):
    """
    将 PyTorch 张量权重转换为 NumPy 数组（用于跨后端切换）

    Args:
        model: IR 模型（参数为 torch tensors）

    Returns:
        模型（参数已转换为 numpy arrays）
    """
    from ir.layers import Layer

    def _convert(obj):
        if isinstance(obj, Layer):
            for key, param in obj.params.items():
                if param is not None and hasattr(param, 'numpy'):
                    obj.params[key] = param.detach().cpu().numpy()
            for attr_name in dir(obj):
                if attr_name.startswith('_'):
                    continue
                try:
                    attr = getattr(obj, attr_name)
                    if isinstance(attr, Layer) and attr is not obj:
                        _convert(attr)
                except:
                    pass
            if hasattr(obj, 'blocks'):
                for block in obj.blocks:
                    _convert(block)

    _convert(model)
    return model
