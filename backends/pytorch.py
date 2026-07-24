"""
PyTorch 后端实现
----------------
将 Backend 接口映射到 PyTorch 操作。

用于训练（GPU + 混合精度 + 梯度计算）。

设备选择策略（通过 src.device 统一管理）：
  - "cuda"  → 显式请求 GPU，不可用时按 CODESPRITE_ALLOW_CPU_FALLBACK 决定回退/报错
  - "cpu"   → 强制 CPU，自动限制线程数
  - "auto"  → 自动最优（CUDA > CPU）
"""

import math
import torch
import torch.nn.functional as F
from typing import Any, Tuple, Optional, Dict
from backends.base import Backend
from src.device import resolve_device


class PyTorchBackend(Backend):
    """PyTorch 计算后端"""

    name = "pytorch"

    def __init__(self, device: str = "auto", dtype=None):
        # 通过统一设备管理模块解析设备，承担回退策略和日志
        resolved = resolve_device(device)
        self.device = torch.device(resolved)
        self._resolved_device = resolved  # 保留字符串，供 print_device_info 等使用
        self.dtype = dtype or torch.float32

    # ============================================================
    # 张量工具
    # ============================================================

    def shape(self, x) -> Tuple[int, ...]:
        return tuple(x.shape)

    def add(self, a, b):
        return a + b

    def multiply(self, a, b):
        return a * b

    def matmul(self, a, b):
        return torch.matmul(a, b)

    def transpose(self, x, dim0: int, dim1: int):
        return x.transpose(dim0, dim1)

    def reshape(self, x, *shape):
        return x.reshape(*shape)

    def zeros_like(self, x):
        return torch.zeros_like(x)

    def ones_like(self, x):
        return torch.ones_like(x)

    def unsqueeze(self, x, dim: int):
        return x.unsqueeze(dim)

    def sqrt(self, x):
        return torch.sqrt(x)

    def log(self, x):
        return torch.log(x)

    def exp(self, x):
        return torch.exp(x)

    def argmax(self, x, dim: int = -1):
        return x.argmax(dim=dim)

    def to_numpy(self, x):
        return x.detach().cpu().numpy()

    # ============================================================
    # 神经网络基础
    # ============================================================

    def linear(self, x, weight, bias=None):
        return F.linear(x, weight, bias)

    def embedding(self, input_ids, weight):
        return F.embedding(input_ids, weight)

    def layer_norm(self, x, weight, bias, eps: float):
        return F.layer_norm(x.float(), (weight.shape[0],), weight, bias, eps).to(x.dtype)

    def rms_norm(self, x, weight, eps: float):
        # RMSNorm: x * weight / sqrt(mean(x^2) + eps)
        # 注意：这里要先转 float32 再算，不然 fp16 下 mean 可能溢出
        # 踩过坑：直接在 fp16 上算 x**2 然后 mean 会出 nan，改成 float() 就好了
        dtype = x.dtype
        x = x.float()
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + eps)
        return (x / rms * weight.float()).to(dtype)

    def dropout(self, x, p: float, training: bool):
        return F.dropout(x, p=p, training=training)

    # ============================================================
    # 激活函数
    # ============================================================

    def silu(self, x):
        return F.silu(x)

    def gelu(self, x):
        return F.gelu(x)

    def softmax(self, x, dim: int = -1):
        return F.softmax(x, dim=dim)

    # ============================================================
    # 注意力
    # ============================================================

    def reshape_for_heads(self, x, batch, seq, num_heads, head_dim):
        # (batch, seq, hidden) → (batch, num_heads, seq, head_dim)
        x = x.view(batch, seq, num_heads, head_dim)
        return x.transpose(1, 2)

    def reshape_from_heads(self, x, batch, seq, hidden):
        # (batch, num_heads, seq, head_dim) → (batch, seq, hidden)
        x = x.transpose(1, 2)
        return x.reshape(batch, seq, hidden)

    def scaled_dot_product_attention(
        self, q, k, v, mask=None, num_kv_heads=None,
        scale=None, dropout_p=0.0, training=True
    ):
        """
        缩放点积注意力（支持 GQA）

        Q: (batch, num_heads, seq_q, head_dim)
        K: (batch, num_kv_heads, seq_k, head_dim)
        V: (batch, num_kv_heads, seq_k, head_dim)

        # NOTE: 原来想直接用 F.scaled_dot_product_attention，但那个接口
        # 在 PyTorch < 2.0 上没有，考虑兼容性还是自己实现了
        # TODO: 版本检测 + 条件使用 flash attention
        """
        if scale is None:
            scale = q.size(-1) ** 0.5

        # GQA: 将 KV 头扩展到与 Q 头数量相同
        if num_kv_heads is not None and q.size(1) != k.size(1):
            # Repeat K/V heads: (batch, num_kv, seq, dim) → (batch, num_q, seq, dim)
            ratio = q.size(1) // k.size(1)
            if ratio > 1:
                k = k.repeat_interleave(ratio, dim=1)
                v = v.repeat_interleave(ratio, dim=1)

        # 缩放点积 + Softmax
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / scale

        if mask is not None:
            # mask shape: (seq, seq) → 扩展到 (1, 1, seq, seq)
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)
            attn_weights = attn_weights.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = F.dropout(attn_weights, p=dropout_p, training=training)

        return torch.matmul(attn_weights, v)

    def causal_mask(self, seq_len: int) -> Any:
        mask = torch.ones(seq_len, seq_len, device=self.device, dtype=torch.bool)
        return torch.tril(mask)

    def concat(self, a, b, dim: int = 0):
        """沿指定维度拼接两个张量（用于 KV-Cache 等场景）"""
        return torch.cat([a, b], dim=dim)

    # ============================================================
    # 位置编码
    # ============================================================

    def rope_precompute(self, seq_len: int, dim: int, theta: float) -> Tuple[Any, Any]:
        """预计算 RoPE cos/sin 表"""
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        t = torch.arange(seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos().to(self.device), emb.sin().to(self.device)

    def rope_apply(self, q, k, cos, sin) -> Tuple[Any, Any]:
        """对 Q/K 应用旋转位置编码"""
        seq_len = q.size(2)
        cos = cos[:seq_len].unsqueeze(0).unsqueeze(0)
        sin = sin[:seq_len].unsqueeze(0).unsqueeze(0)

        def rotate_half(x):
            x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
            return torch.cat([-x2, x1], dim=-1)

        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        return q_embed, k_embed

    # ============================================================
    # 损失函数
    # ============================================================

    def cross_entropy(self, logits, targets, ignore_index: int = -100,
                      label_smoothing: float = 0.0):
        return F.cross_entropy(
            logits, targets,
            ignore_index=ignore_index,
            label_smoothing=label_smoothing
        )

    # ============================================================
    # 训练专用
    # ============================================================

    def backward(self, loss) -> None:
        loss.backward()

    def step_optimizer(self, optimizer) -> None:
        optimizer.step()

    def zero_grad(self, optimizer) -> None:
        optimizer.zero_grad()

    def clip_grad_norm(self, parameters, max_norm: float) -> float:
        return torch.nn.utils.clip_grad_norm_(parameters, max_norm)

    # ============================================================
    # 权重管理
    # ============================================================

    def init_weight(self, shape: Tuple[int, ...], method: str = "xavier") -> Any:
        if method == "xavier":
            # Xavier/Glorot uniform
            # 原来直接用 nn.init.xavier_uniform_，但那个需要 nn.Parameter
            # 这里手动实现一下，逻辑是一样的
            fan_in = shape[0] if len(shape) >= 2 else shape[0]
            fan_out = shape[1] if len(shape) >= 2 else 1
            limit = math.sqrt(6.0 / (fan_in + fan_out)) if fan_in > 0 else 0.01
            return torch.empty(shape, device=self.device, dtype=self.dtype).uniform_(-limit, limit)
        elif method == "normal":
            return torch.randn(shape, device=self.device, dtype=self.dtype) * 0.02
        elif method == "zeros":
            return torch.zeros(shape, device=self.device, dtype=self.dtype)
        elif method == "ones":
            return torch.ones(shape, device=self.device, dtype=self.dtype)
        else:
            raise ValueError(f"未知初始化方法: {method}")

    def init_embedding(self, shape: Tuple[int, ...]) -> Any:
        """初始化 Embedding 权重（正态分布 * 0.02）"""
        return torch.randn(shape, device=self.device, dtype=self.dtype) * 0.02

    def create_parameter(self, shape: Tuple[int, ...], method: str = "xavier") -> Any:
        param = self.init_weight(shape, method)
        param.requires_grad = True
        return param

    def create_optimizer(self, parameters, lr: float, betas=(0.9, 0.999),
                         eps=1e-8, weight_decay=0.01, **kwargs):
        """创建 AdamW 优化器（参数分组）"""
        no_decay = {'bias', 'norm', 'rmsnorm', 'layernorm'}
        grouped = [
            {
                'params': [p for n, p in parameters if not any(nd in n.lower() for nd in no_decay)],
                'weight_decay': weight_decay
            },
            {
                'params': [p for n, p in parameters if any(nd in n.lower() for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]
        return torch.optim.AdamW(grouped, lr=lr, betas=betas, eps=eps, **kwargs)

    def save_checkpoint(self, model, path: str, extra: Dict = None) -> None:
        state = self.get_state_dict(model)
        checkpoint = {"state_dict": state, **(extra or {})}
        torch.save(checkpoint, path)

    def load_checkpoint(self, model, path: str) -> Dict:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        state_dict = checkpoint.get("state_dict", checkpoint)
        self.load_state_dict(model, state_dict)
        return checkpoint

    def get_state_dict(self, model) -> Dict[str, Any]:
        """从 IR 模型收集所有参数，flat key 格式"""
        from ir.utils import collect_params
        return collect_params(model)

    def load_state_dict(self, model, state_dict: Dict[str, Any]) -> None:
        """从 flat key 字典加载参数到 IR 模型"""
        from ir.utils import load_state_dict
        load_state_dict(model, state_dict)

    def multinomial(self, probs, num_samples: int = 1):
        return torch.multinomial(probs, num_samples=num_samples)


# ============================================================
# 模型初始化工具
# ============================================================

def init_model_weights(model, backend: PyTorchBackend):
    """
    使用 PyTorch 后端初始化 IR 模型的所有权重。

    遍历 model.param_shapes() 中定义的每个参数，
    为每个参数创建 tensor 并分配给对应的 Layer。

    Args:
        model: IR TransformerModel 实例
        backend: PyTorchBackend 实例
    """
    from ir.layers import Layer, Embedding, Linear, RMSNorm, LayerNormLayer

    def _init_recursive(obj):
        if isinstance(obj, Embedding):
            # Embedding: 正态分布初始化
            shape = (obj.vocab_size, obj.hidden_size)
            obj.params["weight"] = backend.init_embedding(shape)
        elif isinstance(obj, Linear):
            shape = (obj.out_features, obj.in_features)
            obj.params["weight"] = backend.create_parameter(shape, "xavier")
            if obj.has_bias:
                obj.params["bias"] = backend.init_weight((obj.out_features,), "zeros")
                obj.params["bias"].requires_grad = True
        elif isinstance(obj, RMSNorm):
            obj.params["weight"] = backend.init_weight((obj.normalized_shape,), "ones")
            obj.params["weight"].requires_grad = True
        elif isinstance(obj, LayerNormLayer):
            obj.params["weight"] = backend.init_weight((obj.normalized_shape,), "ones")
            obj.params["weight"].requires_grad = True
            obj.params["bias"] = backend.init_weight((obj.normalized_shape,), "zeros")
            obj.params["bias"].requires_grad = True
        elif isinstance(obj, Layer):
            # 递归子层
            for attr_name in dir(obj):
                if attr_name.startswith('_'):
                    continue
                try:
                    attr = getattr(obj, attr_name)
                    if isinstance(attr, Layer):
                        _init_recursive(attr)
                except:
                    pass
            if hasattr(obj, 'blocks'):
                for block in obj.blocks:
                    _init_recursive(block)

    _init_recursive(model)

    # Tie weights: LM head 共享 Embedding 权重
    if model.config.tie_weights:
        if "weight" in model.embedding.params:
            model.lm_head.params["weight"] = model.embedding.params["weight"]

    return model


def collect_parameters(model) -> list:
    """
    递归收集所有 requires_grad=True 的参数（用于优化器）

    Args:
        model: IR 模型

    Returns:
        (name, param) 元组列表
    """
    from ir.layers import Layer
    params = []

    def _collect(obj, prefix=""):
        if isinstance(obj, Layer):
            for key, param in obj.params.items():
                full_key = f"{prefix}{key}" if prefix else key
                if hasattr(param, 'requires_grad') and param.requires_grad:
                    params.append((full_key, param))
            for attr_name in dir(obj):
                if attr_name.startswith('_'):
                    continue
                try:
                    attr = getattr(obj, attr_name)
                    if isinstance(attr, Layer) and attr is not obj:
                        sub_prefix = f"{prefix}{attr.name}." if prefix else f"{attr.name}."
                        _collect(attr, sub_prefix)
                except:
                    pass
            if hasattr(obj, 'blocks'):
                for i, block in enumerate(obj.blocks):
                    sub_prefix = f"{prefix}block_{i}." if prefix else f"block_{i}."
                    _collect(block, sub_prefix)

    _collect(model)
    return params
