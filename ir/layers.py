"""
IR 层 — 抽象层定义

定义神经网络层的结构（维度和元数据），不包含任何计算逻辑。
所有计算由 backends/ 中的后端完成。

设计原则：
  1. 每个层只存"形状信息"和"元数据"，不存具体数值张量
  2. 层本身不执行计算 — forward() 接受 backend 参数，委托给后端
  3. 零框架导入 — 不 import torch / numpy

# TODO: 以后应该加一个 __repr__ 方法把形状也打出来，调试时比较方便
# TODO: Sequential 目前比较简陋，考虑支持 named layers（像 nn.ModuleDict 那样）
"""

from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass


# ============================================================
# Layer 基类
# ============================================================

class Layer:
    """所有层的抽象基类"""

    def __init__(self, name: str = ""):
        self.name = name
        self.params: Dict[str, Any] = {}        # 权重张量（由后端创建）
        self.buffers: Dict[str, Any] = {}       # 非训练参数（如 RoPE 缓存）
        self._training: bool = True

    def train(self, mode: bool = True):
        """递归切换训练/推理模式（包括所有子层）"""
        self._training = mode
        for attr_name in dir(self):
            if attr_name.startswith('_'):
                continue
            try:
                attr = getattr(self, attr_name)
                if isinstance(attr, Layer):
                    attr.train(mode)
            except AttributeError:
                pass
        if hasattr(self, 'blocks'):
            for block in self.blocks:
                if isinstance(block, Layer):
                    block.train(mode)
        return self

    def eval(self):
        """切换到推理模式"""
        return self.train(False)

    @property
    def training(self) -> bool:
        return self._training

    def forward(self, x, backend, **kwargs):
        """前向传播 — 子类必须实现"""
        raise NotImplementedError(f"{self.__class__.__name__}.forward() 未实现")

    def param_shapes(self) -> Dict[str, Tuple[int, ...]]:
        """返回所有参数的形状（用于后端初始化权重）"""
        return {}

    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}')"


# ============================================================
# Embedding
# ============================================================

class Embedding(Layer):
    """Token 嵌入层"""

    def __init__(self, vocab_size: int, hidden_size: int, name: str = "embedding"):
        super().__init__(name)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

    def param_shapes(self) -> Dict[str, Tuple[int, ...]]:
        return {"weight": (self.vocab_size, self.hidden_size)}

    def forward(self, x, backend, **kwargs):
        return backend.embedding(x, self.params["weight"])


# ============================================================
# Linear
# ============================================================

class Linear(Layer):
    """线性全连接层"""

    def __init__(self, in_features: int, out_features: int,
                 bias: bool = False, name: str = "linear"):
        super().__init__(name)
        self.in_features = in_features
        self.out_features = out_features
        self.has_bias = bias

    def param_shapes(self) -> Dict[str, Tuple[int, ...]]:
        shapes = {"weight": (self.out_features, self.in_features)}
        if self.has_bias:
            shapes["bias"] = (self.out_features,)
        return shapes

    def forward(self, x, backend, **kwargs):
        bias = self.params.get("bias", None)
        return backend.linear(x, self.params["weight"], bias)


# ============================================================
# LayerNorm (传统 LayerNorm)
# ============================================================

class LayerNormLayer(Layer):
    """Layer Normalization"""

    def __init__(self, normalized_shape: int, eps: float = 1e-5, name: str = "layernorm"):
        super().__init__(name)
        self.normalized_shape = normalized_shape
        self.eps = eps

    def param_shapes(self) -> Dict[str, Tuple[int, ...]]:
        return {
            "weight": (self.normalized_shape,),
            "bias": (self.normalized_shape,),
        }

    def forward(self, x, backend, **kwargs):
        return backend.layer_norm(
            x, self.params["weight"], self.params["bias"], self.eps
        )


# ============================================================
# RMSNorm (LLaMA 风格)
# ============================================================

class RMSNorm(Layer):
    """RMS Normalization — LLaMA 风格的归一化"""

    def __init__(self, normalized_shape: int, eps: float = 1e-6, name: str = "rmsnorm"):
        super().__init__(name)
        self.normalized_shape = normalized_shape
        self.eps = eps

    def param_shapes(self) -> Dict[str, Tuple[int, ...]]:
        return {"weight": (self.normalized_shape,)}

    def forward(self, x, backend, **kwargs):
        return backend.rms_norm(x, self.params["weight"], self.eps)


# ============================================================
# Dropout
# ============================================================

class DropoutLayer(Layer):
    """Dropout 正则化"""

    def __init__(self, p: float = 0.1, name: str = "dropout"):
        super().__init__(name)
        self.p = p

    def forward(self, x, backend, **kwargs):
        return backend.dropout(x, self.p, self.training)


# ============================================================
# RoPE (旋转位置编码 — 元数据仅存参数)
# ============================================================

class RoPELayer(Layer):
    """旋转位置编码（仅存参数，计算由后端完成）"""

    def __init__(self, dim: int, max_seq_length: int = 512,
                 theta: float = 10000.0, name: str = "rope"):
        super().__init__(name)
        self.dim = dim
        self.max_seq_length = max_seq_length
        self.theta = theta

    def forward(self, q, k, seq_pos: int, backend, **kwargs):
        """对 Q 和 K 应用旋转位置编码"""
        cos, sin = backend.rope_precompute(seq_pos, self.dim, self.theta)
        return backend.rope_apply(q, k, cos, sin)


# ============================================================
# Attention (多头自注意力)
# ============================================================

class Attention(Layer):
    """
    多头自注意力层（支持 GQA — 分组查询注意力）

    结构:
      - Q 投影: hidden_size -> hidden_size (num_heads * head_dim)
      - K 投影: hidden_size -> kv_size   (num_kv_heads * head_dim)
      - V 投影: hidden_size -> kv_size
      - O 投影: hidden_size -> hidden_size

    # NOTE: 一开始 kv_size 我写错了，以为是 hidden_size，结果 GQA 跑不起来
    # 后来对着 LLaMA 源码检查了一遍才发现问题
    """

    def __init__(self, hidden_size: int, num_heads: int,
                 num_kv_heads: int = 0, use_rope: bool = True,
                 use_bias: bool = False, name: str = "attention"):
        super().__init__(name)
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads > 0 else num_heads
        self.head_dim = hidden_size // num_heads
        self.use_rope = use_rope
        self.use_bias = use_bias

        # GQA: KV 维度 = num_kv_heads * head_dim
        self.kv_size = self.num_kv_heads * self.head_dim

        # 子层（Linear 投影）
        self.q_proj = Linear(hidden_size, hidden_size, bias=use_bias, name="q_proj")
        self.k_proj = Linear(hidden_size, self.kv_size, bias=use_bias, name="k_proj")
        self.v_proj = Linear(hidden_size, self.kv_size, bias=use_bias, name="v_proj")
        self.o_proj = Linear(hidden_size, hidden_size, bias=use_bias, name="o_proj")

    def param_shapes(self) -> Dict[str, Tuple[int, ...]]:
        shapes = {}
        for proj in [self.q_proj, self.k_proj, self.v_proj, self.o_proj]:
            for k, v in proj.param_shapes().items():
                shapes[f"{proj.name}.{k}"] = v
        return shapes

    def forward(self, x, backend, mask=None, kv_cache=None, use_cache=False, **kwargs):
        """
        x: (batch, seq_len, hidden_size)

        KV-Cache 支持:
          - kv_cache: 上一次的 (k_cache, v_cache) 元组，shape 均为
            (batch, num_kv_heads, total_seq_len, head_dim)
          - use_cache: 是否返回更新后的 KV-Cache
          - 返回: 若 use_cache=False → output
                  若 use_cache=True  → (output, new_kv_cache)
        """
        batch_size, seq_len, _ = backend.shape(x)

        # Q/K/V 投影
        q = self.q_proj.forward(x, backend)
        k = self.k_proj.forward(x, backend)
        v = self.v_proj.forward(x, backend)

        # Reshape 为多头
        q = backend.reshape_for_heads(q, batch_size, seq_len, self.num_heads, self.head_dim)
        k = backend.reshape_for_heads(k, batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = backend.reshape_for_heads(v, batch_size, seq_len, self.num_kv_heads, self.head_dim)

        # 应用 RoPE 旋转位置编码
        if self.use_rope:
            past_len = kv_cache[0].shape[2] if kv_cache is not None else 0
            cos, sin = backend.rope_precompute(past_len + seq_len, self.head_dim, 10000.0)
            cos = cos[past_len:past_len + seq_len]
            sin = sin[past_len:past_len + seq_len]
            q, k = backend.rope_apply(q, k, cos, sin)

        # KV-Cache: 拼接历史 K/V（沿序列维度 dim=2）
        if kv_cache is not None:
            past_k, past_v = kv_cache
            k = backend.concat(past_k, k, dim=2)
            v = backend.concat(past_v, v, dim=2)

        new_kv_cache = (k, v) if use_cache else None

        # 缩放点积注意力
        attn_output = backend.scaled_dot_product_attention(
            q, k, v, mask=mask,
            num_kv_heads=self.num_kv_heads,
            scale=self.head_dim ** 0.5,
            dropout_p=0.0 if not self.training else 0.1,
            training=self.training
        )

        # 合并多头 → O 投影
        attn_output = backend.reshape_from_heads(attn_output, batch_size, seq_len, self.hidden_size)
        output = self.o_proj.forward(attn_output, backend)

        if use_cache:
            return output, new_kv_cache
        return output


# ============================================================
# FeedForward (SwiGLU / GELU)
# ============================================================

class FeedForward(Layer):
    """
    前馈网络（支持 SwiGLU 和 GELU）

    SwiGLU: x → gate(Wg·x) ⊙ W1·x → W2·(结果)
    GELU:   x → W1·x → GELU → W2·(结果)
    """

    def __init__(self, hidden_size: int, intermediate_size: int,
                 activation: str = "swiglu", use_bias: bool = False,
                 name: str = "ffn"):
        super().__init__(name)
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.activation = activation
        self.use_bias = use_bias

        if activation == "swiglu":
            self.w1 = Linear(hidden_size, intermediate_size, bias=use_bias, name="w1")
            self.wg = Linear(hidden_size, intermediate_size, bias=use_bias, name="wg")
        else:
            self.w1 = Linear(hidden_size, intermediate_size, bias=use_bias, name="w1")

        self.w2 = Linear(intermediate_size, hidden_size, bias=use_bias, name="w2")

    def param_shapes(self) -> Dict[str, Tuple[int, ...]]:
        shapes = {}
        for k, v in self.w1.param_shapes().items():
            shapes[f"w1.{k}"] = v
        for k, v in self.w2.param_shapes().items():
            shapes[f"w2.{k}"] = v
        if self.activation == "swiglu":
            for k, v in self.wg.param_shapes().items():
                shapes[f"wg.{k}"] = v
        return shapes

    def forward(self, x, backend, **kwargs):
        if self.activation == "swiglu":
            # gate(Wg·x) ⊙ W1·x → W2
            gate = backend.silu(self.wg.forward(x, backend))
            hidden = self.w1.forward(x, backend)
            hidden = backend.multiply(gate, hidden)
        else:
            # W1·x → GELU → W2
            hidden = self.w1.forward(x, backend)
            hidden = backend.gelu(hidden)

        return self.w2.forward(hidden, backend)


# ============================================================
# TransformerBlock (一个完整的 Decoder Block)
# ============================================================

class TransformerBlock(Layer):
    """
    Pre-Norm Transformer Block

    结构:
      x → RMSNorm → Attention (+ residual)
        → RMSNorm → FeedForward (+ residual)
    """

    def __init__(self, config: "ModelConfig", name: str = "block"):
        super().__init__(name)
        self.config = config

        # Pre-attention norm
        self.attn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps,
                                 name="attn_norm")
        # Self-attention
        self.attention = Attention(
            config.hidden_size, config.num_heads,
            num_kv_heads=config.num_kv_heads,
            use_rope=config.use_rope,
            use_bias=config.use_bias,
            name="attn"
        )
        # Pre-FFN norm
        self.ffn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps,
                                name="ffn_norm")
        # Feed-forward
        self.feed_forward = FeedForward(
            config.hidden_size, config.intermediate_size,
            activation=config.activation,
            use_bias=config.use_bias,
            name="ffn"
        )
        # Dropout
        self.dropout = DropoutLayer(p=config.dropout, name="dropout")

    def param_shapes(self) -> Dict[str, Tuple[int, ...]]:
        shapes = {}
        for sub in [self.attn_norm, self.attention, self.ffn_norm, self.feed_forward]:
            for k, v in sub.param_shapes().items():
                shapes[f"{sub.name}.{k}"] = v
        return shapes

    def forward(self, x, backend, mask=None, kv_cache=None, use_cache=False, **kwargs):
        # Pre-Norm + Attention + Residual
        residual = x
        x_norm = self.attn_norm.forward(x, backend)
        x = self.attention.forward(x_norm, backend, mask=mask,
                                   kv_cache=kv_cache, use_cache=use_cache)

        new_kv_cache = None
        if use_cache and isinstance(x, tuple):
            x, new_kv_cache = x

        x = self.dropout.forward(x, backend)
        x = backend.add(residual, x)

        # Pre-Norm + FFN + Residual
        residual = x
        x_norm = self.ffn_norm.forward(x, backend)
        x = self.feed_forward.forward(x_norm, backend)
        x = self.dropout.forward(x, backend)
        x = backend.add(residual, x)

        if use_cache:
            return x, new_kv_cache
        return x


# ============================================================
# Sequential (顺序容器)
# ============================================================

class Sequential(Layer):
    """顺序层容器"""

    def __init__(self, layers: List[Layer], name: str = "sequential"):
        super().__init__(name)
        self.layers = layers

    def forward(self, x, backend, **kwargs):
        for layer in self.layers:
            x = layer.forward(x, backend, **kwargs)
        return x

    def param_shapes(self) -> Dict[str, Tuple[int, ...]]:
        shapes = {}
        for layer in self.layers:
            for k, v in layer.param_shapes().items():
                shapes[f"{layer.name}.{k}"] = v
        return shapes
