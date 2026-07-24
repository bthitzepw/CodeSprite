# CodeSprite 代码 Wiki

> **CodeSprite** 是一个**代码结构分析与转换工具**，基于框架无关的 IR（Intermediate Representation，中间表示）架构。核心分析层（`ir/`）不依赖任何计算框架，计算由可插拔的 `backends/` 后端提供。
>
> 同一套分析逻辑可以：用 PyTorch 训练，用 NumPy 纯 CPU 推理，导出为 GGUF/ONNX 格式。

---

## 目录

1. [项目概览](#1-项目概览)
2. [整体架构](#2-整体架构)
3. [目录结构](#3-目录结构)
4. [核心模块详解](#4-核心模块详解)
   - [4.1 IR 层（`ir/`）](#41-ir-层ir)
   - [4.2 计算后端（`backends/`）](#42-计算后端backends)
   - [4.3 训练模块（`training/`）](#43-训练模块training)
   - [4.4 推理模块（`inference/`）](#44-推理模块inference)
   - [4.5 导出模块（`export/`）](#45-导出模块export)
   - [4.6 评估模块（`eval/`）](#46-评估模块eval)
   - [4.7 工具集（`tools/`）](#47-工具集tools)
   - [4.8 业务组件（`src/`）](#48-业务组件src)
   - [4.9 通用工具（`utils/`）](#49-通用工具utils)
5. [关键类与函数说明](#5-关键类与函数说明)
6. [依赖关系](#6-依赖关系)
7. [配置系统](#7-配置系统)
8. [项目运行方式](#8-项目运行方式)
9. [测试与质量保证](#9-测试与质量保证)
10. [附录](#10-附录)

---

## 1. 项目概览

| 项目 | 说明 |
|------|------|
| 项目名称 | CodeSprite |
| 定位 | 框架无关的代码结构分析与转换工具 |
| 模型规模 | 约 3800 万参数（默认配置） |
| 模型架构 | Decoder-only Transformer（LLaMA 风格） |
| 核心特性 | 零框架依赖 IR + 多后端可插拔 + KV-Cache 推理加速 |
| 训练支持 | PyTorch（GPU/CPU，自动回退） |
| 推理支持 | PyTorch / NumPy（纯 CPU） |
| 导出格式 | GGUF（llama.cpp 兼容）、ONNX |
| 编程语言 | Python 3.10+ |
| 许可证 | MIT |

### 1.1 关键能力

- **代码生成 / 代码补全 / 代码解释 / 中文注释理解**
- **支持语言**：Python / JavaScript / Java / C++ / Go / Rust / TypeScript / SQL / Shell / HTML / CSS + 中文
- **现代训练技术**：RoPE 旋转位置编码、SwiGLU 门控前馈网络、GQA 分组查询注意力、KV-Cache、混合精度（AMP）、标签平滑、EMA、梯度检查点
- **自动学习系统**：用户反馈收集 → 数据增强 → 增量训练
- **Web 服务**：Flask 实现，含安全加固（CSP/HSTS/速率限制/审计日志）

### 1.2 设计原则

1. **框架无关（Framework-Agnostic）**：核心分析逻辑与计算框架完全解耦，`ir/` 不 `import torch/numpy`，只描述"有哪些层、长什么样"。
2. **后端可插拔（Pluggable Backends）**：所有计算通过抽象 `Backend` 接口委托给具体实现（PyTorch、NumPy）。
3. **多端点部署**：同一份代码可训练、推理、导出、嵌入 Web 服务。
4. **隐私与合规**：内置内容审核、合规审计、用户权利保障。

---

## 2. 整体架构

```
                ┌──────────────────────────┐
                │     ir/ (核心分析层)       │
                │   零框架依赖 · 纯结构描述    │
                └─────────────┬────────────┘
                              │ delegate
            ┌─────────────────┼─────────────────┐
            ↓                 ↓                 ↓
     ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
     │ PyTorch后端   │  │  NumPy后端    │  │  (预留) MLX  │
     │ 训练 + GPU推理 │  │  纯 CPU 推理   │  │  Apple 芯片  │
     └──────┬───────┘  └──────┬───────┘  └──────────────┘
            ↓                 ↓
     ┌──────────────┐  ┌──────────────┐
     │ GGUF 导出     │  │ ONNX 导出    │
     │ llama.cpp    │  │ ONNX Runtime │
     └──────────────┘  └──────────────┘
```

### 2.1 模型内部结构

```
Transformer Decoder (LLaMA 风格)

┌──────────────────────────────────┐
│         Token Embedding           │
│         Dropout                   │
├──────────────────────────────────┤
│   TransformerBlock × 8           │
│   ├── RMSNorm (Pre-Norm)         │
│   ├── Self-Attention (GQA)       │
│   │   ├── RoPE Position Encoding │
│   │   ├── Rotating KV-Cache      │
│   │   └── Causal Masking         │
│   ├── Residual                   │
│   ├── RMSNorm (Pre-Norm)         │
│   ├── SwiGLU FeedForward         │
│   └── Residual                   │
├──────────────────────────────────┤
│   Final RMSNorm                  │
│   LM Head (Tied Weights 可选)     │
└──────────────────────────────────┘

Parameters: ~37.9M | Hidden: 512 | Heads: 8 | Layers: 8
Vocab: 4268 | Context: 512 tokens | Activation: SwiGLU
Position: RoPE | Norm: RMSNorm | Attention: GQA
```

### 2.2 端到端数据流

```
训练数据 (data/raw/*.txt)
    ↓
SimpleTokenizer / CodeTokenizer (src/tokenizer.py)
    ↓ token ids
TextDataset + DataLoader
    ↓ batches
Trainer (training/trainer.py)
    ↓ logits
Loss (CrossEntropy + label smoothing)
    ↓ backward
PyTorchBackend (backends/pytorch.py)  ← 计算 + 自动求导
    ↓ weights
Checkpoint (checkpoints/*.pt)
    ↓ load
InferenceEngine (inference/engine.py)  ← KV-Cache 加速
    ↓ text
Web / CLI / 导出 (GGUF/ONNX)
```

---

## 3. 目录结构

```
codesprite/
├── ir/                          # 核心分析层（零框架依赖）
│   ├── __init__.py
│   ├── config.py                # ModelConfig/TrainingConfig 等数据类
│   ├── layers.py                # 抽象层（Linear/Attention/FFN 等）
│   ├── transformer.py           # TransformerModel 完整结构
│   ├── analysis.py              # CFG / DFG / ModuleGraph 静态分析
│   └── semantic.py              # 类型/作用域/调用图语义分析
│
├── backends/                    # 计算后端实现
│   ├── __init__.py
│   ├── base.py                  # Backend 抽象接口
│   ├── pytorch.py               # PyTorch 后端（训练+GPU推理）
│   ├── numpy.py                 # NumPy 后端（纯CPU推理）
│   └── spec.py                  # 后端能力规范与兼容矩阵
│
├── training/                    # 训练模块
│   ├── __init__.py
│   ├── trainer.py               # 后端无关训练器
│   └── optimizer.py             # 优化器工具（AdamW + CosineAnnealingWithWarmup）
│
├── inference/                   # 推理接口
│   ├── __init__.py
│   └── engine.py                # 推理引擎（自动选后端 + KV-Cache）
│
├── export/                      # 跨平台导出
│   ├── __init__.py
│   ├── gguf.py                  # GGUF 格式导出（llama.cpp 兼容）
│   └── onnx.py                  # ONNX 格式导出
│
├── eval/                        # 评估模块
│   ├── __init__.py
│   ├── benchmark.py             # 代码补全基准测试
│   └── metrics.py               # 评估指标（perplexity / syntax / edit distance）
│
├── tools/                       # 工具脚本
│   ├── build_nuitka.py          # Nuitka 打包脚本（Windows 64-bit）
│   ├── convert_checkpoint.py    # 旧权重 → 新 IR 格式
│   ├── data_cleaner.py          # 数据清洗（去重 / 质量过滤 / 语法校验）
│   ├── data_stats.py            # 数据统计报告
│   └── quantize.py              # 权重量化（INT8 / INT4）
│
├── src/                         # 业务组件
│   ├── __init__.py
│   ├── tokenizer.py             # SimpleTokenizer / CodeTokenizer / DataLoader
│   ├── device.py                # 统一设备管理（检测 / 回退 / 资源隔离）
│   ├── model.py                 # 旧版 PyTorch-native 模型（已废弃但保留兼容）
│   ├── moderator.py             # 内容审核（多法域策略）
│   ├── compliance.py            # 安全合规（CSP / HSTS / 速率限制 / 审计）
│   ├── auto_learner.py          # 自动增量学习（反馈收集 / 数据增强 / 后台训练）
│   └── trainer.py               # 旧版训练器（保留兼容）
│
├── utils/                       # 通用工具
│   ├── __init__.py
│   ├── errors.py                # 自定义异常体系
│   └── logger.py                # 结构化日志
│
├── config/
│   └── config.yaml              # 模型 / 训练 / 数据 / 系统配置
│
├── data/
│   └── raw/                     # 训练数据
│       ├── train.txt
│       ├── val.txt
│       └── test.txt
│
├── templates/                   # Flask Web 前端模板
│   ├── index.html
│   ├── privacy.html
│   └── agreement.html
│
├── tests/                       # 单元测试
│   ├── test_backends.py
│   ├── test_ir.py
│   ├── test_tokenizer.py
│   └── test_training_smoke.py
│
├── train.py                     # 训练入口
├── generate.py                  # 交互式分析入口
├── evaluate.py                  # 模型评估入口
├── web_app.py                   # Flask Web 服务入口
├── start_web.sh                 # 启动脚本 (Linux/macOS)
├── start_web.bat                # 启动脚本 (Windows)
│
├── requirements.txt             # Python 依赖
├── README.md                    # 项目说明
├── ROADMAP.md                   # 路线图
├── RISKS.md                     # 风险登记册
├── INVISIBILITY.md              # 隐形架构设计规范
├── LICENSE                      # MIT
└── .github/                     # Issue / PR 模板
```

---

## 4. 核心模块详解

### 4.1 IR 层（`ir/`）

> **核心设计原则**：零框架依赖。`ir/` 中所有代码不 `import torch/numpy`，只描述"有什么层、长什么样"。所有数值计算通过 `backend` 参数委托。

#### 4.1.1 `ir/config.py` — 配置数据类

| 类 | 职责 |
|----|------|
| `ModelConfig` | 模型结构配置（vocab_size、hidden_size、num_layers、num_heads、num_kv_heads、intermediate_size、max_seq_length、dropout、rms_norm_eps、activation、use_rope、rope_theta、tie_weights、use_bias 等） |
| `TrainingConfig` | 训练超参（batch_size、learning_rate、num_epochs、warmup_steps、gradient_accumulation_steps、max_grad_norm、weight_decay、use_amp、label_smoothing、use_ema、ema_decay、early_stopping_patience 等） |
| `DataConfig` | 数据路径（train/val/test_file）与 DataLoader 参数 |
| `SystemConfig` | 运行时参数（device、cpu_threads、seed、precision、log_level、checkpoint_dir、log_dir） |
| `Config` | 顶层聚合类，支持 `from_yaml` / `to_yaml` / `merge_from_args` |

**关键方法**：

- `ModelConfig.from_yaml(config_dict)` — 从 YAML 字典构造
- `ModelConfig.to_dict()` — 序列化为字典
- `ModelConfig.head_dim` / `kv_head_dim` — 维度派生属性
- `Config.from_yaml(path)` — 从 YAML 文件加载完整配置
- `Config.merge_from_args(args)` — 命令行参数覆盖（处理 `--no-amp` / `--use-ema` / `--no-rope` 等 flag）

**参数校验**：`__post_init__` 强制 `hidden_size % num_heads == 0`、`num_heads % num_kv_heads == 0` 等。

#### 4.1.2 `ir/layers.py` — 抽象层定义

所有层都继承自 `Layer` 基类（统一 `name` 命名 + 权重注册）。

| 组件 | 职责 |
|------|------|
| `Layer` | 抽象基类，统一权重命名、状态注册 |
| `Linear(in, out, bias)` | 全连接层（Y = XWᵀ + b） |
| `Embedding(vocab, dim)` | 词嵌入查找表 |
| `LayerNormLayer(dim, eps)` | 标准 LayerNorm |
| `RMSNorm(dim, eps)` | RMSNorm（LLaMA 风格，更快更稳） |
| `DropoutLayer(p)` | Dropout 正则化 |
| `RoPELayer(dim, theta, max_seq)` | 旋转位置编码（支持 KV-Cache 增量计算） |
| `Attention(hidden, num_heads, num_kv_heads, use_rope, use_bias)` | **多头自注意力 + GQA**：Q 投影到 `num_heads`，K/V 投影到 `num_kv_heads` |
| `FeedForward(hidden, intermediate, activation)` | 前馈网络，支持 `swiglu` / `gelu` / `silu` |
| `TransformerBlock(config)` | 单层 Transformer：RMSNorm → Attention → 残差 → RMSNorm → FFN → 残差 |
| `Sequential(*layers)` | 层容器 |

**Attention 关键逻辑**（`Attention.forward`）：

```python
q = self.q_proj(x)  # [B, L, hidden]
k = self.k_proj(x)  # [B, L, kv_size]
v = self.v_proj(x)  # [B, L, kv_size]
# reshape → [B, num_heads, L, head_dim]
if kv_cache is not None:
    k = backend.concat(past_k, k, dim=2)  # 增量更新
    v = backend.concat(past_v, v, dim=2)
attn = backend.scaled_dot_product_attention(q, k, v, mask, num_kv_heads, scale)
output = self.o_proj(attn)
return (output, (k, v)) if use_cache else output
```

**GQA 优势**：当 `num_kv_heads < num_heads` 时，多个 Q 头共享同一组 K/V，显著降低 KV-Cache 显存占用。

#### 4.1.3 `ir/transformer.py` — 完整 Transformer

**`TransformerModel`**：LLaMA 风格 Decoder-only Transformer。

```python
class TransformerModel(Layer):
    """
    Token Embedding
    → Dropout
    → N × TransformerBlock (RMSNorm + Attention + FFN)
    → Final RMSNorm
    → LM Head (Linear)
    """
    def __init__(self, config: ModelConfig, name: str = "codesprite"):
        self.embedding = Embedding(vocab_size, hidden_size)
        self.embed_dropout = DropoutLayer(p=dropout)
        self.blocks = [TransformerBlock(config) for _ in range(num_layers)]
        self.final_norm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.lm_head = Linear(hidden_size, vocab_size, bias=False)
        # 可选：tie_weights 共享 embedding ↔ lm_head 权重

    def forward(self, input_ids, backend, mask=None,
                past_key_values=None, use_cache=False, **kwargs):
        x = self.embedding(input_ids, backend)
        x = self.embed_dropout(x, backend)
        if mask is None and past_key_values is None:
            mask = backend.causal_mask(seq_len)
        new_kv = [] if use_cache else None
        for i, block in enumerate(self.blocks):
            layer_kv = past_key_values[i] if past_key_values else None
            if use_cache:
                x, layer_cache = block(x, backend, mask, layer_kv, use_cache=True)
                new_kv.append(layer_cache)
            else:
                x = block(x, backend, mask, layer_kv, use_cache=False)
        x = self.final_norm(x, backend)
        logits = self.lm_head(x, backend)
        return (logits, new_kv) if use_cache else logits
```

**关键能力**：

- **KV-Cache 推理**：通过 `past_key_values` + `use_cache=True` 复用历史 K/V，每步只需前向一个 token。
- **权重共享**：`tie_weights=True` 时 `lm_head.weight == embedding.weight`。
- **参数计数**：`get_param_count()` 返回总参数量（默认 ~37.9M）。

#### 4.1.4 `ir/analysis.py` — 静态分析

| 组件 | 职责 |
|------|------|
| `BlockKind` | 基本块类型枚举（Entry/Exit/Conditional/Loop/...） |
| `BasicBlock` | 基本块（语句列表 + 前驱/后继） |
| `CFGEdge` | 控制流图边（True/False/Fallthrough 标记） |
| `ControlFlowGraph` | 完整控制流图（构建于 AST 之上） |
| `DefUseChain` | 定义-使用链（变量在某点被定义、在哪些点被使用） |
| `DataFlowGraph` | 数据流图 |
| `ModuleImport` / `ModuleGraph` | 模块依赖图（import 关系） |

#### 4.1.5 `ir/semantic.py` — 语义分析

| 组件 | 职责 |
|------|------|
| `PrimitiveType` / `TypeCategory` / `TypeInfo` | 类型系统原语 |
| `Symbol` / `Scope` | 作用域内符号表 |
| `CallEdge` / `CallGraph` | 函数调用图（caller → callee） |
| `SemanticBlock` / `SemanticExtractor` | 语义块抽取器（基于 AST） |

---

### 4.2 计算后端（`backends/`）

#### 4.2.1 `backends/base.py` — Backend 抽象接口

```python
class Backend(ABC):
    name: str = "base"

    # 张量基础操作
    shape, add, multiply, matmul, linear, embedding

    # 注意力与位置编码
    scaled_dot_product_attention, rope_apply, causal_mask

    # 归一化、激活
    layer_norm, rms_norm, silu, gelu, swiglu, softmax, dropout

    # 损失函数
    cross_entropy_loss

    # 头维度变换
    reshape_for_heads, reshape_from_heads

    # 权重管理
    save_checkpoint(model, path, extra)
    load_checkpoint(model, path) → Dict
```

#### 4.2.2 `backends/pytorch.py` — PyTorch 后端

- **职责**：训练、GPU 推理、混合精度、autograd。
- **关键函数**：
  - `init_model_weights(model, backend)` — Xavier/Kaiming 初始化
  - `collect_parameters(model)` — 收集所有可训练参数 → 传给 PyTorch optimizer
  - `causal_mask(seq_len)` — 上三角 mask（`-inf`）
  - `save_checkpoint` / `load_checkpoint` — 序列化所有 `Layer` 的 `weight/bias`
- **设备管理**：构造时接收 `device="cuda"/"cpu"`，权重自动 `.to(device)`。

#### 4.2.3 `backends/numpy.py` — NumPy 后端

- **职责**：纯 CPU 推理，无需 PyTorch。
- **特性**：
  - 手写 RoPE（按位置增量计算 cos/sin）
  - 数值稳定 softmax（减 max）
  - 兼容 GQA 注意力 reshape
- **关键函数**：
  - `convert_torch_to_numpy(state_dict)` — 将 PyTorch 权重转换为 NumPy 数组
- **优势**：极致轻量、跨平台、易嵌入。

#### 4.2.4 `backends/spec.py` — 后端能力规范

- 定义各后端的能力矩阵（支持的层、激活、是否支持 AMP、KV-Cache 等）。
- 用于运行时能力检测，避免在不同后端上调用不支持的操作。

---

### 4.3 训练模块（`training/`）

#### 4.3.1 `training/trainer.py` — 后端无关训练器

**`Trainer` 类**封装完整训练循环。

| 组件 / 参数 | 说明 |
|------------|------|
| `__init__(model, train_loader, val_loader, backend, config, tokenizer)` | 构造训练器 |
| `use_amp` | 启用 FP16 混合精度（`torch.amp.autocast` + `GradScaler`） |
| `use_ema` | 启用指数移动平均（`ema_decay=0.999`） |
| `label_smoothing` | 标签平滑（0.05） |
| `early_stopping_patience` | 早停耐心值（5 个 epoch） |
| `gradient_accumulation_steps` | 梯度累积（4 步，等效 batch × 4） |
| `max_grad_norm` | 梯度裁剪（1.0） |

**核心方法**：

- `train()` — 主循环：每个 epoch 调 `_train_epoch` + `evaluate`，记录日志、保存 best_model.pt。
- `_train_epoch(epoch)` — 单 epoch 训练：前向 → 损失 → 反向 → 梯度累积 → 优化器步进 → EMA 更新。
- `evaluate()` — 验证：计算 val_loss、perplexity（`e^loss`）。
- `_compute_loss(logits, labels)` — 交叉熵 + 标签平滑。
- `_save_checkpoint(metrics)` — 保存检查点，保留 `save_total_limit` 个最近 epoch。

**支持的现代训练技术**：

- 混合精度（AMP / FP16）
- 标签平滑（Label Smoothing）
- 指数移动平均（EMA）
- 梯度累积（Gradient Accumulation）
- 梯度裁剪（Gradient Clipping）
- 余弦退火 + 预热（Cosine Annealing with Warmup）
- 早停（Early Stopping）

#### 4.3.2 `training/optimizer.py` — 优化器工厂

- `create_optimizer(model, config)` — 创建 AdamW 优化器
- `create_lr_scheduler(optimizer, config)` — CosineAnnealingWithWarmup 学习率调度
- 参数分组（weight_decay 应用于非 bias / 非 norm）

---

### 4.4 推理模块（`inference/`）

#### 4.4.1 `inference/engine.py` — 推理引擎

**`InferenceEngine` 类**统一推理接口。

**构造**：

```python
engine = InferenceEngine(
    model,                       # TransformerModel
    backend=None,                # PyTorch/NumPy，None 表示自动选
    checkpoint_path="...",       # 权重路径
    tokenizer=...,               # SimpleTokenizer
    device="auto"                # "cuda" / "cpu" / "auto"
)
```

**核心方法**：

| 方法 | 说明 |
|------|------|
| `_auto_select_backend(device)` | 优先 PyTorch（GPU/CPU），回退 NumPy（纯 CPU） |
| `load(checkpoint_path)` | 加载权重到 `model` |
| `generate(prompt, max_new_tokens, temperature, top_k, top_p)` | 基础生成（全量前向） |
| `generate_with_kv_cache(prompt, max_new_tokens, temperature, top_k, top_p)` | **KV-Cache 加速生成** |
| `_generate_ir_kv_cache(input_ids, ...)` | IR 模型的 KV-Cache 路径（首步全量，后续步只 1 token） |
| `_sample_token(logits, temperature, top_k, top_p)` | 采样（温度缩放 → top-k 截断 → top-p nucleus） |

**采样策略**：

- **Temperature**：logits / T（T→0 趋向 greedy，T=1 标准分布）
- **Top-K**：保留概率最大的 K 个 token
- **Top-P (Nucleus)**：保留累计概率 ≥ p 的最小 token 集合

**KV-Cache 工作流**：

```
第 1 步:  input = [t1, t2, ..., tN]
         → 全量前向，past_kv = None
         → 返回 (logits, [(K1, V1), (K2, V2), ..., (KL, VL)])

第 2 步:  input = [tN+1]
         past_kv = 上一步缓存
         → 只前向 1 个 token，复用 K/V
         → 返回 (logits, [...])  # 新缓存追加了 K_new

第 3..N 步: 同上
```

**自动降级**：若 IR KV-Cache 失败，自动 fallback 到全量生成。

---

### 4.5 导出模块（`export/`）

#### 4.5.1 `export/gguf.py` — GGUF 导出

- **用途**：将模型导出为 GGUF 格式，兼容 llama.cpp 推理引擎。
- **关键函数**：`export_gguf(model, output_path, tokenizer=None)`。
- **特性**：包含 metadata（架构、参数、张量名），可直接被 `llama.cpp` / `Ollama` / `LM Studio` 加载。

#### 4.5.2 `export/onnx.py` — ONNX 导出

- **用途**：导出为 ONNX 格式，可被 ONNX Runtime、TensorRT、OpenVINO 加速。
- **关键函数**：`export_onnx(model, output_path, sample_input)`。
- **注意**：当前导出对 dynamic axes 支持有限，建议使用固定 batch_size 导出。

---

### 4.6 评估模块（`eval/`）

#### 4.6.1 `eval/benchmark.py` — 代码补全基准

- 内置 8 个预设代码提示（Python / JS / SQL / Shell 等）
- 自动化评估生成质量：语法正确性、是否可编译、与参考代码的相似度
- 输出 PPL 等级（< 10 优秀 / < 30 良好 / < 100 一般 / > 500 需更多训练）

#### 4.6.2 `eval/metrics.py` — 评估指标

| 指标 | 公式 / 含义 |
|------|------------|
| **Perplexity (PPL)** | `e^loss`，越低越好 |
| **Token-level Loss** | 交叉熵损失（per token） |
| **Syntax Validity** | 通过 AST 解析的样本比例 |
| **Compile Rate** | 能被编译/解释执行的样本比例 |
| **Edit Distance** | 生成代码与参考代码的字符级编辑距离 |
| **CodeBLEU Proxy** | 简化版 CodeBLEU（综合 n-gram + AST + 数据流匹配） |

---

### 4.7 工具集（`tools/`）

| 工具 | 用途 |
|------|------|
| `tools/build_nuitka.py` | 使用 Nuitka 将 Python 项目打包成 Windows 64-bit 可执行文件（启动更快、避免 Python 环境依赖） |
| `tools/convert_checkpoint.py` | 旧版 `src/model.py` 权重 → 新版 IR 架构权重；用 `--old` / `--new` 指定路径 |
| `tools/data_cleaner.py` | 数据清洗：去重、长度过滤、语法校验、敏感词过滤 |
| `tools/data_stats.py` | 数据统计报告：样本数、平均长度、词表覆盖率、代码语言分布 |
| `tools/quantize.py` | 权重量化：INT8（per-tensor / per-channel）、INT4（GPTQ 简化版），并报告量化误差 |

---

### 4.8 业务组件（`src/`）

#### 4.8.1 `src/tokenizer.py` — 分词器与数据加载

| 类 | 职责 |
|----|------|
| `CodeTokenizer` | 字符级 BPE，专门为代码优化：<br>- 基础 ASCII（0-255）<br>- 代码高频符号（256-280）：`<INDENT>`、`<DEDENT>`、`<COMMENT>`、`<NEWLINE>`、`->` 等<br>- 特殊 token：`<PAD>` `<UNK>` `<BOS>` `<EOS>` `<MASK>`<br>- CJK 汉字（7019+）：GB2312 一级 3755 + 二级 3008 + 常用扩展 257 |
| `SimpleTokenizer` | 简化版（`vocab_size=4268`），开箱即用，无需预训练 |
| `TextDataset` | PyTorch `Dataset` 实现，按行读 `train.txt` / `val.txt` / `test.txt`，滑动窗口切分为 `max_seq_length` |
| `create_dataloader` | 创建 `DataLoader`，自动 pad / collate |

**特性**：

- **可选 PyTorch 依赖**：`try import torch` 优雅降级，纯推理场景无需安装 PyTorch。
- **代码感知**：缩进/反缩进显式 token，提升代码结构建模能力。
- **中英文混合**：UTF-8 字符级 fallback，CJK 直接进入词表。

#### 4.8.2 `src/device.py` — 统一设备管理

| 函数 | 职责 |
|------|------|
| `resolve_device(strategy, cpu_threads)` | 根据 `auto`/`cuda`/`cpu` 策略选择实际设备，处理 GPU 不可用回退 |
| `print_device_info(device)` | 打印设备信息（GPU 型号、显存、CPU 核心数） |
| `warn_cpu_training()` | CPU 训练时打印性能警告（速度约为 GPU 1/10-1/50） |
| `set_seed(seed)` | 设置 PyTorch / NumPy / Python 随机种子，确保可复现 |

**环境变量**：

- `CODESPRITE_ALLOW_CPU_FALLBACK=false` — 禁用 GPU 不可用时的静默回退。
- `CODESPRITE_WEB_DEVICE=cuda` — Web 服务默认 CPU，可强制 GPU。

#### 4.8.3 `src/moderator.py` — 内容审核

- **多法域策略**：根据地区法规（CN/EU/US）应用不同的关键词黑名单、敏感话题过滤。
- **用途**：Web 服务生成内容后置过滤，避免合规风险。

#### 4.8.4 `src/compliance.py` — 安全合规

| 组件 | 职责 |
|------|------|
| `SecurityHeaders` | 注入 CSP / HSTS / X-Frame-Options / X-Content-Type-Options 等安全响应头 |
| `RateLimiter` | 滑动窗口速率限制（默认 20 req / 60s） |
| `AuditLogger` | 审计日志（请求时间、IP、路径、用户 ID、生成内容） |
| `UserRights` | GDPR / CCPA 合规：数据导出、数据删除、用户同意管理 |

#### 4.8.5 `src/auto_learner.py` — 自动增量学习

**`AutoLearner` 类**实现闭环学习。

**工作流**：

```
用户提交 prompt → 生成 response → 用户点赞/点踩
    ↓
InteractionDB 记录 interaction + feedback
    ↓
满足 min_feedback_samples 阈值？
    ↓ 是
CodeAugmentor 数据增强（变量名替换 / 注释改写 / 同义结构）
    ↓
后台线程启动增量训练（小 LR / 短 epochs）
    ↓
更新 best_model.pt
```

**关键方法**：

- `record_feedback(interaction_id, feedback, comment)` — 记录反馈
- `_check_auto_trigger()` — 检查是否满足自动学习条件
- `start_learning(trigger_type, epochs, lr)` — 启动后台增量训练
- `_run_incremental_training(...)` — 使用 IR 架构执行训练

**配置参数**（`config.yaml` 的 `auto_learning` 段）：

```yaml
auto_learning:
  enabled: false
  min_feedback_samples: 20
  min_positive_ratio: 0.6
  incremental_epochs: 3
  incremental_lr: 0.00005
  max_augmented_samples: 500
  augmentation_enabled: true
  schedule_interval_hours: 24
```

#### 4.8.6 `src/model.py` — 旧版 PyTorch 模型（已废弃）

- 保留仅为向后兼容旧版检查点。
- 新代码应使用 `ir/transformer.py` + `TransformerModel`。
- 通过 `tools/convert_checkpoint.py` 转换旧权重。

#### 4.8.7 `src/trainer.py` — 旧版训练器（已废弃）

- 保留仅为向后兼容。
- 新代码应使用 `training/trainer.py`。

---

### 4.9 通用工具（`utils/`）

#### 4.9.1 `utils/logger.py` — 结构化日志

- 基于 `logging` 的统一日志配置。
- 支持控制台 + 文件双输出，按日期滚动。
- 日志级别：DEBUG / INFO / WARNING / ERROR。

#### 4.9.2 `utils/errors.py` — 自定义异常体系

- `CodeSpriteError` — 根异常
- `ConfigError` — 配置错误
- `BackendError` — 后端不兼容
- `TokenizerError` — 分词错误
- `TrainingError` — 训练过程错误
- `InferenceError` — 推理错误

---

## 5. 关键类与函数说明

### 5.1 顶层入口

| 文件 | 函数 | 职责 |
|------|------|------|
| `train.py` | `main()` | 训练入口：解析 CLI、加载配置、构建模型、训练、评估 |
| `generate.py` | `main()` | 交互式生成入口：单次 `--prompt` 或交互 REPL |
| `evaluate.py` | `main()` | 模型评估入口：计算 PPL / loss |
| `web_app.py` | Flask app | Web 服务入口：注册路由、加载模型、处理请求 |

### 5.2 关键类继承关系

```
Layer (ir/layers.py)
├── Linear
├── Embedding
├── LayerNormLayer
├── RMSNorm
├── DropoutLayer
├── RoPELayer
├── Attention         ← uses Linear
├── FeedForward       ← uses Linear
├── TransformerBlock  ← uses RMSNorm + Attention + FeedForward
└── Sequential

Backend (backends/base.py)  ← ABC
├── PyTorchBackend
└── NumPyBackend

Trainer (training/trainer.py)
└── uses: Backend, Optimizer, Scheduler, EMA

InferenceEngine (inference/engine.py)
└── uses: Backend, Tokenizer, Model

AutoLearner (src/auto_learner.py)
├── InteractionDB
└── CodeAugmentor
```

### 5.3 数据流（一次训练迭代）

```python
# 1. 数据加载
batch = next(train_loader)            # input_ids, labels
input_ids = batch['input_ids'].to(device)

# 2. 前向传播（IR 调用后端）
logits = model.forward(input_ids, backend)

# 3. 损失计算（含标签平滑）
shift_logits = logits[..., :-1, :]
shift_labels = labels[..., 1:]
loss = F.cross_entropy(shift_logits, shift_labels,
                       label_smoothing=0.05)
loss = loss / grad_accum_steps

# 4. 反向传播（PyTorch autograd）
scaler.scale(loss).backward()

# 5. 梯度累积 + 裁剪 + 优化器步进
if (step + 1) % grad_accum_steps == 0:
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
    if use_ema:
        ema.update()

# 6. 验证
val_loss, perplexity = trainer.evaluate()
```

### 5.4 数据流（一次推理生成）

```python
# 1. 编码 prompt
input_ids = tokenizer.encode(prompt)    # List[int]

# 2. 初始化 KV-Cache
past_kv = None
generated = list(input_ids)

# 3. 自回归生成
for step in range(max_new_tokens):
    if step == 0:
        x = np.array([generated])        # 全量
        logits, past_kv = model.forward(x, backend, use_cache=True)
    else:
        x = np.array([[generated[-1]]])  # 增量
        logits, past_kv = model.forward(x, backend,
                                         past_key_values=past_kv,
                                         use_cache=True)

    next_logits = logits[0, -1, :] / temperature
    next_token = sample(next_logits, top_k, top_p)
    if next_token == eos_id: break
    generated.append(next_token)

# 4. 解码
text = tokenizer.decode(generated)
```

---

## 6. 依赖关系

### 6.1 外部依赖（`requirements.txt`）

| 依赖 | 版本要求 | 用途 |
|------|----------|------|
| `torch` | >= 2.0.0 | 训练、PyTorch 后端推理、混合精度（推荐 2.8+ / CUDA 12.x） |
| `numpy` | >= 1.24.0 | NumPy 后端推理、tokenizer 字符处理 |
| `tqdm` | >= 4.65.0 | 训练进度条 |
| `pyyaml` | >= 6.0 | 配置文件加载 |
| `tensorboard` | >= 2.14.0 | 训练指标可视化 |
| `flask` | >= 2.0.0 | Web 服务框架 |

### 6.2 内部模块依赖图

```
train.py
  └─ ir.config, ir.transformer
  └─ backends.pytorch
  └─ training.trainer
  └─ src.tokenizer
  └─ src.device

generate.py / evaluate.py
  └─ ir.config, ir.transformer
  └─ backends.{pytorch,numpy}
  └─ inference.engine
  └─ src.tokenizer

web_app.py
  └─ ir.config, ir.transformer
  └─ backends.pytorch
  └─ inference.engine
  └─ src.tokenizer
  └─ src.compliance
  └─ src.auto_learner
  └─ src.moderator

ir/
  └─ (零依赖，标准库 only)
  └─ ir.config → ir.layers → ir.transformer
  └─ ir.analysis, ir.semantic (独立子模块)

backends/
  └─ backends.base (ABC)
  └─ backends.pytorch → backends.base
  └─ backends.numpy → backends.base
  └─ backends.spec → backends.base

training/
  └─ training.trainer → ir.layers, ir.transformer, backends.base
  └─ training.optimizer → torch

inference/
  └─ inference.engine → ir.transformer, backends.base, src.tokenizer

src/
  └─ src.tokenizer (可选 torch)
  └─ src.device (torch)
  └─ src.compliance (标准库)
  └─ src.moderator (标准库)
  └─ src.auto_learner → inference.engine, training.trainer
  └─ src.model, src.trainer (deprecated)
```

### 6.3 关键依赖约束

1. **`ir/` 不能依赖任何计算库**（torch/numpy），仅允许标准库。
2. **`backends/` 必须实现 `backends/base.py` 的所有抽象方法**。
3. **`training/` 依赖 `ir/` + `backends/`，但不耦合到具体后端**。
4. **`inference/` 自动选后端**，对上层透明。
5. **Web 服务可选用 PyTorch 或 NumPy 后端**，通过 `CODESPRITE_WEB_DEVICE` 环境变量控制。

---

## 7. 配置系统

### 7.1 配置文件：`config/config.yaml`

完整配置结构如下：

```yaml
# === 模型结构 ===
model:
  vocab_size: 4268
  hidden_size: 512
  num_layers: 8
  num_heads: 8
  num_kv_heads: 0              # 0 = MHA, >0 = GQA
  intermediate_size: 2048
  dropout: 0.1
  max_seq_length: 512
  tie_weights: true
  use_rope: true
  use_swiglu: true

# === 训练超参 ===
training:
  batch_size: 16
  learning_rate: 0.0003
  num_epochs: 10
  warmup_steps: 500
  gradient_accumulation_steps: 4
  max_grad_norm: 1.0
  weight_decay: 0.01
  adam_beta1: 0.9
  adam_beta2: 0.999
  adam_epsilon: 1.0e-8
  use_amp: true
  label_smoothing: 0.05
  use_ema: false
  ema_decay: 0.999
  early_stopping_patience: 5
  save_total_limit: 3

# === 自动学习 ===
auto_learning:
  enabled: false
  min_feedback_samples: 20
  min_positive_ratio: 0.6
  incremental_epochs: 3
  incremental_lr: 0.00005
  max_augmented_samples: 500
  augmentation_enabled: true
  schedule_interval_hours: 24

# === 数据 ===
data:
  train_file: "data/raw/train.txt"
  val_file: "data/raw/val.txt"
  test_file: "data/raw/test.txt"
  num_workers: 2

# === 系统 ===
system:
  device: "auto"
  cpu_threads: 2
  seed: 42
  precision: "fp32"
  log_level: "INFO"
  checkpoint_dir: "checkpoints"
  log_dir: "logs"
```

### 7.2 配置加载流程

```
config/config.yaml
    ↓ yaml.safe_load
raw dict
    ↓ Config.from_yaml(path)
Config 对象（聚合 ModelConfig + TrainingConfig + DataConfig + SystemConfig）
    ↓ Config.merge_from_args(args)
最终 Config（CLI 覆盖）
```

### 7.3 CLI 覆盖示例

```bash
python train.py --lr 0.001 --epochs 20 --batch-size 32 --no-amp --use-ema
```

| CLI 参数 | 覆盖的 config key |
|----------|-------------------|
| `--lr 0.001` | `training.learning_rate` |
| `--epochs 20` | `training.num_epochs` |
| `--batch-size 32` | `training.batch_size` |
| `--no-amp` | `training.use_amp = False` |
| `--use-ema` | `training.use_ema = True` |
| `--label-smoothing 0.1` | `training.label_smoothing` |
| `--no-rope` | `model.use_rope = False` |
| `--no-swiglu` | `model.activation = "gelu"` |
| `--device cpu` | `system.device = "cpu"` |
| `--no-cpu-fallback` | 环境变量 `CODESPRITE_ALLOW_CPU_FALLBACK=false` |

---

## 8. 项目运行方式

### 8.1 安装

```bash
# 1. 安装 PyTorch（根据硬件选择版本）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128  # GPU
# 或
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu  # CPU only

# 2. 安装项目依赖
pip install -r requirements.txt
```

### 8.2 训练

```bash
# 基础训练（使用 config/config.yaml 默认值）
python train.py

# 启用 EMA
python train.py --use-ema

# 启用梯度检查点
python train.py --use-checkpointing

# 消融实验
python train.py --no-rope --no-swiglu

# 自定义超参
python train.py --lr 0.001 --epochs 20 --batch-size 32

# 强制 CPU
python train.py --device cpu

# 禁用 CPU 回退（GPU 不可用直接报错）
python train.py --no-cpu-fallback

# 训练模式
python train.py --mode standard   # 标准训练（默认）
python train.py --mode auto       # 自动学习
python train.py --mode find-lr    # 学习率查找器

# 转换旧权重
python train.py --convert-old checkpoints/best_model.pt

# 从检查点恢复
python train.py --resume checkpoints/checkpoint_epoch_5.pt
```

**训练输出**：

```
Epoch 6/10 Summary:
  Train Loss: 4.2698
  Val Loss: 3.2807
  Perplexity: 26.60
  Learning Rate: 6.00e-05
  Epoch Time: 15.5s
  Total Time: 94.1s
  >> New best model! Val Loss: 3.2807
```

**检查点保存**：`checkpoints/best_model.pt`（验证损失最低）+ `checkpoint_epoch_*.pt`。

### 8.3 推理（交互式）

```bash
# 默认（PyTorch 后端）
python generate.py

# NumPy 后端（纯 CPU，无需 PyTorch）
python generate.py --backend numpy

# 单次生成
python generate.py --prompt "def fibonacci(n):" --max-tokens 50

# 指定 GPU
python generate.py --device cuda

# 调整采样参数
python generate.py --temperature 0.5 --top-k 30 --top-p 0.85
```

**交互命令**：

| 命令 | 说明 |
|------|------|
| `<文本>` | 输入提示，生成续写 |
| `:temp <n>` | 设置温度（0.1-2.0） |
| `:topk <n>` | 设置 Top-K（1-200） |
| `:topp <n>` | 设置 Top-P（0.0-1.0） |
| `:len <n>` | 设置最大生成长度（10-500） |
| `:info` | 显示模型信息 |
| `quit` / `exit` / `q` | 退出 |

### 8.4 评估

```bash
# 评估 best_model
python evaluate.py

# 评估指定检查点
python evaluate.py --checkpoint checkpoints/checkpoint_epoch_5.pt

# 使用 NumPy 后端评估
python evaluate.py --backend numpy
```

**输出**：PPL、Token-level Loss、生成质量抽检、自动化质量评级。

### 8.5 Web 服务

```bash
# 启动
python web_app.py

# 或一键启动
./start_web.sh        # Linux/macOS
start_web.bat         # Windows

# 高级选项
./start_web.sh --port 8080
./start_web.sh --debug
```

**访问**：浏览器打开 http://localhost:5000

**API 端点**：

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/generate` | POST | 代码分析 |
| `/api/info` | GET | 引擎信息 |
| `/api/health` | GET | 健康检查 |
| `/api/feedback` | POST | 提交反馈（赞/踩） |
| `/api/learning-status` | GET | 自动学习状态 |
| `/api/learning/start` | POST | 手动触发增量学习 |
| `/api/learning/config` | GET/POST | 学习配置管理 |
| `/api/user-rights` | GET | 用户权利概览 |
| `/api/data-export` | GET | 数据导出 |
| `/api/data-delete` | POST | 数据删除 |

**调用示例**：

```bash
curl -X POST http://localhost:5000/api/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "def hello():", "max_new_tokens": 100, "temperature": 0.8}'

curl -X POST http://localhost:5000/api/feedback \
  -H "Content-Type: application/json" \
  -d '{"interaction_id": "xxx", "feedback": 1}'
```

### 8.6 模型导出

```python
# GGUF 导出（llama.cpp 兼容）
from export.gguf import export_gguf
export_gguf(model, "codesprite.gguf")

# ONNX 导出
from export.onnx import export_onnx
export_onnx(model, "codesprite.onnx")
```

### 8.7 工具脚本

```bash
# 数据清洗
python tools/data_cleaner.py --input data/raw/train.txt --output data/clean/train.txt

# 数据统计
python tools/data_stats.py --input data/raw/train.txt

# 权重量化（INT8）
python tools/quantize.py --input checkpoints/best_model.pt --output checkpoints/best_int8.pt --bits 8

# 旧权重转换
python tools/convert_checkpoint.py --old checkpoints/old.pt --new checkpoints/new.pt

# 打包可执行文件
python tools/build_nuitka.py
```

---

## 9. 测试与质量保证

### 9.1 测试套件（`tests/`）

| 文件 | 测试内容 |
|------|----------|
| `tests/test_ir.py` | IR 层结构正确性（层数、参数、KV-Cache 接口） |
| `tests/test_backends.py` | PyTorch / NumPy 后端计算一致性、KV-Cache 一致性 |
| `tests/test_tokenizer.py` | 分词器 round-trip（encode → decode）、CJK 支持、代码符号 |
| `tests/test_training_smoke.py` | 端到端训练 smoke test（小数据集跑 1 epoch） |

### 9.2 质量保证机制

- **单元测试**：`tests/` 覆盖核心模块
- **CI 友好**：所有命令非交互式（`python xxx.py --arg value`）
- **可复现性**：`set_seed(config.system.seed)` 统一随机种子
- **审计日志**：`AuditLogger` 记录所有 Web 请求与生成内容
- **错误处理**：`utils/errors.py` 自定义异常体系，避免裸异常
- **配置校验**：`ModelConfig.__post_init__` 强制维度约束

---

## 10. 附录

### 10.1 参数量计算（默认配置）

| 组件 | 参数量 |
|------|--------|
| Embedding (4268 × 512) | 2,185,216 |
| 8 × TransformerBlock | ~34,816,000 |
| Final RMSNorm | 512 |
| LM Head (Tied Weights) | 0 |
| **总计** | **~37.9M** |

### 10.2 设备策略

| 场景 | 策略 |
|------|------|
| 训练 | 优先 GPU → 自动回退 CPU（可禁用） |
| 推理（CLI） | 默认 GPU（如可用），否则 CPU |
| 推理（Web） | 默认 CPU（避免 GPU 显存竞争），可设置 `CODESPRITE_WEB_DEVICE=cuda` |
| NumPy 后端 | 始终纯 CPU |
| PyTorch 后端 + CPU | CPU 线程数由 `config.yaml` 的 `cpu_threads` 控制（建议：老电脑=2，新电脑=4，不做其他事=auto） |

### 10.3 性能优化技术清单

| 技术 | 用途 |
|------|------|
| **RoPE 旋转位置编码** | 支持序列外推，无需学习位置嵌入 |
| **SwiGLU 门控前馈** | 相比 ReLU 表达力更强 |
| **GQA 分组查询注意力** | 减少 KV-Cache 显存，提升推理吞吐 |
| **KV-Cache 推理** | 推理时复用历史 K/V，避免重复计算 |
| **混合精度 (FP16/AMP)** | 训练加速 + 显存节省约 40% |
| **标签平滑** | 防止过拟合，提升泛化 |
| **EMA 指数移动平均** | 推理时更稳定的权重 |
| **梯度检查点** | 用计算换显存，支持更大模型 |
| **CosineAnnealing + Warmup** | 稳定的训练收敛 |
| **INT8 / INT4 量化** | 模型压缩与加速部署 |
| **Nuitka 打包** | 启动更快、避免 Python 环境依赖 |

### 10.4 风险与限制

详见 [`RISKS.md`](./RISKS.md) 和 [`INVISIBILITY.md`](./INVISIBILITY.md)。

**主要风险**：

- 38M 参数对**低资源编程语言**（仓颉 / Haskell / COBOL）易产生严重幻觉
- CPU 训练速度约为 GPU 的 1/10 - 1/50
- 训练数据质量与多样性显著影响生成质量
- GGUF/ONNX 导出对 dynamic shape 支持有限

### 10.5 路线图

详见 [`ROADMAP.md`](./ROADMAP.md)：IR 架构增强、输出能力优化、工程化与生态建设。

### 10.6 命名与版本约定

- **IR v1**（`src/model.py`）：旧版 PyTorch-native 模型，**已废弃**。
- **IR v2**（`ir/`）：当前框架无关架构，**主推**。
- **Checkpoint 版本**：旧 `best_model.pt` 需用 `tools/convert_checkpoint.py` 转换到 v2。

### 10.7 常见问题速查

| 问题 | 解决方案 |
|------|----------|
| `inf` Val Loss | 增大学习率 / 减少 warmup_steps / 禁用 AMP |
| 训练太慢 | 减小 batch_size / num_epochs / 用 GPU |
| 想换语言 | 修改 `config.yaml` 后必须重新训练（权重不兼容） |
| 切换推理后端 | `python generate.py --backend numpy` |
| 仓颉/Haskell 输出全错 | 引擎对低资源语言不可靠，仅作参考 |
| Linux 启动脚本权限 | `chmod +x start_web.sh` |
| PowerShell 报错 npm | 本项目不依赖 npm，可忽略 |

---

## 许可证

MIT License — 详见 [LICENSE](./LICENSE)。
