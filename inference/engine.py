"""
推理引擎 — 后端无关的文本生成

自动检测可用后端（PyTorch → NumPy 降级）。
支持 KV-Cache 加速（IR + 旧模型双路径）、流式生成、top-k/top-p 采样。

设备选择策略:
  - "auto"  → 自动最优（CUDA > CPU），推荐
  - "cuda"  → 显式 GPU
  - "cpu"   → 纯 CPU 推理（自动限制线程数）

v2 更新:
  - IR 版 TransformerModel KV-Cache 全链路打通
  - 新增 generate_stream() 流式生成
  - 采样逻辑提取为独立方法
"""

import math
import os
import logging
from typing import List, Optional, Tuple, Generator

logger = logging.getLogger("CodeSprite")


class InferenceEngine:
    """
    CodeSprite 推理引擎

    用法:
        # 自动选择后端 + 设备
        engine = InferenceEngine(model, checkpoint_path="best_model.pt")

        # 文本生成（KV-Cache 加速）
        output = engine.generate_with_kv_cache("def hello(", max_new_tokens=50)

        # 流式生成
        for token_text in engine.generate_stream("def fib(", max_new_tokens=100):
            print(token_text, end="", flush=True)
    """

    def __init__(self, model, backend=None, checkpoint_path: str = None,
                 tokenizer=None, device: str = "auto"):
        self.model = model
        self.tokenizer = tokenizer

        # 自动选择后端
        if backend is not None:
            self.backend = backend
            self._device = getattr(backend, '_resolved_device', device)
        else:
            self.backend, self._device = self._auto_select_backend(device)

        # 打印推理设备信息
        print(f"[Inference] Device: {self._device}, Backend: {self.backend.name}")

        # 加载权重
        if checkpoint_path and os.path.exists(checkpoint_path):
            self.load(checkpoint_path)
        elif not self._has_weights():
            raise RuntimeError(
                "模型未初始化权重。请先加载检查点或初始化权重。"
            )

        # 采样参数
        self.temperature = 1.0
        self.top_k = None
        self.top_p = None

    def _auto_select_backend(self, device: str = "auto"):
        """自动选择可用后端，返回 (backend, resolved_device_str)"""
        from src.device import resolve_device

        resolved_device = resolve_device(device)

        try:
            from backends.pytorch import PyTorchBackend, init_model_weights as _
            backend = PyTorchBackend(device=resolved_device)
            return backend, resolved_device
        except ImportError:
            from backends.numpy import NumPyBackend
            logger.warning("PyTorch 不可用，使用 NumPy 后端（纯 CPU 推理）")
            return NumPyBackend(), "cpu"

    def _has_weights(self) -> bool:
        """检查模型是否已有权重"""
        try:
            if hasattr(self.model, 'embedding') and hasattr(self.model.embedding, 'params'):
                return len(self.model.embedding.params) > 0
            if hasattr(self.model, 'blocks') and len(self.model.blocks) > 0:
                first_block = self.model.blocks[0]
                if hasattr(first_block, 'params'):
                    return len(first_block.params) > 0
            return True
        except AttributeError:
            return True

    def _is_ir_model(self) -> bool:
        """检测是否为 IR TransformerModel"""
        try:
            from ir.transformer import TransformerModel
            return isinstance(self.model, TransformerModel)
        except ImportError:
            return False

    def _is_native_model(self) -> bool:
        """检测是否为旧 src/model.py CodeSprite (nn.Module)"""
        return hasattr(self.model, 'generate') and hasattr(self.model, 'layers')

    def load(self, path: str):
        """加载模型权重"""
        self.backend.load_checkpoint(self.model, path)
        print(f"Loaded weights from {path}")
        print(f"  Backend: {self.backend.name}")
        print(f"  Parameters: {self.model.get_param_count():,}")
        return self

    # ============================================================
    # 采样方法
    # ============================================================

    def _sample_token(self, logits, temperature: float = 1.0,
                      top_k: int = None, top_p: float = None) -> int:
        """
        从 logits 中采样下一个 token

        支持 temperature 缩放、top-k 过滤、top-p (nucleus) 过滤。
        返回采样的 token ID。
        """
        import numpy as np

        # 转为 numpy 进行采样操作
        if hasattr(logits, 'detach'):
            logits_np = logits.detach().cpu().numpy()
        elif hasattr(self.backend, 'to_numpy'):
            logits_np = self.backend.to_numpy(logits)
        else:
            logits_np = np.array(logits)

        # 确保是一维数组
        logits_np = logits_np.flatten()

        # Temperature 缩放
        temp = max(temperature, 1e-10)
        logits_np = logits_np / temp

        # Top-K 过滤
        if top_k is not None and top_k > 0 and top_k < len(logits_np):
            indices = np.argpartition(-logits_np, top_k)[:top_k]
            mask = np.zeros_like(logits_np, dtype=bool)
            mask[indices] = True
            logits_np[~mask] = float('-inf')

        # Top-P (nucleus) 过滤
        if top_p is not None and top_p < 1.0:
            valid_mask = logits_np != float('-inf')
            valid_logits = logits_np[valid_mask]
            valid_indices = np.where(valid_mask)[0]

            if len(valid_logits) == 0:
                return int(np.random.randint(len(logits_np)))

            valid_logits_stable = valid_logits - valid_logits.max()
            valid_probs = np.exp(valid_logits_stable)
            valid_probs = valid_probs / valid_probs.sum()
            cumsum = np.cumsum(valid_probs)
            cutoff_idx = int((cumsum > top_p).argmax()) + 1

            if cutoff_idx < len(valid_indices):
                to_mask = valid_indices[cutoff_idx:]
                logits_np[to_mask] = float('-inf')

        # 数值稳定的 Softmax
        logits_np = logits_np - logits_np.max()
        exp_logits = np.exp(logits_np)
        exp_logits = np.clip(exp_logits, 1e-10, None)
        probs = exp_logits / exp_logits.sum()
        probs = probs / probs.sum()

        # 采样
        next_token = int(np.random.choice(len(probs), p=probs))
        return next_token

    # ============================================================
    # 文本生成
    # ============================================================

    def generate(self, prompt: str, max_new_tokens: int = 50,
                 temperature: float = None, top_k: int = None,
                 top_p: float = None) -> str:
        """
        文本生成（全量推理，每次重新计算整个序列）

        Args:
            prompt: 输入文本
            max_new_tokens: 最大生成 token 数
            temperature: 采样温度（None = 使用默认值）
            top_k: Top-K 采样参数
            top_p: Top-P (nucleus) 采样参数

        Returns:
            生成的文本（含输入）
        """
        if self.tokenizer is None:
            raise RuntimeError("需要设置 tokenizer 才能生成文本")

        temp = temperature if temperature is not None else self.temperature
        tk = top_k if top_k is not None else self.top_k
        tp = top_p if top_p is not None else self.top_p

        input_ids = self.tokenizer.encode(prompt)
        generated_ids = list(input_ids)
        eos_id = self.model.config.eos_token_id

        for _ in range(max_new_tokens):
            max_len = self.model.config.max_seq_length
            context = generated_ids[-max_len:]

            import numpy as np
            x = np.array([context], dtype=np.int64)
            logits = self.model.forward(x, self.backend)

            next_logits = logits[0, -1, :]
            next_token = self._sample_token(next_logits, temp, tk, tp)

            if next_token == eos_id:
                break
            generated_ids.append(next_token)

        return self.tokenizer.decode(generated_ids)

    def generate_ids(self, input_ids: List[int], max_new_tokens: int = 50,
                     temperature: float = 1.0, top_k: int = None,
                     top_p: float = None) -> List[int]:
        """
        从 token IDs 生成（不经过 tokenizer）
        """
        import numpy as np

        generated = list(input_ids)
        eos_id = self.model.config.eos_token_id

        for _ in range(max_new_tokens):
            max_len = self.model.config.max_seq_length
            context = generated[-max_len:]
            x = np.array([context], dtype=np.int64)

            logits = self.model.forward(x, self.backend)
            next_logits = logits[0, -1, :]
            next_token = self._sample_token(next_logits, temperature, top_k, top_p)

            if next_token == eos_id:
                break
            generated.append(next_token)

        return generated

    # ============================================================
    # KV-Cache 加速生成
    # ============================================================

    def generate_with_kv_cache(self, prompt: str, max_new_tokens: int = 100,
                                temperature: float = 0.8, top_k: int = 50,
                                top_p: float = 0.9) -> str:
        """
        使用 KV-Cache 加速的文本生成

        支持两条路径:
          1. IR TransformerModel → 走 IR KV-Cache 路径（优先）
          2. src/model.py CodeSprite → 走原生 nn.Module 路径
          3. 降级 → 全量推理

        Args:
            prompt: 输入文本
            max_new_tokens: 最大生成 token 数
            temperature: 采样温度
            top_k: Top-K 采样参数
            top_p: Top-P (nucleus) 采样参数

        Returns:
            生成的完整文本（含输入 prompt）
        """
        if self.tokenizer is None:
            raise RuntimeError("需要设置 tokenizer 才能使用此方法")

        input_ids = self.tokenizer.encode(prompt)

        # Path 1: IR TransformerModel KV-Cache
        if self._is_ir_model():
            try:
                output_ids = self._generate_ir_kv_cache(
                    input_ids, max_new_tokens, temperature, top_k, top_p
                )
                return self.tokenizer.decode(output_ids)
            except Exception as e:
                logger.warning(f"IR KV-Cache 推理失败，降级到全量: {e}")
                return self.generate(prompt, max_new_tokens, temperature, top_k, top_p)

        # Path 2: 旧 nn.Module 模型
        elif self._is_native_model():
            try:
                import torch
                device = getattr(self.backend, 'device', 'cpu')
                ids_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
                output_ids = self.model.generate(
                    ids_tensor, max_new_tokens=max_new_tokens,
                    temperature=temperature, top_k=top_k, top_p=top_p,
                    use_kv_cache=True
                )
                return self.tokenizer.decode(output_ids[0].tolist())
            except Exception as e:
                logger.warning(f"原生 KV-Cache 推理失败，降级到全量: {e}")
                return self.generate(prompt, max_new_tokens, temperature, top_k, top_p)

        # Path 3: 降级全量推理
        return self.generate(prompt, max_new_tokens, temperature, top_k, top_p)

    def _generate_ir_kv_cache(self, input_ids: List[int],
                               max_new_tokens: int, temperature: float,
                               top_k: int, top_p: float) -> List[int]:
        """
        IR TransformerModel 的 KV-Cache 推理路径

        算法:
          1. 首次：输入完整 prompt → 得到所有层的 KV-Cache + 最后一个 logits
          2. 之后：每次只输入新生成的 1 个 token + 传入之前的 KV-Cache
          3. 每步复用历史 KV，避免重复计算

        TransformerModel.forward() 支持:
          - past_key_values: [(k0,v0), (k1,v1), ...] 每层的 KV-Cache
          - use_cache=True → 返回 (logits, new_key_values)
        """
        import numpy as np

        generated = list(input_ids)
        eos_id = self.model.config.eos_token_id
        past_key_values = None

        for step in range(max_new_tokens):
            if step == 0:
                # 首次：输入完整 prompt，建立 KV-Cache
                x = np.array([generated], dtype=np.int64)
                result = self.model.forward(
                    x, self.backend, past_key_values=None, use_cache=True
                )
            else:
                # 之后：只输入最后 1 个 token + 使用 KV-Cache
                x = np.array([[generated[-1]]], dtype=np.int64)
                result = self.model.forward(
                    x, self.backend, past_key_values=past_key_values, use_cache=True
                )

            # result = (logits, new_key_values)
            logits, past_key_values = result

            # 取最后一个位置的 logits
            next_logits = logits[0, -1, :]
            next_token = self._sample_token(next_logits, temperature, top_k, top_p)

            if next_token == eos_id:
                break
            generated.append(next_token)

        return generated

    # ============================================================
    # 流式生成
    # ============================================================

    def generate_stream(self, prompt: str, max_new_tokens: int = 100,
                        temperature: float = 0.8, top_k: int = 50,
                        top_p: float = 0.9) -> Generator[str, None, None]:
        """
        流式文本生成（逐 token yield）

        每次 yield 的是当前生成的 token 文本。
        如果设置了 tokenizer，yield 解码后的文本；
        否则 yield token ID 字符串。

        用法:
            for token in engine.generate_stream("def fib(", max_new_tokens=100):
                print(token, end="", flush=True)
                # 或发送到 WebSocket / SSE
        """
        if self.tokenizer is None:
            raise RuntimeError("需要设置 tokenizer 才能流式生成")

        input_ids = self.tokenizer.encode(prompt)

        # 优先使用 KV-Cache 加速流式输出
        if self._is_ir_model():
            for token in self._stream_ir_kv_cache(
                input_ids, max_new_tokens, temperature, top_k, top_p
            ):
                yield token
        else:
            for token in self._stream_full(input_ids, max_new_tokens,
                                           temperature, top_k, top_p):
                yield token

    def _stream_ir_kv_cache(self, input_ids: List[int],
                             max_new_tokens: int, temperature: float,
                             top_k: int, top_p: float) -> Generator[str, None, None]:
        """IR TransformerModel 的流式 KV-Cache 推理"""
        import numpy as np

        generated = list(input_ids)
        eos_id = self.model.config.eos_token_id
        past_key_values = None

        for step in range(max_new_tokens):
            if step == 0:
                x = np.array([generated], dtype=np.int64)
                result = self.model.forward(
                    x, self.backend, past_key_values=None, use_cache=True
                )
            else:
                x = np.array([[generated[-1]]], dtype=np.int64)
                result = self.model.forward(
                    x, self.backend, past_key_values=past_key_values, use_cache=True
                )

            logits, past_key_values = result
            next_logits = logits[0, -1, :]
            next_token = self._sample_token(next_logits, temperature, top_k, top_p)

            if next_token == eos_id:
                break
            generated.append(next_token)

            # yield 当前 token 的解码文本
            yield self.tokenizer.decode([next_token])

    def _stream_full(self, input_ids: List[int],
                     max_new_tokens: int, temperature: float,
                     top_k: int, top_p: float) -> Generator[str, None, None]:
        """全量推理的流式生成（降级路径）"""
        import numpy as np

        generated = list(input_ids)
        eos_id = self.model.config.eos_token_id

        for _ in range(max_new_tokens):
            max_len = self.model.config.max_seq_length
            context = generated[-max_len:]
            x = np.array([context], dtype=np.int64)

            logits = self.model.forward(x, self.backend)
            next_logits = logits[0, -1, :]
            next_token = self._sample_token(next_logits, temperature, top_k, top_p)

            if next_token == eos_id:
                break
            generated.append(next_token)
            yield self.tokenizer.decode([next_token])

    # ============================================================
    # 引擎信息
    # ============================================================

    def info(self) -> dict:
        """返回引擎信息"""
        return {
            "backend": self.backend.name,
            "parameters": self.model.get_param_count(),
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "supports_kv_cache": self._is_ir_model() or self._is_native_model(),
            "supports_streaming": True,
        }
