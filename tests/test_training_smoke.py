"""
训练冒烟测试 — 验证整个 pipeline 正常工作：
  1) IR 模型初始化
  2) PyTorch 后端权重初始化
  3) tokenizer 编码 → 前向传播 → loss 计算
  4) 保存/加载 checkpoint
"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tokenizer import CodeTokenizer, TextDataset, create_dataloader


class TestEndToEndPipeline(unittest.TestCase):

    def setUp(self):
        self.tok = CodeTokenizer()

    def test_tokenizer_encode_decode_is_fast(self):
        """普通代码片段应在合理时间内完成 encode"""
        import time
        code = (
            "def hello(name='world'):\n"
            "    # 打印问候\n"
            "    print('hello', name)\n"
            "    return name\n"
        )
        start = time.time()
        for _ in range(100):
            self.tok.encode(code)
        elapsed = time.time() - start
        # 100 次编码应在 5 秒内完成
        self.assertLess(elapsed, 5.0,
                        f"Tokenizer 太慢: {elapsed:.2f}s/100 runs")

    def test_dataloader_with_jsonl(self):
        """确认 DataLoader 能用 JSONL 构造 batch"""
        with tempfile.NamedTemporaryFile(
                'w', suffix='.jsonl', delete=False, encoding='utf-8') as f:
            for i in range(8):
                import json
                json.dump({
                    "prompt": f"问题{i}: 写一个函数",
                    "response": f"def fn_{i}(x):\n    return x * 2"
                }, f)
                f.write('\n')
            path = f.name

        try:
            dataset = TextDataset(path, self.tok, max_length=64)
            loader = create_dataloader(dataset, batch_size=4,
                                       shuffle=False, num_workers=0)
            batches = list(loader)
            self.assertEqual(len(batches), 2)  # 8 samples / batch 4
            batch = batches[0]
            self.assertIn('input_ids', batch)
            self.assertIn('labels', batch)
            self.assertEqual(batch['input_ids'].shape[0], 4)
        finally:
            os.unlink(path)

    def test_ir_model_forward_pytorch(self):
        """真正跑一次 IR 模型前向传播 + 损失计算"""
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch 未安装，跳过")

        from ir.config import ModelConfig
        from ir.transformer import TransformerModel
        from backends.pytorch import PyTorchBackend, init_model_weights

        mc = ModelConfig(
            vocab_size=self.tok.vocab_size,
            hidden_size=64,
            num_layers=2,
            num_heads=4,
            intermediate_size=128,
            max_seq_length=32,
        )
        model = TransformerModel(mc)
        backend = PyTorchBackend(device="cpu")
        init_model_weights(model, backend)
        model.eval()  # 关闭 dropout，保证可重复

        # 准备输入
        prompt = "def foo(x):\n    return x + 1"
        tokens = self.tok.encode(prompt, max_length=32)
        input_ids = torch.tensor([tokens], dtype=torch.long)

        # 前向传播
        logits = model.forward(input_ids, backend)
        self.assertEqual(logits.shape, (1, 32, mc.vocab_size))

        # 计算 next-token loss（与训练循环相同的逻辑）
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        loss = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=self.tok.pad_token_id,
        )
        self.assertFalse(torch.isnan(loss))
        self.assertGreater(loss.item(), 0)

    def test_checkpoint_roundtrip(self):
        """保存 → 加载：权重应一致"""
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch 未安装，跳过")

        from ir.config import ModelConfig
        from ir.transformer import TransformerModel
        from backends.pytorch import PyTorchBackend, init_model_weights

        mc = ModelConfig(
            vocab_size=self.tok.vocab_size,
            hidden_size=64, num_layers=2, num_heads=4,
            intermediate_size=128, max_seq_length=32,
        )

        path = None
        try:
            model1 = TransformerModel(mc)
            backend1 = PyTorchBackend(device="cpu")
            init_model_weights(model1, backend1)
            model1.eval()

            with tempfile.NamedTemporaryFile(
                    'wb', suffix='.pt', delete=False) as f:
                path = f.name
            backend1.save_checkpoint(model1, path, {"epoch": 1})

            # 第二个模型加载
            model2 = TransformerModel(mc)
            backend2 = PyTorchBackend(device="cpu")
            init_model_weights(model2, backend2)
            backend2.load_checkpoint(model2, path)
            model2.eval()

            # 验证两者前向传播结果相同（相同输入 → 相同输出）
            tokens = self.tok.encode("def foo(): pass", max_length=16)
            input_ids = torch.tensor([tokens], dtype=torch.long)

            with torch.no_grad():
                out1 = model1.forward(input_ids, backend1)
                out2 = model2.forward(input_ids, backend2)

            diff = (out1 - out2).abs().max().item()
            self.assertLess(diff, 1e-5,
                            f"Checkpoint round-trip 失败: max diff={diff}")
        finally:
            if path is not None:
                os.unlink(path)

    def test_backend_interface(self):
        """各后端的核心 API 检查"""
        from ir.config import ModelConfig
        from ir.transformer import TransformerModel
        from backends.base import Backend

        # 基础接口
        required_methods = [
            'embedding', 'linear', 'rms_norm', 'softmax',
            'save_checkpoint', 'load_checkpoint',
            'get_state_dict', 'load_state_dict',
            'shape', 'causal_mask', 'scaled_dot_product_attention',
        ]
        for name in required_methods:
            # 确保 Backend 基类或其子类实现了这些方法
            has_in_base = hasattr(Backend, name)
            self.assertTrue(has_in_base, f"Backend 应该声明方法: {name}")

        # NumPy Backend 是否可用
        try:
            from backends.numpy import NumPyBackend
            backend = NumPyBackend()
            for name in required_methods:
                self.assertTrue(hasattr(backend, name),
                                f"NumPyBackend 缺少方法: {name}")
        except ImportError:
            self.skipTest("NumPy 不可用")

        # 模型结构接口
        mc = ModelConfig(
            vocab_size=self.tok.vocab_size,
            hidden_size=64, num_layers=2, num_heads=4,
            intermediate_size=128, max_seq_length=32,
        )
        model = TransformerModel(mc)
        self.assertTrue(hasattr(model, 'params'))
        self.assertTrue(hasattr(model, 'blocks'))
        self.assertTrue(hasattr(model, 'forward'))
        self.assertTrue(hasattr(model, 'eval'))
        self.assertTrue(hasattr(model, 'train'))
        self.assertEqual(len(model.blocks), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
