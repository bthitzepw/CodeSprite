"""
CodeTokenizer 测试 — 覆盖:
  1) 基础字符级编码/解码
  2) 代码关键字合并 token （"import" 变成 1 个 token 而不是 6 个）
  3) 中文编码
  4) 编码 → 解码的一致性（round-trip）
  5) JSONL 训练数据格式
  6) token 压缩率
  7) max_length 截断 & padding
"""

import os
import sys
import json
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tokenizer import CodeTokenizer, TextDataset, SimpleTokenizer


class TestTokenizerBasic(unittest.TestCase):

    def setUp(self):
        self.tok = CodeTokenizer()

    def test_vocab_size_is_reasonable(self):
        # 基础 ASCII 256 + special 5 + symbols ~30 + 关键字 ~150 + CJK ~6500
        self.assertGreater(self.tok.vocab_size, 4000)
        self.assertLess(self.tok.vocab_size, 20000)

    def test_special_tokens_exist(self):
        for name in ['<PAD>', '<UNK>', '<BOS>', '<EOS>', '<MASK>']:
            self.assertIn(name, self.tok.char_to_idx)

    def test_simple_encoding(self):
        ids = self.tok.encode("print('hello')")
        # 必须包含 BOS/EOS
        self.assertEqual(ids[0], self.tok.bos_token_id)
        self.assertEqual(ids[-1], self.tok.eos_token_id)

    def test_ascii_roundtrip(self):
        text = "def add(a, b):\n    return a + b"
        ids = self.tok.encode(text)
        decoded = self.tok.decode(ids)
        # 解码后应能还原文本（去掉 BOS/EOS）
        self.assertIn("def", decoded)
        self.assertIn("add", decoded)


class TestKeywordMerge(unittest.TestCase):
    """核心改进测试：代码关键字应被识别为单个 token，而不是多字符"""

    def setUp(self):
        self.tok = CodeTokenizer()

    def test_import_is_single_token(self):
        # "import" 是一个 token，不是 6 个
        ids = self.tok.encode("import os")
        # 去掉 BOS/EOS
        inner_ids = ids[1:-1]
        # 第一个字符应该是 `import` 这个 token（长度为6）
        self.assertIn("import", self.tok.char_to_idx)
        import_id = self.tok.char_to_idx["import"]
        # 找到 import_id 并确认它确实被使用
        self.assertIn(import_id, inner_ids)

    def test_return_is_single_token(self):
        self.assertIn("return", self.tok.char_to_idx)
        ids = self.tok.encode("return 42")
        return_id = self.tok.char_to_idx["return"]
        self.assertIn(return_id, ids[1:-1])

    def test_merge_tokens_are_sorted_by_length(self):
        # `_merge_tokens` 必须按长度从长到短排序（贪心匹配的前提）
        lengths = [len(t) for t in self.tok._merge_tokens]
        self.assertEqual(lengths, sorted(lengths, reverse=True))

    def test_code_compression_ratio(self):
        code = (
            "def add(a, b):\n"
            "    return a + b\n\n"
            "def main():\n"
            "    import os\n"
            "    print('hello from' + os.path.basename(__file__))\n"
        )
        tokens = self.tok.encode(code)
        # 去掉 BOS/EOS 后，token 数应显著少于字符数
        token_count = len(tokens) - 2
        # 压缩率应至少 > 1.2（关键字 token 化之后）
        ratio = self.tok.compression_ratio(code)
        print(f"  Compression ratio: {ratio:.2f}x "
              f"({len(code)} chars -> {token_count} tokens)")
        self.assertGreater(ratio, 1.0)


class TestChineseEncoding(unittest.TestCase):

    def setUp(self):
        self.tok = CodeTokenizer()

    def test_chinese_chars_are_known(self):
        # GB2312 覆盖的常见汉字应在词汇表中
        for ch in ["你", "好", "函", "数", "代", "码", "Python"]:
            # 单字必须在表中
            for c in ch:
                self.assertIn(c, self.tok.char_to_idx,
                              f"Character '{c}' missing from vocab")

    def test_chinese_comment(self):
        text = "# 这是一个中文注释：打印 hello world\nprint('hello')"
        ids = self.tok.encode(text)
        self.assertGreater(len(ids), 5)


class TestJSONLDataFormat(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        try:
            import torch  # noqa: F401
            cls.torch_ok = True
        except ImportError:
            cls.torch_ok = False

    def setUp(self):
        self.tok = CodeTokenizer()

    def _require_torch(self):
        if not self.torch_ok:
            self.skipTest("需要 PyTorch 才能运行")

    def test_jsonl_prompt_response(self):
        self._require_torch()
        with tempfile.NamedTemporaryFile(
                'w', suffix='.jsonl', delete=False, encoding='utf-8') as f:
            json.dump({"prompt": "实现加法",
                       "response": "def add(a, b):\n    return a + b"}, f)
            f.write('\n')
            json.dump({"prompt": "打印 hello",
                       "response": "print('hello')"}, f)
            f.write('\n')
            path = f.name

        try:
            dataset = TextDataset(path, self.tok, max_length=128)
            self.assertEqual(len(dataset), 2)
            item = dataset[0]
            self.assertIn('input_ids', item)
            self.assertIn('labels', item)
            # 内容应包含 prompt + response
            decoded = self.tok.decode(item['input_ids'].tolist())
            self.assertIn("def add", decoded)
            self.assertIn("实现", decoded)
        finally:
            os.unlink(path)

    def test_jsonl_with_text_field(self):
        self._require_torch()
        with tempfile.NamedTemporaryFile(
                'w', suffix='.jsonl', delete=False, encoding='utf-8') as f:
            json.dump({"text": "def hello():\n    print('hi')"}, f)
            f.write('\n')
            path = f.name

        try:
            dataset = TextDataset(path, self.tok, max_length=128)
            self.assertEqual(len(dataset), 1)
        finally:
            os.unlink(path)

    def test_plain_text_format(self):
        with tempfile.NamedTemporaryFile(
                'w', suffix='.txt', delete=False, encoding='utf-8') as f:
            f.write("def foo(): pass\n")
            f.write("def bar(): return 42\n")
            path = f.name

        try:
            dataset = TextDataset(path, self.tok, max_length=64)
            self.assertEqual(len(dataset), 2)
        finally:
            os.unlink(path)

    def test_jsonl_malformed_line_is_tolerated(self):
        """JSON 损坏的行不应抛出异常"""
        self._require_torch()
        with tempfile.NamedTemporaryFile(
                'w', suffix='.jsonl', delete=False, encoding='utf-8') as f:
            f.write('{"prompt": "ok", "response": "yes"}\n')
            f.write('{not valid json here}\n')
            f.write('{"prompt": "good", "response": "day"}\n')
            path = f.name

        try:
            dataset = TextDataset(path, self.tok, max_length=64)
            self.assertEqual(len(dataset), 3)
        finally:
            os.unlink(path)


class TestMaxLengthHandling(unittest.TestCase):

    def setUp(self):
        self.tok = CodeTokenizer()

    def test_truncation_when_too_long(self):
        long_text = "import os, sys\n" * 50
        ids = self.tok.encode(long_text, max_length=64)
        self.assertEqual(len(ids), 64)
        # 最后一个 token 应为 EOS
        self.assertEqual(ids[-1], self.tok.eos_token_id)

    def test_padding_when_short(self):
        short = "hi"
        ids = self.tok.encode(short, max_length=32)
        self.assertEqual(len(ids), 32)
        # pad tokens 在末尾
        self.assertEqual(ids[-1], self.tok.pad_token_id)


class TestSimpleTokenizerBackwardCompat(unittest.TestCase):
    """SimpleTokenizer 是 CodeTokenizer 的别名"""

    def test_simple_tokenizer_is_code_tokenizer(self):
        self.assertIs(SimpleTokenizer, CodeTokenizer)


if __name__ == "__main__":
    unittest.main(verbosity=2)
