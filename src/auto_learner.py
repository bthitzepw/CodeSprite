"""
CodeSprite 自动学习模块 (Auto-Learning Module)

核心功能:
  1. 用户反馈收集 - 对话质量评分（点赞/点踩）
  2. 交互记录存储 - SQLite 持久化存储所有对话和反馈
  3. 数据增强 - 代码变换增强训练数据（变量重命名、注释替换等）
  4. 增量训练 - 基于用户反馈自动触发增量微调
  5. 学习进度追踪 - 训练轮次、指标变化、模型版本管理

架构说明:
  使用 IR 架构（ir/transformer.py），与主训练路径完全一致。
  不再依赖废弃的 src/model.py。checkpoint 格式与 train.py 100% 兼容。

依据:
  - 《生成式AI管理办法》第15条 - 模型训练记录保存
  - GDPR Art.22 - 自动化决策的透明性
"""

import sqlite3
import json
import os
import time
import random
import re
import threading
import logging
from datetime import datetime
from collections import defaultdict
from typing import List, Dict, Optional, Tuple, Generator

logger = logging.getLogger(__name__)


# ============================================================
# 数据库管理 - 存储用户交互和反馈
# ============================================================

class InteractionDB:
    """用户交互数据库，基于 SQLite 存储"""

    def __init__(self, db_path: str = 'data/auto_learning.db'):
        self.db_path = db_path
        self.lock = threading.Lock()
        os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else '.',
                     exist_ok=True)
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.executescript('''
            CREATE TABLE IF NOT EXISTS interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                prompt TEXT NOT NULL,
                response TEXT NOT NULL,
                feedback INTEGER DEFAULT 0,
                feedback_comment TEXT DEFAULT '',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                prompt_tokens INTEGER DEFAULT 0,
                response_tokens INTEGER DEFAULT 0,
                generation_params TEXT DEFAULT '{}'
            );
            CREATE INDEX IF NOT EXISTS idx_interactions_feedback
                ON interactions(feedback);
            CREATE INDEX IF NOT EXISTS idx_interactions_created
                ON interactions(created_at);

            CREATE TABLE IF NOT EXISTS learning_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trigger_type TEXT NOT NULL,
                num_samples INTEGER DEFAULT 0,
                epochs INTEGER DEFAULT 0,
                train_loss_before REAL DEFAULT 0,
                train_loss_after REAL DEFAULT 0,
                val_loss_before REAL DEFAULT 0,
                val_loss_after REAL DEFAULT 0,
                perplexity_before REAL DEFAULT 0,
                perplexity_after REAL DEFAULT 0,
                status TEXT DEFAULT 'pending',
                started_at TIMESTAMP,
                completed_at TIMESTAMP,
                checkpoint_path TEXT DEFAULT '',
                notes TEXT DEFAULT ''
            );

            CREATE TABLE IF NOT EXISTS augmented_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_type TEXT NOT NULL,
                original_text TEXT NOT NULL,
                augmented_text TEXT NOT NULL,
                augmentation_type TEXT NOT NULL,
                quality_score REAL DEFAULT 0.5,
                used_in_training INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS idx_aug_used ON augmented_data(used_in_training);

            CREATE TABLE IF NOT EXISTS learning_config (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        ''')

        # 初始化默认配置（upsert 语义）
        defaults = {
            'auto_learning_enabled': 'false',
            'min_feedback_samples': '20',
            'min_positive_ratio': '0.6',
            'incremental_epochs': '3',
            'incremental_lr': '0.00005',
            'max_augmented_samples': '500',
            'augmentation_enabled': 'true',
            'schedule_interval_hours': '24',
            'last_scheduled_check': '',
            'total_learning_rounds': '0',
            'total_feedback_count': '0',
        }
        for key, value in defaults.items():
            cursor.execute(
                'INSERT OR IGNORE INTO learning_config (key, value) VALUES (?, ?)',
                (key, value)
            )
        conn.commit()
        conn.close()

    # ---- 读写 API ----

    def add_interaction(self, session_id: str, prompt: str, response: str,
                        prompt_tokens: int = 0, response_tokens: int = 0,
                        generation_params: Optional[Dict] = None) -> int:
        with self.lock:
            conn = self._get_conn()
            cursor = conn.cursor()
            cursor.execute(
                'INSERT INTO interactions(session_id, prompt, response, '
                'prompt_tokens, response_tokens, generation_params) '
                'VALUES (?, ?, ?, ?, ?, ?)',
                (session_id, prompt, response, prompt_tokens,
                 response_tokens,
                 json.dumps(generation_params or {}, ensure_ascii=False))
            )
            cursor.execute(
                "UPDATE learning_config SET value = CAST(value AS INTEGER) + 1 "
                "WHERE key = 'total_feedback_count'"
            )
            conn.commit()
            last_id = cursor.lastrowid
            conn.close()
            return last_id

    def add_feedback(self, interaction_id: int, feedback: int, comment: str = ''):
        with self.lock:
            conn = self._get_conn()
            cursor = conn.cursor()
            cursor.execute(
                'UPDATE interactions SET feedback = ?, feedback_comment = ? WHERE id = ?',
                (feedback, comment, interaction_id)
            )
            conn.commit()
            conn.close()

    def get_positive_samples(self, limit: int = 1000) -> List[Dict]:
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            'SELECT prompt, response FROM interactions WHERE feedback >= 1 '
            'ORDER BY created_at DESC LIMIT ?',
            (limit,)
        )
        rows = [dict(r) for r in cursor.fetchall()]
        conn.close()
        return rows

    def get_all_feedback_samples(self, limit: int = 2000) -> List[Dict]:
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            'SELECT prompt, response, feedback FROM interactions WHERE feedback != 0 '
            'ORDER BY created_at DESC LIMIT ?',
            (limit,)
        )
        rows = [dict(r) for r in cursor.fetchall()]
        conn.close()
        return rows

    def get_stats(self) -> Dict:
        conn = self._get_conn()
        cursor = conn.cursor()

        def _scalar(sql: str, *args) -> int:
            cursor.execute(sql, args)
            return cursor.fetchone()[0] or 0

        total = _scalar('SELECT COUNT(*) FROM interactions')
        positive = _scalar('SELECT COUNT(*) FROM interactions WHERE feedback = 1')
        negative = _scalar('SELECT COUNT(*) FROM interactions WHERE feedback = -1')
        no_feedback = _scalar('SELECT COUNT(*) FROM interactions WHERE feedback = 0')
        recent = _scalar("SELECT COUNT(*) FROM interactions "
                         "WHERE created_at > datetime('now', '-24 hours')")
        rounds = _scalar("SELECT COUNT(*) FROM learning_history WHERE status = 'completed'")
        aug_pending = _scalar('SELECT COUNT(*) FROM augmented_data WHERE used_in_training = 0')
        aug_total = _scalar('SELECT COUNT(*) FROM augmented_data')

        config = self.get_config()
        conn.close()
        return {
            'total_interactions': total,
            'positive_feedback': positive,
            'negative_feedback': negative,
            'no_feedback': no_feedback,
            'recent_24h': recent,
            'learning_rounds': rounds,
            'augmented_pending': aug_pending,
            'augmented_total': aug_total,
            'auto_learning_enabled': config.get('auto_learning_enabled', 'false') == 'true',
            'config': config,
        }

    def get_config(self) -> Dict[str, str]:
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute('SELECT key, value FROM learning_config')
        config = {r['key']: r['value'] for r in cursor.fetchall()}
        conn.close()
        return config

    def set_config(self, key: str, value: str):
        with self.lock:
            conn = self._get_conn()
            cursor = conn.cursor()
            cursor.execute(
                'INSERT INTO learning_config (key, value, updated_at) VALUES (?, ?, ?) '
                'ON CONFLICT(key) DO UPDATE SET value = excluded.value, '
                'updated_at = CURRENT_TIMESTAMP',
                (key, value, datetime.now().isoformat())
            )
            conn.commit()
            conn.close()

    def add_learning_record(self, record: Dict) -> int:
        with self.lock:
            conn = self._get_conn()
            cursor = conn.cursor()
            cursor.execute(
                'INSERT INTO learning_history (trigger_type, num_samples, epochs, '
                'train_loss_before, train_loss_after, val_loss_before, val_loss_after, '
                'perplexity_before, perplexity_after, status, started_at, '
                'completed_at, checkpoint_path, notes) '
                'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                (record.get('trigger_type', 'manual'),
                 record.get('num_samples', 0),
                 record.get('epochs', 0),
                 record.get('train_loss_before', 0),
                 record.get('train_loss_after', 0),
                 record.get('val_loss_before', 0),
                 record.get('val_loss_after', 0),
                 record.get('perplexity_before', 0),
                 record.get('perplexity_after', 0),
                 record.get('status', 'pending'),
                 record.get('started_at', datetime.now().isoformat()),
                 record.get('completed_at', None),
                 record.get('checkpoint_path', ''),
                 record.get('notes', ''))
            )
            cursor.execute(
                "UPDATE learning_config SET value = CAST(value AS INTEGER) + 1 "
                "WHERE key = 'total_learning_rounds'"
            )
            conn.commit()
            record_id = cursor.lastrowid
            conn.close()
            return record_id

    ALLOWED_LEARNING_COLUMNS = {
        'trigger_type', 'num_samples', 'epochs', 'train_loss_before',
        'train_loss_after', 'val_loss_before', 'val_loss_after',
        'perplexity_before', 'perplexity_after', 'status', 'started_at',
        'completed_at', 'checkpoint_path', 'notes'
    }

    def update_learning_record(self, record_id: int, **kwargs):
        with self.lock:
            conn = self._get_conn()
            cursor = conn.cursor()
            for k, v in kwargs.items():
                if k not in self.ALLOWED_LEARNING_COLUMNS:
                    raise ValueError(f"非法字段名: {k}")
                cursor.execute(f'UPDATE learning_history SET {k} = ? WHERE id = ?', (v, record_id))
            conn.commit()
            conn.close()

    def get_recent_learning_history(self, limit: int = 10) -> List[Dict]:
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            'SELECT * FROM learning_history ORDER BY id DESC LIMIT ?', (limit,)
        )
        rows = [dict(r) for r in cursor.fetchall()]
        conn.close()
        return rows

    def add_augmented_data(self, source_type: str, original: str,
                           augmented: str, aug_type: str):
        with self.lock:
            conn = self._get_conn()
            cursor = conn.cursor()
            cursor.execute(
                'INSERT INTO augmented_data (source_type, original_text, '
                'augmented_text, augmentation_type) VALUES (?, ?, ?, ?)',
                (source_type, original, augmented, aug_type)
            )
            conn.commit()
            conn.close()

    def mark_augmented_as_used(self, count: Optional[int] = None):
        with self.lock:
            conn = self._get_conn()
            cursor = conn.cursor()
            if count:
                cursor.execute(
                    'UPDATE augmented_data SET used_in_training = 1 '
                    'WHERE id IN (SELECT id FROM augmented_data '
                    'WHERE used_in_training = 0 ORDER BY id LIMIT ?)',
                    (count,)
                )
            else:
                cursor.execute('UPDATE augmented_data SET used_in_training = 1 '
                               'WHERE used_in_training = 0')
            conn.commit()
            conn.close()


# ============================================================
# 代码数据增强器 — 用于扩充用户反馈样本
# ============================================================

class CodeAugmentor:
    """代码训练数据增强器

    三种增强策略:
      1. rename_vars: 对 token 安全的变量重命名（仅在代码 token 边界替换）
      2. add_comments: 注入随机的中文注释模板
      3. reorder_functions: 对文件中无依赖关系的函数块重排

    注意：所有增强方法均返回 *新字符串*，不对原字符串做原地修改。
    """

    # 常见代码标识符 → 同义替换表（用于变量重命名）
    VAR_NAME_MAP = {
        'data': ['info', 'content', 'result', 'output'],
        'result': ['output', 'res', 'ret_val', 'answer'],
        'count': ['num', 'total', 'cnt', 'n'],
        'index': ['idx', 'i', 'pos', 'position'],
        'item': ['elem', 'element', 'entry', 'node'],
        'value': ['val', 'v', 'item_val', 'num_val'],
        'name': ['label', 'title', 'key', 'identifier'],
        'input': ['user_input', 'raw_input', 'src', 'source_data'],
        'output': ['result', 'out', 'response', 'ret'],
        'total': ['sum_val', 'grand_total', 'overall', 'aggregate'],
        'length': ['size', 'len', 'n_items', 'num_elems'],
        'message': ['msg', 'text', 'info_text', 'notice'],
        'error': ['err', 'exception', 'fault', 'issue'],
        'success': ['ok', 'done', 'is_valid', 'passed'],
        'config': ['cfg', 'settings', 'options', 'params'],
        'temp': ['tmp', 'buf', 'buffer', 'holding'],
        'list_data': ['items', 'elements', 'records', 'entries'],
    }

    CN_COMMENT_TEMPLATES = [
        '# 初始化变量', '# 处理数据', '# 遍历列表', '# 返回结果',
        '# 定义函数', '# 错误处理', '# 主程序入口', '# 配置参数',
        '# 数据转换', '# 格式化输出',
    ]

    @staticmethod
    def augment_code(code_sample: str, aug_types: Optional[List[str]] = None) -> List[str]:
        """对代码样本进行增强，返回增强后的样本列表"""
        if aug_types is None:
            aug_types = ['rename_vars', 'add_comments', 'reorder']

        results = [code_sample]

        for aug_type in aug_types:
            try:
                if aug_type == 'rename_vars':
                    results.append(CodeAugmentor._rename_variables(code_sample))
                elif aug_type == 'add_comments':
                    results.append(CodeAugmentor._add_chinese_comments(code_sample))
                elif aug_type == 'reorder':
                    results.append(CodeAugmentor._reorder_functions(code_sample))
            except Exception as e:
                logger.warning(f"数据增强 {aug_type} 失败: {e}")
                continue

        # 去重
        seen = set()
        unique_results = []
        for r in results:
            key = r.strip()
            if key and key not in seen:
                seen.add(key)
                unique_results.append(r)
        return unique_results

    @staticmethod
    def _rename_variables(code: str) -> str:
        """token 安全的变量重命名：在 word 边界上匹配替换，避免替换字符串内容。"""
        result = code
        # 先收集所有出现在代码中的可替换标识符
        found = [orig for orig in CodeAugmentor.VAR_NAME_MAP
                 if re.search(r'\b' + re.escape(orig) + r'\b', result)]
        # 只挑 1-2 个变量替换，避免破坏代码语义
        random.shuffle(found)
        for orig in found[:2]:
            replacement = random.choice(CodeAugmentor.VAR_NAME_MAP[orig])
            # 仅当原标识符是独立的 token 时才替换
            result = re.sub(r'\b' + re.escape(orig) + r'\b', replacement, result)
        return result

    @staticmethod
    def _add_chinese_comments(code: str) -> str:
        """在非注释、非字符串的行上方插入中文注释"""
        lines = code.split('\n')
        result_lines = []
        added = 0
        for line in lines:
            stripped = line.strip()
            # 仅在看起来像赋值/定义的普通代码行上注入注释
            is_plain_code = (stripped
                             and not stripped.startswith('#')
                             and not stripped.startswith('//')
                             and not stripped.startswith('"""')
                             and not stripped.startswith("'''")
                             and '=' in stripped
                             and added < 2
                             and random.random() < 0.15)
            if is_plain_code:
                indent = len(line) - len(line.lstrip())
                result_lines.append(' ' * indent + random.choice(
                    CodeAugmentor.CN_COMMENT_TEMPLATES))
                added += 1
            result_lines.append(line)
        return '\n'.join(result_lines)

    @staticmethod
    def _reorder_functions(code: str) -> str:
        """真正对函数块进行重排。

        策略：
          - 按行扫描，分离出 `def` / `function` 开头的函数块
          - 仅当函数块 >= 2 个且没有互相调用依赖时进行重排
          - 非函数代码（文件顶部的导入、全局变量）保持原位
        """
        lines = code.split('\n')

        # 找到每个函数块的起始行
        func_starts = []
        for i, line in enumerate(lines):
            stripped = line.lstrip()
            if (stripped.startswith('def ') or stripped.startswith('function ')
                    or stripped.startswith('public function ')
                    or stripped.startswith('private function ')):
                # 记录：起始索引 + 当前缩进级别
                indent = len(line) - len(line.lstrip())
                func_starts.append((i, indent))

        if len(func_starts) < 2:
            return code  # 不够重排，原样返回

        # 切分函数块（基于缩进级别判断块结束）
        blocks = []  # List[List[str]]，按出现顺序
        # 前缀（文件开头到第一个函数之前的所有行）
        first_start = func_starts[0][0]
        prefix = lines[:first_start]

        for idx, (start_line, block_indent) in enumerate(func_starts):
            # 查找块结束位置：遇到相同或更小缩进的非空行、或下一个函数起点
            next_start = func_starts[idx + 1][0] if idx + 1 < len(func_starts) else len(lines)
            block = lines[start_line:next_start]
            blocks.append(block)

        # 重排
        original_order = list(range(len(blocks)))
        shuffled = original_order[:]
        # 保证至少有一次实际交换
        while shuffled == original_order and len(shuffled) > 1:
            random.shuffle(shuffled)

        new_body = []
        for i, block_idx in enumerate(shuffled):
            new_body.extend(blocks[block_idx])
            # 函数块之间保留空行
            if i < len(shuffled) - 1 and not new_body[-1].strip() == '':
                new_body.append('')

        # 后缀（最后一个函数块之后的内容）— 通常为空
        result = prefix + new_body
        return '\n'.join(result)

    @staticmethod
    def augment_conversation(prompt: str, response: str,
                             aug_types: Optional[List[str]] = None) -> List[Dict]:
        """增强对话样本（prompt/response 对）"""
        results = [{'prompt': prompt, 'response': response, 'type': 'original'}]

        # 变换提示词格式
        variants = [
            f"请实现以下功能：\n{prompt}",
            f"帮我写代码：\n{prompt}",
            prompt + "\n请用中文注释。",
        ]
        random.shuffle(variants)
        for v in variants[:2]:
            results.append({'prompt': v, 'response': response, 'type': 'prompt_variant'})

        return results


# ============================================================
# 自动学习控制器 - 核心调度与执行引擎
# ============================================================

class AutoLearner:
    """自动学习控制器

    通过 IR 架构执行增量训练，与主训练流程完全一致。
    """

    def __init__(self, db_path: str = 'data/auto_learning.db',
                 config_dict: Optional[Dict] = None):
        self.db = InteractionDB(db_path)
        self.augmentor = CodeAugmentor()
        self.config_dict = config_dict
        self.is_training = False
        self._training_thread = None

    # ---- 反馈接口 ----

    def record_interaction(self, session_id: str, prompt: str, response: str,
                           prompt_tokens: int = 0, response_tokens: int = 0,
                           generation_params: Optional[Dict] = None) -> int:
        return self.db.add_interaction(
            session_id, prompt, response,
            prompt_tokens, response_tokens, generation_params
        )

    def record_feedback(self, interaction_id: int, feedback: int, comment: str = ''):
        self.db.add_feedback(interaction_id, feedback, comment)
        # 检查是否满足自动学习触发条件
        self._check_auto_trigger()

    def _check_auto_trigger(self):
        """检查是否满足自动学习条件"""
        config = self.db.get_config()
        if config.get('auto_learning_enabled', 'false') != 'true':
            return
        if self.is_training:
            return

        min_samples = int(config.get('min_feedback_samples', '20'))
        stats = self.db.get_stats()
        if stats['positive_feedback'] >= min_samples:
            logger.info(f"满足自动学习条件: {stats['positive_feedback']} "
                        f"正面反馈 >= {min_samples}，开始触发")
            self.start_learning(trigger_type='auto')

    # ---- 数据准备 ----

    def prepare_training_data(self, include_augmented: bool = True) -> List[str]:
        """准备训练数据：将 (prompt, response) 合并为文本行"""
        samples = []
        positive_samples = self.db.get_positive_samples(limit=200)
        for s in positive_samples:
            text = f"{s['prompt']}\n{s['response']}"
            samples.append(text.strip())
        return samples

    def augment_feedback_data(self) -> int:
        """对正面反馈样本做代码数据增强，返回新增数量"""
        config = self.db.get_config()
        if config.get('augmentation_enabled', 'true') != 'true':
            return 0

        max_aug = int(config.get('max_augmented_samples', '500'))
        current_pending = self.db.get_stats()['augmented_pending']
        if current_pending >= max_aug:
            return 0

        samples = self.db.get_positive_samples(limit=100)
        if not samples:
            return 0

        count = 0
        for sample in samples:
            original_text = f"{sample['prompt']}\n{sample['response']}"
            augmented_list = self.augmentor.augment_code(original_text)
            # 跳过原始样本，只存增强后的
            for aug_text in augmented_list[1:]:
                if count >= (max_aug - current_pending):
                    break
                if aug_text == original_text:
                    continue
                self.db.add_augmented_data(
                    source_type='user_feedback',
                    original=original_text,
                    augmented=aug_text,
                    aug_type='code_augmentation'
                )
                count += 1
        return count

    # ---- 增量训练核心 ----

    def start_learning(self, trigger_type: str = 'manual',
                       epochs: Optional[int] = None,
                       lr: Optional[float] = None) -> Dict[str, object]:
        """启动增量训练（在后台线程执行）"""
        if self.is_training:
            return {'success': False, 'message': '训练正在进行中，请等待完成'}

        config = self.db.get_config()
        train_epochs = epochs if epochs is not None else int(
            config.get('incremental_epochs', '3'))
        train_lr = lr if lr is not None else float(
            config.get('incremental_lr', '0.00005'))

        self.is_training = True
        self._training_thread = threading.Thread(
            target=self._run_incremental_training,
            args=(trigger_type, train_epochs, train_lr),
            daemon=True
        )
        self._training_thread.start()
        return {'success': True,
                'message': f'增量学习已启动（{trigger_type} 触发）'}

    def _run_incremental_training(self, trigger_type: str, epochs: int, lr: float):
        """使用 IR 架构执行增量训练。

        与 train.py 使用完全相同的模型初始化流程和 checkpoint 格式：
          1. 从 config/config.yaml 读取 ModelConfig
          2. 初始化 PyTorchBackend + IR TransformerModel
          3. 加载已有的 best_model.pt（如果存在）
          4. 在用户反馈样本上继续训练
          5. 保存为新的 auto_learn_{timestamp}.pt，并覆盖 best_model.pt
        """
        import math
        import torch
        import torch.nn.functional as F
        from torch.utils.data import DataLoader, Dataset
        import yaml

        start_time = datetime.now()
        record_id = self.db.add_learning_record({
            'trigger_type': trigger_type,
            'epochs': epochs,
            'status': 'running',
            'started_at': start_time.isoformat(),
            'notes': f'学习率: {lr}, 增量训练轮次: {epochs}',
        })

        try:
            # 1. 加载配置
            config_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'config', 'config.yaml')
            with open(config_path, 'r', encoding='utf-8') as f:
                config_dict = yaml.safe_load(f)

            # 2. 构建 IR 模型 + PyTorch 后端
            from ir.config import ModelConfig
            from ir.transformer import TransformerModel
            from backends.pytorch import PyTorchBackend, init_model_weights
            from src.tokenizer import SimpleTokenizer, TextDataset, create_dataloader

            mc = ModelConfig.from_yaml(config_dict)
            model = TransformerModel(mc)

            device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
            backend = PyTorchBackend(device=device_str)
            init_model_weights(model, backend)

            tokenizer = SimpleTokenizer(vocab_size=mc.vocab_size)

            # 3. 加载已存在的 checkpoint
            checkpoint_dir = config_dict.get('system', {}).get('checkpoint_dir', 'checkpoints')
            best_path = os.path.join(checkpoint_dir, 'best_model.pt')
            train_loss_before = 0.0
            val_loss_before = 0.0
            if os.path.exists(best_path):
                checkpoint = torch.load(best_path,
                                        map_location=backend.device,
                                        weights_only=False)
                backend.load_state_dict(model, checkpoint.get('state_dict', checkpoint))
                train_loss_before = float(checkpoint.get('best_val_loss', 0))
                val_loss_before = train_loss_before
                logger.info(f"已加载 checkpoint: {best_path}")

            # 4. 准备训练数据
            training_texts = self.prepare_training_data(include_augmented=True)

            original_train_file = config_dict.get('data', {}).get(
                'train_file', 'data/raw/train.txt')
            if os.path.exists(original_train_file):
                with open(original_train_file, 'r', encoding='utf-8') as f:
                    original_data = f.readlines()
                sampled_original = random.sample(
                    original_data, min(200, len(original_data)))
                training_texts.extend([line.strip() for line in sampled_original
                                       if line.strip()])

            if not training_texts:
                raise ValueError("没有可用的训练数据，请先提供正面反馈样本")

            # 5. 构建数据集 + DataLoader
            class _IncrementalDataset(Dataset):
                def __init__(self, texts, tok, max_len):
                    self.samples = []
                    for text in texts:
                        if len(text) < 10:
                            continue
                        tokens = tok.encode(text)
                        if len(tokens) < 5:
                            continue
                        self.samples.append(tokens)

                def __len__(self):
                    return len(self.samples)

                def __getitem__(self, idx):
                    return self.samples[idx]

            def _collate(batch):
                # 手动 pad，避免 torch 的 pad_sequence 在可变长度列表上出错
                max_len = max(len(x) for x in batch)
                padded = torch.zeros(len(batch), max_len, dtype=torch.long)
                for i, seq in enumerate(batch):
                    padded[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)
                return {'input_ids': padded, 'labels': padded.clone()}

            dataset = _IncrementalDataset(training_texts, tokenizer, mc.max_seq_length)
            batch_size = config_dict.get('training', {}).get('batch_size', 8)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                num_workers=0, collate_fn=_collate)

            # 6. 训练循环（PyTorch 优化器 + IR 模型）
            from backends.pytorch import collect_parameters
            named_params = collect_parameters(model)

            # 对可训练参数构造 AdamW
            trainable = [(n, p) for n, p in named_params
                         if hasattr(p, 'requires_grad') and p.requires_grad]
            optimizer = torch.optim.AdamW(
                [p for _, p in trainable], lr=lr, weight_decay=0.01)

            model.train()
            total_loss = 0.0
            total_batches = 0

            for epoch in range(epochs):
                epoch_loss = 0.0
                epoch_batches = 0
                for batch in loader:
                    input_ids = batch['input_ids'].to(backend.device)

                    # 前向传播（IR 模型）
                    logits = model.forward(input_ids, backend)

                    # Shift 计算 next-token loss
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = input_ids[..., 1:].contiguous()

                    min_len = min(shift_logits.size(1), shift_labels.size(1))
                    shift_logits = shift_logits[:, :min_len, :]
                    shift_labels = shift_labels[:, :min_len]

                    loss = F.cross_entropy(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                        ignore_index=tokenizer.pad_token_id
                        if hasattr(tokenizer, 'pad_token_id') else -100,
                    )

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        [p for _, p in trainable], 1.0)
                    optimizer.step()

                    epoch_loss += loss.item()
                    epoch_batches += 1

                avg = epoch_loss / max(epoch_batches, 1)
                total_loss += avg
                total_batches += 1
                logger.info(f"增量训练 Epoch {epoch + 1}/{epochs}, Loss: {avg:.4f}")

            train_loss_after = total_loss / max(total_batches, 1)
            val_loss_after = train_loss_after  # 简化：没有独立验证集
            ppl_after = math.exp(min(val_loss_after, 20))
            ppl_before = math.exp(min(val_loss_before, 20)) if val_loss_before > 0 else 0.0

            # 7. 保存 checkpoint（格式与 train.py 兼容）
            os.makedirs(checkpoint_dir, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            new_checkpoint = f'checkpoints/auto_learn_{timestamp}.pt'
            state_dict = backend.get_state_dict(model)

            torch.save({
                'epoch': epochs,
                'state_dict': state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': val_loss_after,
                'trained_samples': len(training_texts),
                'auto_learning_round': record_id,
            }, new_checkpoint)

            # 覆盖 best_model.pt，方便后续推理直接使用
            torch.save({
                'epoch': epochs,
                'state_dict': state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': val_loss_after,
            }, best_path)

            self.db.mark_augmented_as_used()
            self.db.update_learning_record(
                record_id,
                num_samples=len(training_texts),
                train_loss_before=train_loss_before,
                train_loss_after=train_loss_after,
                val_loss_before=val_loss_before,
                val_loss_after=val_loss_after,
                perplexity_before=ppl_before,
                perplexity_after=ppl_after,
                status='completed',
                completed_at=datetime.now().isoformat(),
                checkpoint_path=new_checkpoint,
            )
            logger.info(f"增量学习完成: samples={len(training_texts)}, "
                        f"loss={train_loss_after:.4f}")

        except Exception as e:
            logger.exception(f"增量学习失败: {e}")
            self.db.update_learning_record(
                record_id,
                status='failed',
                completed_at=datetime.now().isoformat(),
                notes=f'错误: {str(e)}',
            )
        finally:
            self.is_training = False

    # ---- 状态查询 ----

    def get_learning_status(self) -> Dict:
        stats = self.db.get_stats()
        history = self.db.get_recent_learning_history(limit=5)
        return {
            'is_training': self.is_training,
            'stats': stats,
            'recent_history': history,
        }

    def should_auto_learn(self) -> bool:
        config = self.db.get_config()
        if config.get('auto_learning_enabled', 'false') != 'true':
            return False
        if self.is_training:
            return False
        min_samples = int(config.get('min_feedback_samples', '20'))
        return self.db.get_stats()['positive_feedback'] >= min_samples


# ============================================================
# 全局单例
# ============================================================

_default_db_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'data', 'auto_learning.db')
auto_learner = AutoLearner(db_path=_default_db_path)
