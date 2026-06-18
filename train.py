"""
CodeSprite v2 训练入口 — 框架无关 IR 架构

用法:
  python train.py                  # 标准训练（自动选择设备）
  python train.py --mode auto      # 自动学习模式
  python train.py --find-lr        # 学习率查找器
  python train.py --no-amp         # 禁用混合精度
  python train.py --device cpu     # 强制 CPU 训练
  python train.py --no-cpu-fallback  # GPU 不可用时直接报错（不静默回退）
  python train.py --convert-old checkpoints/best_model.pt  # 转换旧权重

设备策略:
  训练：优先 GPU，自动回退 CPU（可通过 --no-cpu-fallback / 环境变量禁用回退）
  推理：默认 CPU，可按需切换 GPU
"""

import torch
import random
import numpy as np
import os
import sys
import argparse

# 添加项目根目录
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ir.config import Config, ModelConfig
from ir.transformer import TransformerModel
from backends.pytorch import PyTorchBackend, init_model_weights, collect_parameters
from training.trainer import Trainer
from src.tokenizer import SimpleTokenizer, TextDataset, create_dataloader
from src.device import resolve_device, print_device_info, warn_cpu_training, set_seed


def prepare_data(config):
    """准备训练/验证数据集和 DataLoader"""
    print("Initializing tokenizer...")
    tokenizer = SimpleTokenizer(vocab_size=config.model.vocab_size)

    print("Loading datasets...")
    train_dataset = TextDataset(
        config.data.train_file, tokenizer, config.model.max_seq_length
    )
    val_dataset = TextDataset(
        config.data.val_file, tokenizer, config.model.max_seq_length
    )

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")

    train_loader = create_dataloader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
    )
    val_loader = create_dataloader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
    )

    return tokenizer, train_loader, val_loader


def print_model_info(model, backend):
    """打印模型信息"""
    device_display = getattr(backend, "_resolved_device", "unknown")
    mc = model.config
    print(f"\n{'='*50}")
    print(f"CodeSprite v2 — Framework-Agnostic IR Architecture")
    print(f"{'='*50}")
    print(f"  Total parameters:     {model.get_param_count():,} "
          f"({model.get_param_count()/1e6:.1f}M)")
    print(f"  Hidden size:          {mc.hidden_size}")
    print(f"  Layers:               {mc.num_layers}")
    print(f"  Heads:                {mc.num_heads}")
    print(f"  KV Heads:             {mc.num_kv_heads}")
    print(f"  Vocab size:           {mc.vocab_size}")
    print(f"  Max sequence:         {mc.max_seq_length}")
    print(f"  Activation:           {mc.activation}")
    print(f"  RoPE:                 {mc.use_rope}")
    print(f"  Backend:              {backend.name} ({device_display})")
    print(f"{'='*50}\n")


def main():
    parser = argparse.ArgumentParser(description="CodeSprite v2 Training")
    parser.add_argument("--mode", type=str, default="standard",
                        choices=["standard", "auto", "find-lr"],
                        help="Training mode")
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision")
    parser.add_argument("--no-rope", action="store_true", help="Disable RoPE")
    parser.add_argument("--no-swiglu", action="store_true", help="Disable SwiGLU")
    parser.add_argument("--use-ema", action="store_true", help="Enable EMA")
    parser.add_argument("--label-smoothing", type=float, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", type=str, default=None,
                        help="Device: auto (default), cuda, cpu")
    parser.add_argument("--no-cpu-fallback", action="store_true",
                        help="Abort if GPU unavailable (sets CODESPRITE_ALLOW_CPU_FALLBACK=false)")
    parser.add_argument("--convert-old", type=str, default=None,
                        help="Convert old checkpoint to new format and exit")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint")
    args = parser.parse_args()

    # ---- 配置加载 + CLI 合并 ----
    config = Config.from_yaml("config/config.yaml")
    config.merge_from_args(args)

    set_seed(config.system.seed)

    # ---- 设备选择 ----
    if args.no_cpu_fallback:
        os.environ["CODESPRITE_ALLOW_CPU_FALLBACK"] = "false"

    resolved = resolve_device(args.device or config.system.device,
                              cpu_threads=config.system.cpu_threads)
    device = torch.device(resolved)

    print_device_info(resolved)

    if resolved == "cpu":
        warn_cpu_training()
    elif resolved == "cuda":
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "available"
        print(f"  GPU: {gpu_name}\n")

    # ---- 构建 IR 模型 ----
    mc = ModelConfig.from_yaml(config.to_dict())
    model = TransformerModel(mc)
    backend = PyTorchBackend(device=resolved)
    init_model_weights(model, backend)
    print_model_info(model, backend)

    # ---- 旧权重转换模式 ----
    if args.convert_old:
        from tools.convert_checkpoint import convert_old_to_new
        convert_old_to_new(args.convert_old, args.convert_old.replace(".pt", "_v2.pt"))
        return

    # ---- 加载已有检查点 ----
    best_path = os.path.join(config.system.checkpoint_dir, "best_model.pt")
    resume_path = args.resume or (best_path if os.path.exists(best_path) else None)
    if resume_path and os.path.exists(resume_path):
        print(f"\nLoading checkpoint: {resume_path}")
        backend.load_checkpoint(model, resume_path)

    # ---- 准备数据 ----
    tokenizer, train_loader, val_loader = prepare_data(config)

    os.makedirs(config.system.checkpoint_dir, exist_ok=True)
    os.makedirs(config.system.log_dir, exist_ok=True)

    # ---- 创建训练器 ----
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        backend=backend,
        config=config,
        tokenizer=tokenizer,
    )

    print("\n" + "=" * 50)
    print(f"Starting {args.mode} training!")
    print(f"  AMP: {config.training.use_amp}")
    print(f"  Label Smoothing: {config.training.label_smoothing}")
    print(f"  EMA: {config.training.use_ema}")
    print(f"  LR: {config.training.learning_rate}")
    print(f"  Epochs: {config.training.num_epochs}")
    print("=" * 50 + "\n")

    trainer.train()

    print("\nTraining finished! Running final evaluation...")
    val_metrics = trainer.evaluate()
    print(f"Final validation loss: {val_metrics['val_loss']:.4f}")
    print(f"Final perplexity: {val_metrics['perplexity']:.2f}")


if __name__ == "__main__":
    main()
