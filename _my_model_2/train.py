# train.py
# Script huấn luyện CLIMP-PAR trên Kaggle 2× T4 GPU
# Hỗ trợ: DDP, AMP (fp16), Gradient Accumulation, Gradient Checkpointing

import os
import sys
import time
import random
import argparse
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import GradScaler, autocast

# Thêm đường dẫn project
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(PROJECT_DIR)
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, PROJECT_DIR)

from config import CLIMPConfig, get_kaggle_config
from model import CLIMPPAR
from losses import CLIMPPARLoss
from dataset.clip_dataset import PARDataset, get_clip_transforms


# ============================================================================
# Metrics
# ============================================================================
def compute_metrics(predictions, labels, threshold=0.5):
    """
    Tính các metric cho PAR: mA, Accuracy, Precision, Recall, F1.
    
    Args:
        predictions: (N, num_attrs) — xác suất sau sigmoid
        labels: (N, num_attrs) — ground truth 0/1
        threshold: ngưỡng phân loại
    
    Returns:
        dict chứa các metric
    """
    preds_binary = (predictions >= threshold).astype(np.float32)
    
    # Mean Accuracy (mA) — trung bình accuracy mỗi thuộc tính
    num_attrs = labels.shape[1]
    attr_accuracies = []
    for i in range(num_attrs):
        tp = ((preds_binary[:, i] == 1) & (labels[:, i] == 1)).sum()
        tn = ((preds_binary[:, i] == 0) & (labels[:, i] == 0)).sum()
        p = (labels[:, i] == 1).sum()
        n = (labels[:, i] == 0).sum()
        acc_pos = tp / max(p, 1)
        acc_neg = tn / max(n, 1)
        attr_accuracies.append((acc_pos + acc_neg) / 2.0)
    mA = np.mean(attr_accuracies)
    
    # Instance-level metrics
    correct_preds = (preds_binary == labels)
    
    # Accuracy (instance-level, exact match ratio per attribute)
    accuracy = correct_preds.mean()
    
    # Precision, Recall, F1 (instance-level)
    tp = (preds_binary * labels).sum(axis=1)
    fp = (preds_binary * (1 - labels)).sum(axis=1)
    fn = ((1 - preds_binary) * labels).sum(axis=1)
    
    precision = (tp / (tp + fp + 1e-8)).mean()
    recall = (tp / (tp + fn + 1e-8)).mean()
    f1 = (2 * precision * recall / (precision + recall + 1e-8))
    
    return {
        'mA': float(mA),
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
    }


# ============================================================================
# Setup
# ============================================================================
def set_seed(seed):
    """Đặt random seed cho tính reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_ddp(rank, world_size):
    """Khởi tạo process group cho DDP."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_ddp():
    """Dọn dẹp DDP."""
    dist.destroy_process_group()


def is_main_process(rank, use_ddp):
    """Kiểm tra có phải process chính không (để in log, lưu checkpoint)."""
    return (not use_ddp) or (rank == 0)


# ============================================================================
# Training Loop
# ============================================================================
def train_one_epoch(model, dataloader, criterion, optimizer, scaler, config, 
                    epoch, rank, text_features_cache):
    """
    Huấn luyện 1 epoch.
    
    Args:
        text_features_cache: (N_attrs, embed_dim) — đã cache sẵn text features
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    optimizer.zero_grad()
    
    for batch_idx, (images, labels, _) in enumerate(dataloader):
        images = images.to(rank if config.use_ddp else config.device)
        labels = labels.to(rank if config.use_ddp else config.device)
        
        # Forward pass với AMP
        with autocast(enabled=config.use_amp):
            # Truyền cached_text_features qua forward để DDP hoạt động đúng
            logits, _ = model(images, cached_text_features=text_features_cache)
            
            loss = criterion(logits, labels)
            loss = loss / config.grad_accum_steps  # Scale loss cho gradient accumulation
        
        # Backward pass
        scaler.scale(loss).backward()
        
        # Gradient accumulation step
        if (batch_idx + 1) % config.grad_accum_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        total_loss += loss.item() * config.grad_accum_steps
        num_batches += 1
        
        # Log
        if is_main_process(rank, config.use_ddp) and (batch_idx + 1) % config.log_interval == 0:
            avg_loss = total_loss / num_batches
            print(f"  [Epoch {epoch+1}] Batch {batch_idx+1}/{len(dataloader)} | "
                  f"Loss: {avg_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")
    
    return total_loss / max(num_batches, 1)


@torch.no_grad()
def evaluate(model, dataloader, criterion, config, rank, text_features_cache):
    """Đánh giá trên tập validation/test."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    for images, labels, _ in dataloader:
        images = images.to(rank if config.use_ddp else config.device)
        labels = labels.to(rank if config.use_ddp else config.device)
        
        with autocast(enabled=config.use_amp):
            # Truyền cached_text_features qua forward
            logits, _ = model(images, cached_text_features=text_features_cache)
            
            loss = criterion(logits, labels)
        
        total_loss += loss.item()
        all_preds.append(torch.sigmoid(logits).cpu().numpy())
        all_labels.append(labels.cpu().numpy())
    
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    avg_loss = total_loss / max(len(dataloader), 1)
    metrics = compute_metrics(all_preds, all_labels)
    metrics['loss'] = avg_loss
    
    return metrics


@torch.no_grad()
def cache_text_features(model, dataset, config, device):
    """
    Cache text features CHO TẤT CẢ thuộc tính MỘT LẦN.
    Trong PAR, text prompts là cố định → không cần chạy text encoder mỗi batch.
    
    Returns:
        text_features: (N_attrs, embed_dim) — positive prompt features
    """
    model.eval()
    neg_prompts, pos_prompts = dataset.get_prompts()
    
    # Dùng positive prompts (ví dụ: "a photo of a pedestrian with hat")
    actual_model = model.module if config.use_ddp else model
    text_features = actual_model.text_encoder.encode_prompts(
        pos_prompts, batch_size=32
    )
    
    return text_features.to(device)


# ============================================================================
# Main
# ============================================================================
def main(rank=0, world_size=1, config=None):
    """Main training function (gọi trực tiếp hoặc qua DDP spawn)."""
    
    if config is None:
        config = get_kaggle_config()
    
    # DDP setup
    if config.use_ddp and world_size > 1:
        setup_ddp(rank, world_size)
        device = rank
    else:
        config.use_ddp = False
        device = config.device
    
    set_seed(config.seed + rank)
    
    main_proc = is_main_process(rank, config.use_ddp)
    if main_proc:
        print("=" * 60)
        print("CLIMP-PAR Training")
        print(f"  Device: {'DDP' if config.use_ddp else 'Single GPU'} "
              f"(world_size={world_size})")
        print(f"  VMamba: {config.vmamba_variant}")
        print(f"  Mamba Text: {config.mamba_model}")
        print(f"  Embed dim: {config.embed_dim}")
        print(f"  Batch size (per GPU): {config.batch_size}")
        print(f"  Grad accum: {config.grad_accum_steps}")
        print(f"  Effective batch: {config.batch_size * world_size * config.grad_accum_steps}")
        print(f"  AMP: {config.use_amp} (fp16)")
        print(f"  Epochs: {config.epochs}")
        print("=" * 60)
    
    # ========================
    # 1. Dataset & DataLoader
    # ========================
    train_transform, val_transform = get_clip_transforms(config.img_height, config.img_width)
    
    train_dataset = PARDataset(
        pkl_path=config.pkl_path,
        img_dir=config.img_dir,
        split='train',
        transform=train_transform
    )
    val_dataset = PARDataset(
        pkl_path=config.pkl_path,
        img_dir=config.img_dir,
        split='val',
        transform=val_transform
    )
    
    if main_proc:
        print(f"\nDataset: {config.dataset_name}")
        print(f"  Train: {len(train_dataset)} samples")
        print(f"  Val: {len(val_dataset)} samples")
        print(f"  Attributes: {train_dataset.attr_num}")
    
    # Sampler cho DDP
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True) \
        if config.use_ddp else None
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False) \
        if config.use_ddp else None
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size * 2,  # Val không cần backward → batch lớn hơn
        shuffle=False,
        sampler=val_sampler,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    # ========================
    # 2. Model
    # ========================
    model = CLIMPPAR(config)
    model = model.to(device)
    
    if config.use_ddp:
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    
    if main_proc:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\nModel parameters:")
        print(f"  Total: {total_params / 1e6:.1f}M")
        print(f"  Trainable: {trainable_params / 1e6:.1f}M")
    
    # ========================
    # 3. Cache text features
    # ========================
    if main_proc:
        print("\nCaching text features (one-time)...")
    text_features = cache_text_features(model, train_dataset, config, device)
    if main_proc:
        print(f"  Cached: {text_features.shape}")
    
    # ========================
    # 4. Loss, Optimizer, Scheduler
    # ========================
    criterion = CLIMPPARLoss(loss_type=config.loss_type)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay
    )
    
    # Cosine annealing with warmup
    total_steps = len(train_loader) * config.epochs // config.grad_accum_steps
    warmup_steps = len(train_loader) * config.warmup_epochs // config.grad_accum_steps
    
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + np.cos(np.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # AMP GradScaler (fp16 cho T4)
    scaler = GradScaler(enabled=config.use_amp)
    
    # ========================
    # 5. Training Loop
    # ========================
    best_mA = 0.0
    
    for epoch in range(config.epochs):
        if config.use_ddp:
            train_sampler.set_epoch(epoch)
        
        start_time = time.time()
        
        # Train
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, 
            config, epoch, rank if config.use_ddp else device, text_features
        )
        scheduler.step()
        
        epoch_time = time.time() - start_time
        
        # Eval
        if main_proc:
            metrics = evaluate(
                model, val_loader, criterion, config, 
                rank if config.use_ddp else device, text_features
            )
            
            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1}/{config.epochs} — {epoch_time:.1f}s")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss:   {metrics['loss']:.4f}")
            print(f"  mA:         {metrics['mA']:.4f}")
            print(f"  Accuracy:   {metrics['accuracy']:.4f}")
            print(f"  Precision:  {metrics['precision']:.4f}")
            print(f"  Recall:     {metrics['recall']:.4f}")
            print(f"  F1:         {metrics['f1']:.4f}")
            print(f"{'='*60}\n")
            
            # Save best model
            if metrics['mA'] > best_mA:
                best_mA = metrics['mA']
                save_path = os.path.join(config.output_dir, 'best_model.pth')
                save_model = model.module if config.use_ddp else model
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': save_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_mA': best_mA,
                    'metrics': metrics,
                    'config': config,
                }, save_path)
                print(f"  ★ Best model saved! mA={best_mA:.4f}")
            
            # Periodic checkpoint
            if (epoch + 1) % config.save_interval == 0:
                save_path = os.path.join(config.output_dir, f'checkpoint_epoch{epoch+1}.pth')
                save_model = model.module if config.use_ddp else model
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': save_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'best_mA': best_mA,
                }, save_path)
    
    if main_proc:
        print(f"\nTraining complete! Best mA: {best_mA:.4f}")
    
    if config.use_ddp:
        cleanup_ddp()


# ============================================================================
# Entry point
# ============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CLIMP-PAR Training')
    parser.add_argument('--pkl_path', type=str, default='', help='Path to dataset pkl')
    parser.add_argument('--img_dir', type=str, default='', help='Path to image directory')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size per GPU')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--no_ddp', action='store_true', help='Disable DDP (single GPU)')
    parser.add_argument('--no_amp', action='store_true', help='Disable AMP')
    parser.add_argument('--output_dir', type=str, default='', help='Output directory')
    args = parser.parse_args()
    
    config = get_kaggle_config()
    
    # Override from args
    if args.pkl_path:
        config.pkl_path = args.pkl_path
    if args.img_dir:
        config.img_dir = args.img_dir
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.epochs:
        config.epochs = args.epochs
    if args.lr:
        config.lr = args.lr
    if args.no_ddp:
        config.use_ddp = False
    if args.no_amp:
        config.use_amp = False
    if args.output_dir:
        config.output_dir = args.output_dir
        os.makedirs(config.output_dir, exist_ok=True)
    
    # Phát hiện số GPU
    world_size = torch.cuda.device_count()
    
    if config.use_ddp and world_size > 1:
        print(f"Launching DDP with {world_size} GPUs...")
        torch.multiprocessing.spawn(
            main,
            args=(world_size, config),
            nprocs=world_size,
            join=True
        )
    else:
        config.use_ddp = False
        main(rank=0, world_size=1, config=config)
