# train.py
# Script huấn luyện CLIMP-PAR v8.1 trên Kaggle 2× T4 GPU
# Pipeline: 3-branch (Text + Vision + Cross-Modal Mamba) với Spatial Cross-Attention
# Hỗ trợ: DDP, AMP (fp16), Gradient Accumulation
# v8.1 fixes: scheduler step-level, eval DDP gather, deprecated autocast

import os
import sys
import time
import random
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.amp import GradScaler, autocast

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
    tp = (preds_binary * labels).sum(axis=1)
    fp = (preds_binary * (1 - labels)).sum(axis=1)
    fn = ((1 - preds_binary) * labels).sum(axis=1)
    
    precision = (tp / (tp + fp + 1e-8)).mean()
    recall = (tp / (tp + fn + 1e-8)).mean()
    f1 = (2 * precision * recall / (precision + recall + 1e-8))
    
    # Accuracy (per-attribute average)
    accuracy = (preds_binary == labels).mean()
    
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
    """Kiểm tra có phải process chính không."""
    return (not use_ddp) or (rank == 0)


# ============================================================================
# Training Loop
# ============================================================================
def train_one_epoch(model, dataloader, criterion, optimizer, scaler, config,
                    epoch, device, attribute_names, scheduler=None):
    """
    Huấn luyện 1 epoch.
    
    v8.1: scheduler.step() được gọi sau mỗi optimizer step (không phải mỗi epoch).
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    rank = device if config.use_ddp else 0
    optimizer.zero_grad()
    
    for batch_idx, (images, labels, _, domain_tokens) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)
        domain_tokens = domain_tokens.to(device)
        
        # Forward pass với AMP
        with autocast('cuda', enabled=config.use_amp):
            actual_model = model.module if config.use_ddp else model
            logits, features = actual_model(images, domain_tokens, attribute_names)
            
            loss = criterion(logits, labels, features)
            loss = loss / config.grad_accum_steps
        
        # Backward pass
        scaler.scale(loss).backward()
        
        # Gradient accumulation step
        if (batch_idx + 1) % config.grad_accum_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            # [FIX] Scheduler step sau mỗi optimizer step (không phải mỗi epoch)
            if scheduler is not None:
                scheduler.step()
        
        total_loss += loss.item() * config.grad_accum_steps
        num_batches += 1
        
        # Log
        if is_main_process(rank, config.use_ddp) and (batch_idx + 1) % config.log_interval == 0:
            avg_loss = total_loss / num_batches
            print(f"  [Epoch {epoch+1}] Batch {batch_idx+1}/{len(dataloader)} | "
                  f"Loss: {avg_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")
    
    # Flush remaining gradients nếu batch cuối không chia hết cho grad_accum_steps
    if len(dataloader) % config.grad_accum_steps != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
    
    return total_loss / max(num_batches, 1)


@torch.no_grad()
def evaluate(model, dataloader, criterion, config, device, attribute_names, rank=0):
    """
    Đánh giá trên tập validation/test.
    
    v8.1: Gather predictions từ tất cả ranks khi dùng DDP để tính metric chính xác.
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    for images, labels, _, domain_tokens in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        domain_tokens = domain_tokens.to(device)
        
        with autocast('cuda', enabled=config.use_amp):
            actual_model = model.module if config.use_ddp else model
            logits, features = actual_model(images, domain_tokens, attribute_names)
            
            loss = criterion(logits, labels, features)
        
        total_loss += loss.item()
        all_preds.append(torch.sigmoid(logits))
        all_labels.append(labels)
    
    # Concat local predictions
    all_preds = torch.cat(all_preds, dim=0)   # (N_local, 57)
    all_labels = torch.cat(all_labels, dim=0) # (N_local, 57)
    
    # [FIX] Gather từ tất cả ranks khi dùng DDP
    if config.use_ddp and dist.is_initialized():
        world_size = dist.get_world_size()
        
        # Gather sizes trước (mỗi rank có thể có số sample khác nhau)
        local_size = torch.tensor([all_preds.shape[0]], device=device)
        all_sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
        dist.all_gather(all_sizes, local_size)
        max_size = max(s.item() for s in all_sizes)
        
        # Pad để all_gather đồng nhất shape
        if all_preds.shape[0] < max_size:
            pad_size = max_size - all_preds.shape[0]
            all_preds = torch.cat([all_preds, torch.zeros(pad_size, all_preds.shape[1], device=device)])
            all_labels = torch.cat([all_labels, torch.zeros(pad_size, all_labels.shape[1], device=device)])
        
        # Gather
        gathered_preds = [torch.zeros_like(all_preds) for _ in range(world_size)]
        gathered_labels = [torch.zeros_like(all_labels) for _ in range(world_size)]
        dist.all_gather(gathered_preds, all_preds)
        dist.all_gather(gathered_labels, all_labels)
        
        # Trim padding và concat
        final_preds = []
        final_labels = []
        for i in range(world_size):
            actual_size = all_sizes[i].item()
            final_preds.append(gathered_preds[i][:actual_size])
            final_labels.append(gathered_labels[i][:actual_size])
        
        all_preds = torch.cat(final_preds, dim=0)
        all_labels = torch.cat(final_labels, dim=0)
        
        # Tính loss trung bình qua các ranks
        total_loss_tensor = torch.tensor([total_loss], device=device)
        dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
        total_loss = total_loss_tensor.item() / world_size
    
    # Chuyển sang numpy
    all_preds = all_preds.cpu().numpy()
    all_labels = all_labels.cpu().numpy()
    
    avg_loss = total_loss / max(len(dataloader), 1)
    metrics = compute_metrics(all_preds, all_labels)
    metrics['loss'] = avg_loss
    
    return metrics


# ============================================================================
# Main
# ============================================================================
def main(rank=0, world_size=1, config=None):
    """Main training function."""
    
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
        print("CLIMP-PAR v8.1 Training (Spatial Cross-Attention)")
        print(f"  Device: {'DDP' if config.use_ddp else 'Single GPU'} "
              f"(world_size={world_size})")
        print(f"  VMamba Vision: {config.vmamba_variant} (backbone frozen)")
        print(f"  Mamba Text: {config.mamba_model} (backbone frozen)")
        print(f"  Embed dim: {config.embed_dim}")
        print(f"  Image size: {config.img_height}x{config.img_width}")
        print(f"  Batch size (per GPU): {config.batch_size}")
        print(f"  Grad accum: {config.grad_accum_steps}")
        print(f"  Effective batch: {config.batch_size * world_size * config.grad_accum_steps}")
        print(f"  AMP: {config.use_amp} (fp16)")
        print(f"  Epochs: {config.epochs}")
        print(f"  Loss: {config.loss_type}")
        print(f"  Domain tokens: {config.domain_tokens_path}")
        print("=" * 60)
    
    # ========================
    # 1. Dataset & DataLoader
    # ========================
    train_transform, val_transform = get_clip_transforms(config.img_height, config.img_width)
    
    train_dataset = PARDataset(
        pkl_path=config.pkl_path,
        img_dir=config.img_dir,
        split='train',
        transform=train_transform,
        domain_tokens_path=config.domain_tokens_path
    )
    val_dataset = PARDataset(
        pkl_path=config.pkl_path,
        img_dir=config.img_dir,
        split='val',
        transform=val_transform,
        domain_tokens_path=config.domain_tokens_path
    )
    
    # Lấy danh sách attribute names
    attribute_names = train_dataset.attributes
    
    if main_proc:
        print(f"\nDataset: {config.dataset_name}")
        print(f"  Train: {len(train_dataset)} samples")
        print(f"  Val: {len(val_dataset)} samples")
        print(f"  Attributes ({len(attribute_names)}): {attribute_names[:5]}...")
    
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
        batch_size=config.batch_size * 2,
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
        frozen_params = total_params - trainable_params
        print(f"\nModel parameters:")
        print(f"  Total:     {total_params / 1e6:.1f}M")
        print(f"  Trainable: {trainable_params / 1e6:.1f}M")
        print(f"  Frozen:    {frozen_params / 1e6:.1f}M")
        
        # Chi tiết các module có thể học
        print(f"\n  Trainable modules:")
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f"    {name}: {param.numel() / 1e3:.1f}K")
    
    # ========================
    # 3. Loss, Optimizer, Scheduler
    # ========================
    criterion = CLIMPPARLoss(config=config)
    
    # Chỉ optimize các tham số trainable
    trainable_params_list = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params_list,
        lr=config.lr,
        weight_decay=config.weight_decay
    )
    
    # [FIX] Cosine annealing with warmup — tính theo optimizer steps
    total_steps = len(train_loader) * config.epochs // config.grad_accum_steps
    warmup_steps = len(train_loader) * config.warmup_epochs // config.grad_accum_steps
    
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + np.cos(np.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # AMP GradScaler (fp16 cho T4)
    scaler = GradScaler('cuda', enabled=config.use_amp)
    
    # ========================
    # 4. Resume from checkpoint (nếu có)
    # ========================
    start_epoch = 0
    best_mA = 0.0
    resume_path = os.path.join(config.output_dir, 'latest_checkpoint.pth')
    if os.path.exists(resume_path):
        if main_proc:
            print(f"\nResuming from {resume_path}...")
        ckpt = torch.load(resume_path, map_location='cpu')
        save_model = model.module if config.use_ddp else model
        save_model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if 'scaler_state_dict' in ckpt:
            scaler.load_state_dict(ckpt['scaler_state_dict'])
        if 'scheduler_state_dict' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt.get('epoch', 0)
        best_mA = ckpt.get('best_mA', 0.0)
        if main_proc:
            print(f"  Resumed at epoch {start_epoch}, best mA: {best_mA:.4f}")
    
    # ========================
    # 5. Training Loop
    # ========================
    if main_proc:
        print(f"\nBắt đầu huấn luyện từ epoch {start_epoch + 1}...")
    
    for epoch in range(start_epoch, config.epochs):
        if config.use_ddp:
            train_sampler.set_epoch(epoch)
        
        start_time = time.time()
        
        # Train — [FIX] truyền scheduler vào để step mỗi optimizer step
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler,
            config, epoch, device, attribute_names, scheduler=scheduler
        )
        
        epoch_time = time.time() - start_time
        
        # [FIX] Eval trên TẤT CẢ ranks, gather predictions
        metrics = evaluate(
            model, val_loader, criterion, config, device, attribute_names, rank=rank
        )
        
        if main_proc:
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
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_mA': best_mA,
                    'metrics': metrics,
                    'config': vars(config),
                }, save_path)
                print(f"  ★ Best model saved! mA={best_mA:.4f}")
            
            # Latest checkpoint (để resume)
            save_model = model.module if config.use_ddp else model
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': save_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'best_mA': best_mA,
            }, os.path.join(config.output_dir, 'latest_checkpoint.pth'))
            
            # Periodic checkpoint
            if (epoch + 1) % config.save_interval == 0:
                save_path = os.path.join(config.output_dir, f'checkpoint_epoch{epoch+1}.pth')
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': save_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
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
    parser = argparse.ArgumentParser(description='CLIMP-PAR v8.1 Training')
    parser.add_argument('--pkl_path', type=str, default='', help='Path to dataset pkl')
    parser.add_argument('--img_dir', type=str, default='', help='Path to image directory')
    parser.add_argument('--domain_tokens_path', type=str, default='', help='Path to domain_tokens.pt')
    parser.add_argument('--batch_size', type=int, default=0, help='Batch size per GPU')
    parser.add_argument('--epochs', type=int, default=0, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0, help='Learning rate')
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
    if args.domain_tokens_path:
        config.domain_tokens_path = args.domain_tokens_path
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
