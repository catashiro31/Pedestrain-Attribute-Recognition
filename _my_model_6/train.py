# train.py
# Script huấn luyện CLIMP-PAR v6 trên Kaggle 2× T4 GPU
# Hỗ trợ: DDP, AMP (fp16), Gradient Accumulation, Gradient Checkpointing
#
# v6 thay đổi:
#   - Cache word embeddings (tĩnh) thay vì cache text features hoàn chỉnh
#   - Forward pass sử dụng CoCoOp mode (cached_word_embeddings)
#   - Text features được sinh online mỗi batch (phụ thuộc ảnh đầu vào)

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
                    epoch, rank, word_embeddings_cache, word_mask_cache):
    """
    Huấn luyện 1 epoch.
    
    v6: Sử dụng cached word embeddings thay vì cached text features.
    Text features được sinh ONLINE mỗi batch bởi ConditionalPromptLearner + Text Encoder.
    
    Args:
        word_embeddings_cache: (N_attrs, L_w, d_model) — word embeddings đã cache
        word_mask_cache: (N_attrs, L_w) — attention mask cho word embeddings
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    optimizer.zero_grad(set_to_none=True)
    
    device = rank if config.use_ddp else config.device
    for batch_idx, (images, labels, _) in enumerate(dataloader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        # Forward pass với AMP
        with autocast(enabled=config.use_amp):
            # v6: Truyền cached_word_embeddings (CoCoOp mode)
            logits, _ = model(
                images, 
                cached_word_embeddings=word_embeddings_cache,
                cached_word_mask=word_mask_cache
            )
            
            loss = criterion(logits, labels)
            loss = loss / config.grad_accum_steps  # Scale loss cho gradient accumulation
        
        # Backward pass
        scaler.scale(loss).backward()
        
        # Gradient accumulation step
        if (batch_idx + 1) % config.grad_accum_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        
        total_loss += loss.item() * config.grad_accum_steps
        num_batches += 1
        
        # Log
        if is_main_process(rank, config.use_ddp) and (batch_idx + 1) % config.log_interval == 0:
            avg_loss = total_loss / num_batches
            print(f"  [Epoch {epoch+1}] Batch {batch_idx+1}/{len(dataloader)} | "
                  f"Loss: {avg_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")
    
    return total_loss / max(num_batches, 1)


@torch.inference_mode()
def evaluate(model, dataloader, criterion, config, rank, word_embeddings_cache, word_mask_cache):
    """
    Đánh giá trên tập validation/test.
    v6: Sử dụng cached word embeddings (CoCoOp mode).
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    device = rank if config.use_ddp else config.device
    for images, labels, _ in dataloader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        with autocast(enabled=config.use_amp):
            # v6: Truyền cached_word_embeddings (CoCoOp mode)
            logits, _ = model(
                images, 
                cached_word_embeddings=word_embeddings_cache,
                cached_word_mask=word_mask_cache
            )
            
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


@torch.inference_mode()
def cache_word_embeddings(model, dataset, config, device):
    """
    v6: Cache word embeddings (KHÔNG phải text features hoàn chỉnh).
    
    Word embeddings là kết quả chỉ qua embedding layer (lookup table),
    CHƯA qua SSM blocks. Chúng cố định vì chỉ phụ thuộc vào prompt text.
    
    ConditionalPromptLearner sẽ dùng word embeddings này + vision feature 
    để sinh prompt embeddings riêng cho mỗi ảnh.
    
    Returns:
        word_embeddings: (N_attrs, L_w, d_model) — word embeddings thô
        word_mask: (N_attrs, L_w) — attention mask
    """
    model.eval()
    neg_prompts, pos_prompts = dataset.get_prompts()
    
    # Dùng positive prompts (ví dụ: "a photo of a pedestrian with hat")
    actual_model = model.module if config.use_ddp else model
    word_embeddings, word_mask = actual_model.text_encoder.get_word_embeddings(
        pos_prompts, device=device
    )
    
    return word_embeddings.to(device), word_mask.to(device)


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
    
    # ========================
    # CUDA Backend Optimizations
    # ========================
    torch.backends.cudnn.benchmark = True        # Tự tìm conv algorithm nhanh nhất
    torch.backends.cuda.matmul.allow_tf32 = True # TF32 cho matmul (Turing+)
    torch.backends.cudnn.allow_tf32 = True       # TF32 cho cudnn
    
    main_proc = is_main_process(rank, config.use_ddp)
    if main_proc:
        print("=" * 60)
        print("CLIMP-PAR v6 Training (CoCoOp Conditional Prompt Learning)")
        print(f"  Device: {'DDP' if config.use_ddp else 'Single GPU'} "
              f"(world_size={world_size})")
        print(f"  VMamba: {config.vmamba_variant}")
        print(f"  Mamba Text: {config.mamba_model} (freeze={config.mamba_freeze})")
        print(f"  CoCoOp: n_ctx={config.n_ctx}, meta_hidden={config.meta_net_hidden_dim}")
        print(f"  Embed dim: {config.embed_dim}")
        print(f"  Batch size (per GPU): {config.batch_size}")
        print(f"  Grad accum: {config.grad_accum_steps}")
        print(f"  Effective batch: {config.batch_size * world_size * config.grad_accum_steps}")
        print(f"  AMP: {config.use_amp} (fp16)")
        print(f"  Gradient Checkpointing: {config.gradient_checkpointing}")
        print(f"  Text Forward Batch Size: {config.text_forward_batch_size}")
        print(f"  torch.compile: {config.use_compile}")
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
    
    dl_kwargs = {}
    if config.num_workers > 0:
        dl_kwargs['persistent_workers'] = True   # Giữ workers sống giữa các epoch
        dl_kwargs['prefetch_factor'] = 2          # Pre-load sẵn batches
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
        **dl_kwargs
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size * 2,  # Val không cần backward → batch lớn hơn
        shuffle=False,
        sampler=val_sampler,
        num_workers=config.num_workers,
        pin_memory=True,
        **dl_kwargs
    )
    
    # ========================
    # 2. Model
    # ========================
    model = CLIMPPAR(config)
    model = model.to(device)
    
    # torch.compile (PyTorch 2.0+)
    if getattr(config, 'use_compile', False):
        if main_proc:
            print("\nApplying torch.compile(mode='reduce-overhead')...")
        model = torch.compile(model, mode='reduce-overhead')
    
    if config.use_ddp:
        model = DDP(
            model, device_ids=[rank],
            find_unused_parameters=getattr(config, 'ddp_find_unused_params', False)
        )
    
    if main_proc:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\nModel parameters:")
        print(f"  Total: {total_params / 1e6:.1f}M")
        print(f"  Trainable: {trainable_params / 1e6:.1f}M")
        
        # In chi tiết params của CoCoOp modules
        prompt_params = sum(p.numel() for p in (model.module if config.use_ddp else model).prompt_learner.parameters())
        print(f"  CoCoOp Prompt Learner: {prompt_params / 1e3:.1f}K")
    
    # ========================
    # 3. Cache word embeddings (v6: KHÔNG cache text features hoàn chỉnh)
    # ========================
    if main_proc:
        print("\nCaching word embeddings (one-time, static)...")
    word_embs, word_mask = cache_word_embeddings(model, train_dataset, config, device)
    if main_proc:
        print(f"  Word embeddings: {word_embs.shape}")
        print(f"  Word mask: {word_mask.shape}")
    
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
    # 5. Resume from checkpoint (nếu có)
    # ========================
    start_epoch = 0
    best_mA = 0.0
    
    if config.resume_from and os.path.isfile(config.resume_from):
        if main_proc:
            print(f"\nResuming from checkpoint: {config.resume_from}")
        
        checkpoint = torch.load(config.resume_from, map_location=f'cuda:{device}' if isinstance(device, int) else device, weights_only=False)
        
        # Load model state (strict=False để bỏ qua keys thừa/thiếu, ví dụ VMamba classifier head)
        load_model = model.module if config.use_ddp else model
        load_result = load_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        if main_proc and (load_result.missing_keys or load_result.unexpected_keys):
            if load_result.missing_keys:
                print(f"  ⚠ Missing keys: {load_result.missing_keys[:5]}...")
            if load_result.unexpected_keys:
                print(f"  ⚠ Unexpected keys (ignored): {load_result.unexpected_keys[:5]}...")
        
        # Load optimizer state (có thể fail nếu param groups thay đổi)
        try:
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except (ValueError, RuntimeError) as e:
            if main_proc:
                print(f"  ⚠ Không load được optimizer state (param groups thay đổi), dùng optimizer mới.")
                print(f"    Lý do: {e}")
        
        # Load scheduler state
        scheduler_loaded = False
        try:
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                scheduler_loaded = True
        except (ValueError, RuntimeError, KeyError) as e:
            if main_proc:
                print(f"  ⚠ Không load được scheduler state, sẽ tính lại từ start_epoch.")
        
        # Load scaler state (AMP)
        try:
            if 'scaler_state_dict' in checkpoint:
                scaler.load_state_dict(checkpoint['scaler_state_dict'])
        except (ValueError, RuntimeError) as e:
            if main_proc:
                print(f"  ⚠ Không load được scaler state, dùng scaler mới.")
        
        # Restore epoch & best metric
        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch']  # checkpoint lưu epoch+1, nên đây là epoch tiếp theo
        if 'best_mA' in checkpoint:
            best_mA = checkpoint['best_mA']
        
        # Nếu scheduler không load được, advance đến đúng vị trí
        if not scheduler_loaded and start_epoch > 0:
            for _ in range(start_epoch):
                scheduler.step()
            if main_proc:
                print(f"  ℹ Scheduler advanced to step {start_epoch}, LR={optimizer.param_groups[0]['lr']:.2e}")
        
        if main_proc:
            print(f"  Resumed from epoch {start_epoch}, best_mA={best_mA:.4f}")
        
        del checkpoint
        torch.cuda.empty_cache()
    elif config.resume_from:
        if main_proc:
            print(f"\n⚠ Checkpoint not found: {config.resume_from}, training from scratch.")
    
    # ========================
    # 6. Training Loop
    # ========================
    
    for epoch in range(start_epoch, config.epochs):
        if config.use_ddp:
            train_sampler.set_epoch(epoch)
        
        start_time = time.time()
        
        # Train
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, 
            config, epoch, rank if config.use_ddp else device, 
            word_embs, word_mask
        )
        scheduler.step()
        
        epoch_time = time.time() - start_time
        
        # Eval — TẤT CẢ ranks cùng evaluate (tránh DDP timeout)
        metrics = evaluate(
            model, val_loader, criterion, config, 
            rank if config.use_ddp else device, 
            word_embs, word_mask
        )
        
        # Chỉ rank 0 in kết quả và lưu checkpoint
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
                    'scaler_state_dict': scaler.state_dict(),
                    'best_mA': best_mA,
                    'metrics': metrics,
                    'config': config,
                }, save_path)
                print(f"  ★ Best model saved! mA={best_mA:.4f}")
            
            # Save last model (luôn ghi đè → dùng để continue training)
            last_save_path = os.path.join(config.output_dir, 'last_model.pth')
            save_model = model.module if config.use_ddp else model
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': save_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'best_mA': best_mA,
                'metrics': metrics,
            }, last_save_path)
            
            # Periodic checkpoint
            if (epoch + 1) % config.save_interval == 0:
                save_path = os.path.join(config.output_dir, f'checkpoint_epoch{epoch+1}.pth')
                save_model = model.module if config.use_ddp else model
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': save_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'best_mA': best_mA,
                }, save_path)
        
        # Đồng bộ tất cả ranks sau khi save (tránh DDP timeout)
        if config.use_ddp:
            dist.barrier()
    
    if main_proc:
        print(f"\nTraining complete! Best mA: {best_mA:.4f}")
    
    if config.use_ddp:
        cleanup_ddp()


# ============================================================================
# Entry point
# ============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CLIMP-PAR v6 Training (CoCoOp)')
    parser.add_argument('--pkl_path', type=str, default='', help='Path to dataset pkl')
    parser.add_argument('--img_dir', type=str, default='', help='Path to image directory')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size per GPU')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--n_ctx', type=int, default=4, help='Number of learnable context tokens')
    parser.add_argument('--no_ddp', action='store_true', help='Disable DDP (single GPU)')
    parser.add_argument('--no_amp', action='store_true', help='Disable AMP')
    parser.add_argument('--no_freeze', action='store_true', help='Do not freeze text backbone')
    parser.add_argument('--output_dir', type=str, default='', help='Output directory')
    parser.add_argument('--resume', type=str, default='', help='Path to checkpoint for continue training')
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
    if args.n_ctx:
        config.n_ctx = args.n_ctx
    if args.no_ddp:
        config.use_ddp = False
    if args.no_amp:
        config.use_amp = False
    if args.no_freeze:
        config.mamba_freeze = False
    if args.output_dir:
        config.output_dir = args.output_dir
        os.makedirs(config.output_dir, exist_ok=True)
    if args.resume:
        config.resume_from = args.resume
    
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
