# import torch
# import triton

# # --- 1. Bypass lỗi set_allocator của Triton 3.8.0 ---
# if not hasattr(triton, 'set_allocator'):
#     triton.set_allocator = lambda *args, **kwargs: None

# # --- 2. Bypass toàn bộ các kiểu dữ liệu siêu mới chưa có trong PyTorch Stable ---
# missing_dtypes = [
#     'uint16', 'uint32', 'uint64',
#     'float4_e2m1fn_x2', 'float4_e2m1fn', 
#     'float8_e4m3fn', 'float8_e5m2', 
#     'float8_e4m3fnuz', 'float8_e5m2fnuz',
#     'float8_e8m0fnu', 'float8_e4m3b11fnuz'  # Bổ sung chuẩn mới bị thiếu
# ]
# for dtype in missing_dtypes:
#     if not hasattr(torch, dtype):
#         setattr(torch, dtype, torch.int8)  # Ép tạm thành int8 để khởi tạo
# # -----------------------------------------------------------------------

# # (Các dòng import và code của CLIMPPAR giữ nguyên bên dưới)

# (Các dòng code còn lại của bạn giữ nguyên, ví dụ: from model import CLIMPPAR...)
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

# Tắt toàn bộ các cảnh báo (FutureWarning, UserWarning,...) để log sạch sẽ
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import GradScaler  # Giữ lại cho backward compat, nhưng đã dùng torch.amp.GradScaler

# ============================================================================
# TỐI ƯU HÓA GPU AMPERE (RTX 30xx/40xx, A100)
# ============================================================================
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True  # Bật cuDNN để tối ưu tốc độ Conv/BN trên RTX 3090

# Thêm đường dẫn project
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(PROJECT_DIR)
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, PROJECT_DIR)

from config import CLIMPConfig, get_kaggle_config
from model import CLIMPPAR
from losses import CLIMPPARLoss, DisentanglementLoss
from dataset.clip_dataset import PARDataset, get_clip_transforms


# ============================================================================
# Logger
# ============================================================================
class Logger(object):
    """
    Tiện ích giúp tự động ghi log vào file văn bản song song với việc in ra màn hình.
    """
    def __init__(self, log_path):
        self.terminal = sys.stdout
        self.log = open(log_path, "a", encoding="utf-8")
        
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
        
    def flush(self):
        self.terminal.flush()
        self.log.flush()


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
    
    num_attrs = labels.shape[1]
    attr_accuracies = []
    per_attr_metrics = []
    
    for i in range(num_attrs):
        tp = ((preds_binary[:, i] == 1) & (labels[:, i] == 1)).sum()
        tn = ((preds_binary[:, i] == 0) & (labels[:, i] == 0)).sum()
        fp = ((preds_binary[:, i] == 1) & (labels[:, i] == 0)).sum()
        fn = ((preds_binary[:, i] == 0) & (labels[:, i] == 1)).sum()
        
        p = (labels[:, i] == 1).sum()
        n = (labels[:, i] == 0).sum()
        
        acc_pos = tp / max(p, 1)
        acc_neg = tn / max(n, 1)
        attr_mA = (acc_pos + acc_neg) / 2.0
        attr_accuracies.append(attr_mA)
        
        attr_acc = (tp + tn) / max(p + n, 1)
        attr_prec = tp / max(tp + fp, 1e-8)
        attr_rec = tp / max(tp + fn, 1e-8)
        attr_f1 = 2 * attr_prec * attr_rec / max(attr_prec + attr_rec, 1e-8)
        
        per_attr_metrics.append({
            'mA': float(attr_mA),
            'accuracy': float(attr_acc),
            'precision': float(attr_prec),
            'recall': float(attr_rec),
            'f1': float(attr_f1)
        })
        
    mA = np.mean(attr_accuracies)
    
    # Instance-level metrics
    correct_preds = (preds_binary == labels)
    accuracy = correct_preds.mean()
    
    tp_inst = (preds_binary * labels).sum(axis=1)
    fp_inst = (preds_binary * (1 - labels)).sum(axis=1)
    fn_inst = ((1 - preds_binary) * labels).sum(axis=1)
    
    precision = (tp_inst / (tp_inst + fp_inst + 1e-8)).mean()
    recall = (tp_inst / (tp_inst + fn_inst + 1e-8)).mean()
    f1 = (2 * precision * recall / (precision + recall + 1e-8))
    
    return {
        'mA': float(mA),
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'per_attribute': per_attr_metrics
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
def train_one_epoch(model, dataloader, criterion, disentangle_criterion, optimizer, scaler, scheduler, config, 
                    epoch, rank):
    """
    Huấn luyện 1 epoch.
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    optimizer.zero_grad(set_to_none=True)
    
    for batch_idx, (images, bg_images, labels, _) in enumerate(dataloader):
        # non_blocking=True giúp chuyển dữ liệu lên GPU bất đồng bộ
        images = images.to(rank if config.use_ddp else config.device, non_blocking=True)
        bg_images = bg_images.to(rank if config.use_ddp else config.device, non_blocking=True)
        labels = labels.to(rank if config.use_ddp else config.device, non_blocking=True)
        
        # Forward pass với AMP (Dùng bfloat16 thay vì float16 để chống tràn Gradient của kiến trúc Mamba)
        with torch.amp.autocast('cuda', enabled=config.use_amp, dtype=torch.bfloat16):
            logits, (disentangled, style_emb, bg_context) = model(images, bg_images, return_features=True)
            
            main_loss = criterion(logits, labels)
            dis_loss, l_cons, l_ground, l_ortho = disentangle_criterion(disentangled, style_emb, bg_context)
            
            # Kết hợp Loss chính và Loss triết lý (trọng số 0.6)
            loss = main_loss + 0.5 * dis_loss
            loss = loss / config.grad_accum_steps
        
        # [QUAN TRỌNG] Phát hiện NaN sớm: bỏ qua batch lỗi
        # Nhưng trong DDP, nếu rank 0 skip mà rank 1 chạy backward -> DEADLOCK ngay lập tức.
        # Do đó, thay vì skip (continue), ta cho loss = 0 và ép gradient = 0
        is_invalid = torch.isnan(loss) or torch.isinf(loss)
        
        if config.use_ddp:
            # Đồng bộ cờ lỗi giữa các GPU. Nếu bất kỳ GPU nào bị NaN, tất cả cùng skip
            invalid_tensor = torch.tensor(1.0 if is_invalid else 0.0, device=images.device)
            dist.all_reduce(invalid_tensor, op=dist.ReduceOp.MAX)
            is_invalid = invalid_tensor.item() > 0

        if is_invalid:
            if is_main_process(rank, config.use_ddp):
                print(f"  ⚠️  [Epoch {epoch+1}] Batch {batch_idx+1}: Loss=NaN/Inf detected trên ít nhất 1 GPU → Bỏ qua cập nhật")
            optimizer.zero_grad(set_to_none=True)
            continue
        
        # Backward pass
        scaler.scale(loss).backward()
        
        # Gradient accumulation step
        if (batch_idx + 1) % config.grad_accum_steps == 0:
            # [QUAN TRỌNG] Unscale và Gradient Clipping để chống nổ Gradient -> NaN
            scaler.unscale_(optimizer)
            
            # Tương tự, đồng bộ cờ NaN của gradient
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            is_grad_invalid = torch.isnan(grad_norm) or torch.isinf(grad_norm)
            if config.use_ddp:
                invalid_grad_tensor = torch.tensor(1.0 if is_grad_invalid else 0.0, device=images.device)
                dist.all_reduce(invalid_grad_tensor, op=dist.ReduceOp.MAX)
                is_grad_invalid = invalid_grad_tensor.item() > 0

            if is_grad_invalid:
                if is_main_process(rank, config.use_ddp):
                    print(f"  ⚠️  [Epoch {epoch+1}] Batch {batch_idx+1}: Grad NaN/Inf → Bỏ qua cập nhật")
                optimizer.zero_grad(set_to_none=True)
                scaler.update()
                continue
            
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)  # set_to_none=True giúp giải phóng VRAM nhanh hơn
            scheduler.step()  # Step LR mỗi optimizer step (không phải mỗi epoch)
        
        total_loss += loss.item() * config.grad_accum_steps
        num_batches += 1
        
        # Log
        if is_main_process(rank, config.use_ddp) and (batch_idx + 1) % config.log_interval == 0:
            avg_loss = total_loss / num_batches
            print(f"  [Epoch {epoch+1}] Batch {batch_idx+1}/{len(dataloader)} | "
                  f"Loss: {avg_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")
    
    return total_loss / max(num_batches, 1)


@torch.no_grad()
def evaluate(model, dataloader, criterion, disentangle_criterion, config, rank):
    """Đánh giá trên tập validation/test."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    for images, bg_images, labels, _ in dataloader:
        images = images.to(rank if config.use_ddp else config.device, non_blocking=True)
        bg_images = bg_images.to(rank if config.use_ddp else config.device, non_blocking=True)
        labels = labels.to(rank if config.use_ddp else config.device, non_blocking=True)
        
        with torch.amp.autocast('cuda', enabled=config.use_amp, dtype=torch.bfloat16):
            logits, (disentangled, style_emb, bg_context) = model(images, bg_images, return_features=True)
            
            main_loss = criterion(logits, labels)
            dis_loss, _, _, _ = disentangle_criterion(disentangled, style_emb, bg_context)
            loss = main_loss + 0.1 * dis_loss
        
        total_loss += loss.item()
        all_preds.append(torch.sigmoid(logits).float().cpu().numpy())
        all_labels.append(labels.float().cpu().numpy())
    
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    avg_loss = total_loss / max(len(dataloader), 1)
    metrics = compute_metrics(all_preds, all_labels)
    metrics['loss'] = avg_loss
    
    return metrics



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
        # Thiết lập Logger để ghi file văn bản
        log_file = os.path.join(config.output_dir, f"train_log_{time.strftime('%Y%m%d_%H%M%S')}.txt")
        sys.stdout = Logger(log_file)
        
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
        print(f"  AMP: {config.use_amp} (bfloat16)")
        print(f"  Epochs: {config.epochs}")
        print("=" * 60)
    
    # ========================
    # 1. Dataset & DataLoader
    # ========================
    train_transform, val_transform = get_clip_transforms(config.img_height, config.img_width)
    
    train_dataset = PARDataset(
        pkl_path=config.pkl_path,
        img_dir=config.img_dir,
        bg_img_dir=config.bg_img_dir,
        split='train',
        transform=train_transform,
        cache_in_memory=getattr(config, 'cache_in_memory', False)
    )
    val_dataset = PARDataset(
        pkl_path=config.pkl_path,
        img_dir=config.img_dir,
        bg_img_dir=config.bg_img_dir,
        split='val',
        transform=val_transform,
        cache_in_memory=getattr(config, 'cache_in_memory', False)
    )
    
    if main_proc:
        print(f"\nDataset: {config.dataset_name}")
        print(f"  Train: {len(train_dataset)} samples")
        print(f"  Val: {len(val_dataset)} samples")
        print(f"  Attributes: {train_dataset.attr_num}")
    
    # Sampler cho DDP
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True) \
        if config.use_ddp else None
    
    # [QUAN TRỌNG] TẮT DistributedSampler cho Validation
    # Vì evaluate() chỉ được gọi ở main_proc (GPU 0), nếu dùng DistributedSampler, GPU 0 sẽ chỉ chấm điểm trên 50% dataset.
    # Đặt thành None để GPU 0 nạp toàn bộ 100% dữ liệu validation, còn GPU 1 sẽ tự động đứng chờ (idle) sang epoch tiếp theo.
    val_sampler = None
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
        prefetch_factor=2 if config.num_workers > 0 else None,
        persistent_workers=True if config.num_workers > 0 else False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size * 2,  # Val không cần backward → batch lớn hơn
        shuffle=False,
        sampler=val_sampler,
        num_workers=config.num_workers,
        pin_memory=True,
        prefetch_factor=2 if config.num_workers > 0 else None,
        persistent_workers=True if config.num_workers > 0 else False
    )
    
    # ========================
    # 2. Model
    # ========================
    model = CLIMPPAR(config, attributes=train_dataset.attributes)
    model = model.to(device)
        
    if config.use_ddp:
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)
        # [QUAN TRỌNG] Bật static_graph để giải quyết xung đột giữa Gradient Checkpointing và DDP find_unused_parameters=True
        model._set_static_graph()
    
    if main_proc:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\nModel parameters:")
        print(f"  Total: {total_params / 1e6:.1f}M")
        print(f"  Trainable: {trainable_params / 1e6:.1f}M")
        
        # Tạm tắt đo FLOPs vì hàm profile gọi CUDA forward trước khi DataLoader fork worker
        # sẽ gây ra lỗi "Segmentation fault" trên hệ thống Linux.
        # try:
        #     from thop import profile, clever_format
        #     # Tạo tensor giả để đo kích thước
        #     dummy_img = torch.randn(1, 3, config.img_height, config.img_width).to(device)
        #     dummy_bg = torch.randn(1, 3, config.img_height, config.img_width).to(device)
        #     
        #     macs, params = profile(model, inputs=(dummy_img, dummy_bg), verbose=False)
        #     macs, params = clever_format([macs, params], "%.2f")
        #     
        #     print(f"  MACs/FLOPs (thop): {macs} (input: {config.img_height}x{config.img_width})")
        #     print(f"  Params (thop): {params}")
        # except ImportError:
        #     print("  (Chưa cài đặt 'thop' để đo MACs/FLOPs)")
    
    # Không còn cache_text_features vì prompt sinh động từ background (online)
    
    # ========================
    # 4. Loss, Optimizer, Scheduler
    # ========================
    criterion = CLIMPPARLoss(
        loss_type=config.loss_type,
        asl_gamma_neg=getattr(config, 'asl_gamma_neg', 4.0),
        asl_gamma_pos=getattr(config, 'asl_gamma_pos', 1.0),
        asl_clip=getattr(config, 'asl_clip', 0.05)
    )
    
    disentangle_criterion = DisentanglementLoss(w_cons=1.0, w_ground=1.0, w_ortho=1.0).to(device)
    
    # Chỉ optimize các tham số có requires_grad=True (bỏ qua frozen backbone)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
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
    
    # AMP GradScaler (bfloat16 không cần scale gradient như fp16, nên tắt đi để tránh xung đột)
    scaler = torch.amp.GradScaler('cuda', enabled=False)
    
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
            model, train_loader, criterion, disentangle_criterion, optimizer, scaler, scheduler,
            config, epoch, rank if config.use_ddp else device
        )
        # scheduler.step() đã được gọi bên trong train_one_epoch (mỗi optimizer step)
        
        epoch_time = time.time() - start_time
        
        # Eval
        if main_proc:
            # [QUAN TRỌNG] Truyền model.module để bypass DDP.
            # Nếu truyền DDP model, PyTorch sẽ ngầm tăng counter của rank 0, làm lệch (desync) sequence number với rank 1 -> Deadlock ở Epoch kế tiếp.
            eval_model = model.module if config.use_ddp else model
            metrics = evaluate(
                eval_model, val_loader, criterion, disentangle_criterion, config, 
                rank if config.use_ddp else device
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
        
        # [QUAN TRỌNG] Đồng bộ tất cả GPU sau khi GPU 0 hoàn thành Validation + Save
        # Nếu không có barrier, GPU 1 sẽ chạy trước vào epoch mới gây desync/deadlock
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
    parser = argparse.ArgumentParser(description='CLIMP-PAR Training')
    parser.add_argument('--pkl_path', type=str, default='', help='Path to dataset pkl')
    parser.add_argument('--img_dir', type=str, default='', help='Path to image directory')
    parser.add_argument('--bg_img_dir', type=str, default='', help='Path to background image directory')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size per GPU')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--loss_type', type=str, default='', help='Loss type (hybrid_asl_bce, asl, bce, hybrid_loss)')
    parser.add_argument('--no_ddp', action='store_true', help='Disable DDP (single GPU)')
    parser.add_argument('--no_amp', action='store_true', help='Disable AMP')
    parser.add_argument('--no_grad_ckpt', action='store_true', help='Tắt Gradient Checkpointing để tăng tốc (tốn VRAM hơn)')
    parser.add_argument('--output_dir', type=str, default='', help='Output directory')
    parser.add_argument('--cache_in_memory', action='store_true', help='Lưu toàn bộ dataset vào RAM để đọc siêu nhanh')
    args = parser.parse_args()
    
    config = get_kaggle_config()
    
    # Override from args
    if args.pkl_path:
        config.pkl_path = args.pkl_path
    if args.img_dir:
        config.img_dir = args.img_dir
    if args.bg_img_dir:
        config.bg_img_dir = args.bg_img_dir
    if args.no_grad_ckpt:
        config.gradient_checkpointing = False
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.epochs:
        config.epochs = args.epochs
    if args.lr:
        config.lr = args.lr
    if args.loss_type:
        config.loss_type = args.loss_type
    if args.no_ddp:
        config.use_ddp = False
    if args.no_amp:
        config.use_amp = False
    if args.cache_in_memory:
        config.cache_in_memory = True
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
