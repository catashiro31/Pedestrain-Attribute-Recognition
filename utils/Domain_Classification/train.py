import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import sys

# Đảm bảo có thể import các module ở thư mục hiện tại
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Đảm bảo có thể import VMamba từ thư mục gốc
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import từ các module đã tạo (cùng thư mục)
from dataloader import DomainDataLoader
from model import VMambaClassifier

def train(data_dir, num_epochs=10, batch_size=16, learning_rate=1e-4, save_dir="./checkpoints", checkpoint_path=None):
    # Tạo thư mục lưu model nếu chưa có
    os.makedirs(save_dir, exist_ok=True)
    
    # Thiết lập device (ưu tiên GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Sử dụng device: {device}")
    
    # 1. Chuẩn bị DataLoader (Train và Val)
    print("[*] Đang tải dữ liệu và chia tập Train/Validation...")
    loader_manager = DomainDataLoader(data_dir=data_dir, batch_size=batch_size)
    
    # Sử dụng hàm mới để chia tập dữ liệu (80% train, 20% val)
    train_loader, val_loader, class_names = loader_manager.get_train_val_loaders(val_split=0.2)
    
    num_classes = len(class_names)
    print(f"[*] Tìm thấy {num_classes} classes: {class_names}")
    print(f"[*] Số lượng batch - Train: {len(train_loader)} | Val: {len(val_loader)}")
    
    # 2. Khởi tạo mô hình
    print("[*] Đang khởi tạo mô hình VMamba Small...")
    model = VMambaClassifier(num_classes=num_classes, model_type='small')
    model = model.to(device)
    
    # 3. Định nghĩa hàm Loss và Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    best_val_acc = 0.0 # Biến để theo dõi độ chính xác tốt nhất trên tập Validation
    start_epoch = 0
    
    # [TÍNH NĂNG MỚI] Load Checkpoint nếu có
    if checkpoint_path and os.path.isfile(checkpoint_path):
        print(f"[*] Đang tải checkpoint từ: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Load weights mô hình
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load trạng thái optimizer để tiếp tục quá trình học (resume training)
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch']
            
        if 'accuracy' in checkpoint:
            best_val_acc = checkpoint['accuracy']
            
        print(f"[*] Đã tải thành công (từ epoch {start_epoch}, best_val_acc: {best_val_acc:.2f}%)")
    elif checkpoint_path:
        print(f"[!] Cảnh báo: Không tìm thấy checkpoint tại {checkpoint_path}. Bắt đầu train từ đầu.")
    
    # 4. Vòng lặp huấn luyện
    for epoch in range(start_epoch, num_epochs):
        
        # ==================== TRAINING PHASE ====================
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        
        for inputs, labels in train_progress:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            curr_loss = train_loss / (train_total / batch_size + 1e-5)
            curr_acc = 100. * train_correct / train_total
            train_progress.set_postfix({'loss': f"{curr_loss:.4f}", 'acc': f"{curr_acc:.2f}%"})
            
        epoch_train_loss = train_loss / len(train_loader)
        epoch_train_acc = 100. * train_correct / train_total
        
        # ==================== VALIDATION PHASE ====================
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        # Biến đếm để tính độ chính xác từng class trên tập Val
        class_correct = [0] * num_classes
        class_total = [0] * num_classes
        
        val_progress = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]  ")
        
        with torch.no_grad(): # Tắt tính toán gradient
            for inputs, labels in val_progress:
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                # Cập nhật số liệu cho từng class
                for i in range(labels.size(0)):
                    label = labels[i].item()
                    pred = predicted[i].item()
                    class_total[label] += 1
                    if label == pred:
                        class_correct[label] += 1
                        
                curr_loss = val_loss / (val_total / batch_size + 1e-5)
                curr_acc = 100. * val_correct / val_total
                val_progress.set_postfix({'loss': f"{curr_loss:.4f}", 'acc': f"{curr_acc:.2f}%"})
                
        epoch_val_loss = val_loss / len(val_loader)
        epoch_val_acc = 100. * val_correct / val_total
        
        # ==================== END OF EPOCH SUMMARY ====================
        print(f"\n=> Kết thúc Epoch [{epoch+1}/{num_epochs}]")
        print(f"   Train Loss: {epoch_train_loss:.4f} | Train Acc (Overall): {epoch_train_acc:.2f}%")
        print(f"   Val Loss:   {epoch_val_loss:.4f} | Val Acc (Overall):   {epoch_val_acc:.2f}%")
        
        # In độ chính xác từng class
        print("   Độ chính xác từng class (Validation):")
        for i in range(num_classes):
            if class_total[i] > 0:
                acc_i = 100. * class_correct[i] / class_total[i]
                print(f"    - {class_names[i]}: {acc_i:.2f}% ({class_correct[i]}/{class_total[i]})")
            else:
                print(f"    - {class_names[i]}: N/A (0 sample)")
        
        # 5. Chỉ lưu mô hình có độ chính xác Val Acc tốt nhất (best model)
        if epoch_val_acc > best_val_acc:
            print(f"[*] Cập nhật best model (Val Acc tăng từ {best_val_acc:.2f}% -> {epoch_val_acc:.2f}%)")
            best_val_acc = epoch_val_acc
            save_path = os.path.join(save_dir, "vmamba_small_best_model.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_val_loss,
                'accuracy': best_val_acc,
                'class_names': class_names
            }, save_path)
            
        print("-" * 50)
    
    print("[*] Huấn luyện hoàn tất!")

if __name__ == "__main__":
    # Đường dẫn thư mục dữ liệu
    DATA_DIR = "/media/catashiro31/DATA/Nghiên cứu khoa học/Pedestrain Attribute Recognition/MSP60k/SUBMIT/domain_images"
    
    # Đường dẫn thư mục để lưu các file weights (.pth)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    SAVE_DIR = os.path.join(current_dir, "checkpoints")
    
    # [TÍNH NĂNG MỚI] Gắn đường dẫn checkpoint vào đây nếu muốn tiếp tục train (resume)
    # Ví dụ: CHECKPOINT_PATH = os.path.join(SAVE_DIR, "vmamba_small_best_model.pth")
    CHECKPOINT_PATH = None
    
    # Gọi hàm huấn luyện
    train(
        data_dir=DATA_DIR,
        num_epochs=20,            # Số vòng lặp huấn luyện
        batch_size=16,            # Kích thước mỗi batch
        learning_rate=1e-4,       # Tốc độ học (learning rate)
        save_dir=SAVE_DIR,
        checkpoint_path=CHECKPOINT_PATH
    )
