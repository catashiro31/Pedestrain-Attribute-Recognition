import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import datasets, transforms, models
from torchvision.models import ResNet18_Weights

# --- Custom Dataset Wrapper để áp dụng Transform khác biệt cho Train/Val/Test ---
class SubsetWithTransform(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y
        
    def __len__(self):
        return len(self.subset)

def main():
    # ==========================
    # 1. Cấu hình Hyperparameters
    # ==========================
    data_dir = "../domain_noper_images_inpainted_refined"
    batch_size = 32
    num_epochs = 20
    learning_rate = 1e-4
    
    # Thiết lập device (GPU/CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Đang sử dụng thiết bị: {device}")
    
    # ==========================
    # 2. Định nghĩa Transforms (Chuẩn ImageNet)
    # ==========================
    # Data Augmentation cho tập Train
    train_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Chỉ Resize & Normalize cho tập Val / Test
    val_test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # ==========================
    # 3. Load & Split Dataset
    # ==========================
    # Load ImageFolder (chưa gắn transform để xử lý phân chia trước)
    full_dataset = datasets.ImageFolder(root=data_dir)
    class_names = full_dataset.classes
    num_classes = len(class_names)
    
    print(f"\nTìm thấy {len(full_dataset)} ảnh thuộc {num_classes} lớp: {class_names}")

    # Chia 70% Train, 15% Val, 15% Test
    total_size = len(full_dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size

    # Cố định random seed để kết quả chia luôn giống nhau ở các lần chạy
    generator = torch.Generator().manual_seed(42)
    train_subset, val_subset, test_subset = random_split(
        full_dataset, [train_size, val_size, test_size], generator=generator
    )

    # Gắn transform tương ứng bằng wrapper
    train_dataset = SubsetWithTransform(train_subset, transform=train_transforms)
    val_dataset = SubsetWithTransform(val_subset, transform=val_test_transforms)
    test_dataset = SubsetWithTransform(test_subset, transform=val_test_transforms)

    print(f"- Số lượng Train: {len(train_dataset)}")
    print(f"- Số lượng Validation: {len(val_dataset)}")
    print(f"- Số lượng Test: {len(test_dataset)}\n")

    # Tạo DataLoaders
    # set num_workers=4 để tăng tốc load data (điều chỉnh lại nếu máy báo lỗi bộ nhớ)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # ==========================
    # 4. Khởi tạo Mô hình ResNet
    # ==========================
    # Lưu ý: PyTorch chuẩn không có kiến trúc "resnet11", gần nhất và phổ biến nhất là "resnet18".
    # Tôi sẽ sử dụng ResNet-18 cho kịch bản này.
    print("Khởi tạo mô hình ResNet-18 với trọng số pre-train từ ImageNet...")
    model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    
    # Sửa lớp Linear cuối cùng (fc) cho khớp với số class của dataset
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    model = model.to(device)

    # ==========================
    # 5. Cấu hình Huấn luyện
    # ==========================
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Lưu trọng số của epoch tốt nhất
    best_val_acc = 0.0
    save_path = "resnet18_background_best.pth"

    print("\nBẮT ĐẦU HUẤN LUYỆN...")
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # --- Vòng lặp Train ---
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            
        epoch_train_loss = running_loss / train_size
        epoch_train_acc = running_corrects.double() / train_size
        
        # --- Vòng lặp Validation ---
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)
                
        epoch_val_loss = val_loss / val_size
        epoch_val_acc = val_corrects.double() / val_size
        
        time_elapsed = time.time() - start_time
        
        print(f"Epoch {epoch+1}/{num_epochs} [{time_elapsed:.0f}s] - "
              f"Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.4f} | "
              f"Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.4f}")
              
        # Cập nhật và lưu mô hình tốt nhất
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), save_path)
            print(f" --> [LƯU MÔ HÌNH] Best Validation Accuracy: {best_val_acc:.4f}")

    # ==========================
    # 6. Đánh giá trên tập Test
    # ==========================
    print("\nBẮT ĐẦU ĐÁNH GIÁ TRÊN TẬP TEST...")
    # Tải lại trọng số tốt nhất
    model.load_state_dict(torch.load(save_path))
    model.eval()
    
    test_corrects = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            test_corrects += torch.sum(preds == labels.data)
            
    test_acc = test_corrects.double() / test_size
    print(f"Độ chính xác (Accuracy) trên tập Test: {test_acc:.4f}")
    print(f"Hoàn thành! Trọng số mô hình đã được lưu tại: {os.path.abspath(save_path)}")

if __name__ == '__main__':
    main()
