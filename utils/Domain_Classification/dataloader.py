import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class DomainDataLoader:
    def __init__(self, data_dir, batch_size=32, num_workers=4, image_size=(224, 224)):
        """
        Class khởi tạo DataLoader cho bài toán phân loại domain (6 classes).
        
        Args:
            data_dir (str): Đường dẫn đến thư mục chứa dữ liệu domain_images.
            batch_size (int): Kích thước batch.
            num_workers (int): Số lượng workers để load data.
            image_size (tuple): Kích thước ảnh đầu vào của mô hình.
        """
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        
        # Định nghĩa phép biến đổi ảnh (Data Augmentation & Preprocessing)
        self.train_transforms = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.val_transforms = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def get_loader(self, is_train=True):
        """
        Trả về DataLoader và danh sách các class.
        
        Args:
            is_train (bool): Nếu True sẽ bật xáo trộn (shuffle) và áp dụng augmentation.
        """
        transform = self.train_transforms if is_train else self.val_transforms
        
        # Tải dataset bằng ImageFolder
        dataset = datasets.ImageFolder(root=self.data_dir, transform=transform)
        class_names = dataset.classes
        
        # Tạo DataLoader
        dataloader = DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            shuffle=is_train, 
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        return dataloader, class_names

    def get_train_val_loaders(self, val_split=0.2):
        """
        Trả về train_loader và val_loader sau khi split dataset, và danh sách các class.
        
        Args:
            val_split (float): Tỉ lệ dữ liệu dành cho validation (0.0 đến 1.0).
        """
        import torch
        
        # Tạo 2 instance dataset với transform khác nhau
        train_dataset = datasets.ImageFolder(root=self.data_dir, transform=self.train_transforms)
        val_dataset = datasets.ImageFolder(root=self.data_dir, transform=self.val_transforms)
        
        class_names = train_dataset.classes
        total_size = len(train_dataset)
        val_size = int(total_size * val_split)
        train_size = total_size - val_size
        
        # Khởi tạo generator với seed cố định để đảm bảo 2 subset được split giống hệt nhau
        generator = torch.Generator().manual_seed(42)
        train_subset, _ = torch.utils.data.random_split(train_dataset, [train_size, val_size], generator=generator)
        
        # Dùng lại cùng một seed để lấy đúng phần validation split
        generator = torch.Generator().manual_seed(42)
        _, val_subset = torch.utils.data.random_split(val_dataset, [train_size, val_size], generator=generator)
        
        train_loader = DataLoader(train_subset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)
        val_loader = DataLoader(val_subset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)
        
        return train_loader, val_loader, class_names

if __name__ == "__main__":
    # Test dataloader
    data_dir = "/media/catashiro31/DATA/Nghiên cứu khoa học/Pedestrain Attribute Recognition/MSP60k/SUBMIT/domain_images"
    
    loader_manager = DomainDataLoader(data_dir, batch_size=16)
    dataloader, classes = loader_manager.get_loader(is_train=True)
    
    print(f"Danh sách các class ({len(classes)}): {classes}")
    
    for inputs, labels in dataloader:
        print(f"Kích thước inputs batch: {inputs.shape}")
        print(f"Kích thước labels batch: {labels.shape}")
        break
