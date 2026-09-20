Xây dựng pipline huấn luyện như sau:
Có 3 phần chính:
    nhánh text có 3 loại prompt:
        - style: domain tokens + các từ ẩn (ko có các class/atribute)
        - attribute: domain tokens + các từ ẩn + chỉ 1 class/atribute. (Do vậy đối chiếu với tập dataset sẽ có 57 prompts dạng này)
        - full: domain tokens + các từ ẩn + tất cả các class/atribute.
        tất cả prompts đều đi qua text encoder (VMamba-130M được đóng băng)
            - embeddings của style sẽ dùng để trừ đi cho các attribute embeddings (nhằm loại bỏ các phụ thuộc vào domain). 57 embeddings được tạo ra từ đây sẽ được concat lại. Sau đó qua An upsampling convolution block để đưa về kích thước ảnh đầu vào (gọi nó là (z)) 
            - full embeddings sẽ được dùng làm kết quả của text branch.
        
    nhánh vision:
        - Ảnh gốc concat với (z) sau đó với thực hiện PadToSize lên 448x448 và qua the vision projector. Sau đó chuyển đến bước Learned Latent Visual Space. Sau đó qua VMamba-Small được đóng băng tạo thành vision embeddings là kết quả của vision branch
    
    nhánh cross:
        - Sử dụng cross-mamba giống như trong _my_model_4 để tổng hợp 2 nhánh và tính đầu ra.
    
Mở tất cả các thành phần cho huấn luyện chỉ đóng băng (vision và text encoder từ mamba)