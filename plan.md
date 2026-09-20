Hiện mô hình 2 (_my_model_2) giống y hệt mô hình 1 (_my_model_1)
Nhưng giờ hãy cải tiến nó:
+ Bổ sung thêm một câu lệnh prompt nữa để đưa vào LLM (xây dựng ma trận 57x57 thể hiện mối quan hệ giữa các thuộc tính từ thực tế)
+ Ma trận mối quan hệ sau đó kết hợp với vector embeddings (sinh ra từ LLM khi cho các câu input độc lập) đi vào một mô hình GCN cơ bản. Đầu ra của GCN vẫn là text embeddings kích thước như hiện tại và được xử lý tương tự như hiện có. 