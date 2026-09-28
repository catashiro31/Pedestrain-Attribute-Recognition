# Kế hoạch Kiểm thử & Chẩn đoán CLIMP-PAR v4 (_my_model_9)

Kế hoạch này tập trung vào hai khía cạnh chính để kiểm tra độ mạnh mẽ và tính minh bạch của mô hình: **Trực quan hóa Không gian Nhúng (Embedding Space)** và **Bản đồ Độ nổi bật (Saliency Maps)**.

---

## 1. Trực quan hóa Không gian Nhúng (t-SNE / UMAP)

### Mục tiêu
Đánh giá mức độ hiệu quả của module **Disentanglement** (khử rối). Kiểm tra xem các vector đặc trưng của thuộc tính có thực sự tách biệt khỏi bối cảnh (domain/style) hay không.

### Cách thức thực hiện

**Bước 1: Trích xuất các Vector Đặc trưng**
Bạn cần viết một script (`test_embeddings.py`) chạy qua một tập con của tập `train` (Source Domain) và `test` (Target Domain), truyền cờ `return_features=True` vào hàm `forward` của mô hình.

```python
# Trong vòng lặp inference:
with torch.no_grad():
    logits, features = model(images, bg_images, return_features=True)
    disentangled, style_emb, bg_context = features
    
    # Tính lại Attr_Embs (trước khi trừ nhiễu)
    attr_embs = disentangled + style_emb.unsqueeze(1)
```

**Bước 2: Giảm chiều với t-SNE / UMAP và Vẽ biểu đồ**
Thu thập `attr_embs`, `style_emb` và `disentangled` cho toàn bộ ảnh, kèm theo nhãn miền (Source/Target). Dùng thư viện `umap-learn` hoặc `sklearn.manifold.TSNE` để giảm chiều xuống 2D và vẽ bằng `matplotlib`.

```python
import umap
import matplotlib.pyplot as plt

reducer = umap.UMAP(n_components=2, random_state=42)
# Chạy reducer.fit_transform() cho disentangled embeddings
# Vẽ scatter plot, tô màu theo "Miền" (Source/Target) hoặc theo "Thuộc tính"
```

### Hướng dẫn Đọc bệnh (Diagnosis)

| Hiện tượng trên Biểu đồ | Chẩn đoán |
| :--- | :--- |
| Các điểm dữ liệu của cùng một thuộc tính (VD: Backpack) bị chia làm 2 cụm hoàn toàn tách biệt theo Source và Target. | **Domain Shift:** Mô hình chưa học được đặc trưng tổng quát, bị phụ thuộc vào phân bố dữ liệu của từng tập. |
| Các `Attr_Embs` (trước disentangle) gom cụm theo bối cảnh (ngày/đêm, trong nhà/ngoài trời) thay vì gom theo thuộc tính. | **Entanglement (Vướng víu bối cảnh):** Vector thuộc tính bị "nhiễm" thông tin bối cảnh. |
| Sau khi trừ nhiễu, các `Disentangled Embs` của một thuộc tính hội tụ thành 1 cụm duy nhất, bất kể Source hay Target. | **Mô hình Tốt:** Module khử rối hoạt động hiệu quả, phân tách rõ ràng ngữ nghĩa thuộc tính. |

---

## 2. Phân tích Bản đồ Độ nổi bật (Saliency Maps)

### Mục tiêu
Kiểm tra xem mạng lưới đang "nhìn" vào đâu trên bức ảnh khi đưa ra quyết định dự đoán thuộc tính. Do VMamba dùng cấu trúc SSM (không có attention map truyền thống như ViT), ta sẽ dùng **Grad-CAM** để theo dõi luồng đạo hàm (gradient).

### Cách thức thực hiện

Sử dụng thư viện `pytorch-grad-cam`. Do `VMamba-Tiny` là kiến trúc hierarchical, ta sẽ trích xuất feature map ở layer cuối của backbone (trước khi bị adaptive pool).

**Bước 1: Áp dụng Grad-CAM vào lớp mục tiêu**
Layer lý tưởng để gắn hook là block cuối cùng của `vision_encoder.backbone`.

```python
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# Target lớp cuối cùng của VMamba backbone
target_layers = [model.vision_encoder.backbone.layers[-1]]

# Khởi tạo Grad-CAM
cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)

# Lọc thuộc tính mục tiêu (ví dụ: dự đoán Backpack - class id = 9)
targets = [ClassifierOutputTarget(9)]

# Tính toán CAM (grayscale)
grayscale_cam = cam(input_tensor=img_batch, targets=targets)[0, :]
```

**Bước 2: Chồng Heatmap lên ảnh gốc**
```python
# Chuyển ảnh gốc về dải [0, 1] dạng RGB float32
rgb_img = np.float32(raw_img) / 255
cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

plt.imshow(cam_image)
plt.title("Grad-CAM cho thuộc tính 'Backpack'")
plt.show()
```

*(Ghi chú: Đối với cấu trúc đa nhánh như của bạn, bạn có thể cần custom một class wrapper bọc model lại để tương thích hoàn toàn với thư viện Grad-CAM, nhận input ảnh và trả ra logits trực tiếp)*

### Hướng dẫn Đọc bệnh (Diagnosis)

| Vị trí Heatmap (Màu đỏ/cam) | Chẩn đoán |
| :--- | :--- |
| Tập trung vào bãi cỏ, bầu trời, nền đất gạch thay vì người. | **Thiên kiến bối cảnh (Spurious Correlation):** Mô hình đoán bừa dựa vào hoàn cảnh xung quanh (VD đoán "mặc quần đùi" vì ảnh có "bãi cỏ xanh"). |
| Focus vào đúng vị trí cơ thể (đoán mũ nhìn vào đầu, đoán giày nhìn vào chân). | **Học đúng Bản chất:** Mô hình có tính minh bạch và độ tin cậy cao. |
| Toàn bộ bức ảnh đều có độ nổi bật như nhau, không có điểm nhấn. | **Thiếu năng lực trích xuất:** Model chưa phân biệt được vùng quan trọng. |

---

## 3. Các bước tiếp theo để triển khai (Action Items)

1. **Tạo file script `test_embeddings.py`**: Chạy batch qua toàn bộ dataset, lưu mảng NumPy của embeddings và vẽ t-SNE plot.
2. **Tạo file script `test_gradcam.py`**: Xây dựng wrapper model cho Grad-CAM và xuất ra một grid ảnh so sánh Ground Truth / Dự đoán / Heatmap.
3. **Thực thi và đối chiếu:** Đánh giá các đồ thị được sinh ra so với phần chẩn đoán ở trên để xem _my_model_9 có gặp lỗi thiết kế hay không.
