# Tổng quan Kiến trúc CLIMP-PAR v4

Tài liệu này trình bày chi tiết kiến trúc mô hình **CLIMP-PAR v4** dành cho bài toán nhận diện thuộc tính người đi bộ (Pedestrian Attribute Recognition). Phiên bản v4 bổ sung **Background Extraction & Prompting** (trích xuất domain context từ background), **Hidden Words** (12 nhóm learnable prompt tokens), và **Disentanglement** (phép trừ Style Embedding) để loại bỏ ảnh hưởng domain bias lên attribute features.

---

## 1. Kiến trúc Tổng thể (High-level Architecture)

```mermaid
graph TD
    classDef input fill:#f9f6f0,stroke:#333,stroke-width:2px,color:#000,rx:5px,ry:5px;
    classDef model fill:#e1f5fe,stroke:#0288d1,stroke-width:2px,color:#000,rx:10px,ry:10px;
    classDef process fill:#fff3e0,stroke:#f57c00,stroke-width:2px,color:#000;
    classDef data fill:#e8f5e9,stroke:#388e3c,stroke-width:2px,color:#000,rx:5px,ry:5px;
    classDef module fill:#f3e5f5,stroke:#8e24aa,stroke-width:2px,color:#000,rx:10px,ry:10px;
    classDef output fill:#ffebee,stroke:#d32f2f,stroke-width:2px,color:#000,rx:5px,ry:5px;
    classDef hidden fill:#fff8e1,stroke:#fbc02d,stroke-width:2px,color:#000,stroke-dasharray: 5 5;

    %% INPUTS
    subgraph Data Input
        I([🖼️ Ảnh gốc<br/>256x128]):::input
        Attr([🏷️ 57 Attributes<br/>Class Names]):::input
        P_Person([✏️ Prompt: 'Person']):::input
    end

    %% HIDDEN WORDS
    subgraph Learnable Prompts
        HW_Bg[["Group 0:<br>Background HW"]]:::hidden
        HW_Attr[["Group 1-11:<br>Attribute HWs"]]:::hidden
    end

    %% BACKGROUND PIPELINE
    subgraph Background Pipeline
        SAM3[(SAM3)]:::model
        LaMa[(LaMa Refine)]:::model
        ResNet[(ResNet18 Frozen)]:::model
        MLP_Bg[MLP Projection]:::module
        
        Bg_Img([🖼️ Background Image]):::data
        Bg4P([🧩 background4prompt]):::data
    end

    %% PROMPT GENERATION
    subgraph Prompt Generation
        Style_P([🎨 Style Prompt<br>1 x 768]):::data
        Attr_P([📝 Attribute Prompts<br>57 x 768]):::data
    end

    %% CLIMP-PAR MODEL
    subgraph CLIMP-PAR v4
        V[(👁️ Vision Encoder<br>VMamba-Tiny)]:::model
        TE[(🧠 Text Encoder<br>Mamba-130M Frozen)]:::model
        
        Minus{{➖ Subtract<br>Attr_Embs - Style_Emb}}:::process
        TC[Text Compression<br>Linear 57→32]:::module
        CM[⚡ Cross-Modal Mamba Block]:::module
        
        Concat{{➕ Concat z_out & t_out}}:::process
        GAP[Global Average Pooling]:::process
        
        subgraph Fusion Head
            L1[Linear 1536→768]:::module
            N[LayerNorm]:::module
            G[GELU]:::process
            L2[Linear 768→57]:::module
        end
    end
    
    O([🎯 Logits Output<br>B x 57]):::output

    %% FLOW
    I --> SAM3
    P_Person --> SAM3
    SAM3 -- "Mask" --> LaMa
    I -- "Image" --> LaMa
    LaMa -- "Inpainted" --> Bg_Img
    Bg_Img --> ResNet
    ResNet --> MLP_Bg
    MLP_Bg --> Bg4P

    Bg4P --> Style_P
    HW_Bg -.-> Style_P

    Bg4P --> Attr_P
    HW_Attr -.-> Attr_P
    Attr --> Attr_P

    Style_P --> TE
    Attr_P --> TE
    
    TE -- "Style Emb" --> Minus
    TE -- "Attr Embs" --> Minus
    
    Minus -- "Disentangled Embs" --> TC
    TC -- "t (32x768)" --> CM
    
    I --> V
    V -- "z (32x768)" --> CM

    CM -- "z_out" --> Concat
    CM -- "t_out" --> Concat

    Concat --> GAP
    GAP --> L1
    L1 --> N
    N --> G
    G --> L2
    L2 --> O
```

---

## 2. Chi tiết các Thành phần Mạng (Detailed Components)

### 2.1. Background Extraction & Prompting (MỚI trong v4)

#### 2.1.1. Pipeline Offline (SAM3 → LaMa)
- **Mục đích:** Tách người đi bộ ra khỏi ảnh, chỉ giữ lại background.
- **SAM3 (Segment Anything Model 3):** Nhận prompt `"Person"` và ảnh gốc, tạo mask vùng người đi bộ.
- **Phép trừ ảnh:** Loại bỏ vùng người khỏi ảnh gốc (Image − Mask).
- **LaMa Refine:** Inpainting vùng bị xóa → ảnh background hoàn chỉnh, tự nhiên.
- **Lưu ý:** Pipeline này chạy **offline** — ảnh background được xử lý trước và lưu sẵn trong thư mục `domain_noper_images_inpainted_refined/`.

#### 2.1.2. Background Encoder (Online)
- **Module:** `model/background_encoder.py` → class `BackgroundEncoder`
- **Kiến trúc:** `ResNet18(frozen)` → `MLP(Linear(512, 768) + LayerNorm + GELU)`
- **Input:** Ảnh background inpainted `(B, 3, 256, 128)`
- **Output:** `background4prompt` — vector domain context `(B, 768)`
- **ResNet18 pretrained:** Sử dụng checkpoint `resnet18_background_best.pth` (đã train phân loại background).
- **Freeze:** ResNet18 backbone được **đóng băng**, chỉ MLP projection là learnable.

### 2.2. Hidden Words — 12 Nhóm Learnable Prompt Tokens (MỚI trong v4)
- **Module:** `model/hidden_words.py` → class `HiddenWords`
- **Cảm hứng:** CoOp (Context Optimization) — thay prompt cố định bằng token có thể học.
- **Cấu trúc:** 12 nhóm × 4 tokens × 768 dims = **36,864 learnable parameters**
  - **Group 0** (Background HW): Dùng trong Style Prompt để encode style/domain chung.
  - **Groups 1-11** (Attribute HWs): 11 nhóm ngữ nghĩa, mỗi nhóm ứng với ~5 thuộc tính.
- **Ánh xạ thuộc tính:** 57 thuộc tính được phân vào 11 Semantic Groups (nhóm ngữ nghĩa) theo logic matching (chính xác hoặc một phần), cụ thể như sau:
  
  | Nhóm | Attribute Group | Details (Từ khóa) |
  |:---:|:---|:---|
  | 1 | Gender | Female |
  | 2 | Age | Child, Adult, Elderly |
  | 3 | Body Size | Fat, Normal, Thin |
  | 4 | Viewpoint | Front, Back, Side |
  | 5 | Head | Bald, Long Hair, Black Hair, Hat, Glasses, Mask, Helmet, Scarf, Gloves |
  | 6 | Upper Body | Short Sleeves, Long Sleeves, Shirt, Jacket, Suit, Vest, Cotton Coat, Coat, Graduation Gown, Chef Uniform |
  | 7 | Lower Body | Trousers, Shorts, Jeans, Long Skirt, Short Skirt, Dress |
  | 8 | Shoes | Leather Shoes, Casual Shoes, Boots, Sandals, Other Shoes |
  | 9 | Bag | Backpack, Shoulder Bag, Hand Bag, Plastic Bag, Paper Bag, Suitcase, Others |
  | 10 | Activity | Calling, Smoking, Hands Back, Arms Crossed |
  | 11 | Posture | Walking, Running, Standing, Bicycle, Scooter, Skateboard |


### 2.3. Prompt Generator (MỚI trong v4)
- **Module:** `model/prompt_generator.py` → class `PromptGenerator`
- **Tạo 2 loại prompt:**
  1. **Style Prompt:** `[background4prompt] + [HW_group0_tokens]` → 1 sequence duy nhất.
  2. **57 Attribute Prompts:** `[background4prompt] + [HW_group_k_tokens] + [attr_name_word_embeds]` — mỗi prompt ứng với 1 thuộc tính.
- **Mã hóa:** Cả 2 loại prompt được đưa qua Text Encoder (Mamba backbone frozen, last-token pooling).
- **Output:** `style_emb (B, 768)` và `attr_embs (B, 57, 768)`.

### 2.4. Disentanglement (MỚI trong v4)
- **Phép trừ:** `disentangled = attr_embs − style_emb`
- **Mục đích:** Loại bỏ thông tin domain/background ra khỏi attribute embeddings.
- **Kết quả:** `disentangled_embeds (B, 57, 768)` — chỉ chứa thông tin thuộc tính, không bị "nhiễm" bởi background.

### 2.5. Vision Encoder (VMamba-Tiny & Adaptive Spatial Pooling) — Giữ nguyên từ v3
- **Backbone:** VMamba-Tiny (SS2D — 2D Selective Scan, O(N) complexity).
- **Adaptive Pooling:** `AdaptiveAvgPool2d(8, 4)` → 32 spatial vision tokens.
- **Output:** `z (B, 32, 768)`.

### 2.6. Text Encoder (Mamba-130M) — Nâng cấp trong v4
- **Backbone:** Mamba-130M (`state-spaces/mamba-130m-hf`), **frozen** (đóng băng hoàn toàn).
- **Phương thức mới:** `encode_from_embeddings()` — nhận pre-built embedding sequences (từ PromptGenerator) thay vì token IDs.
- **Gradient flow:** Mặc dù backbone frozen, gradients vẫn chảy ngược qua nó để cập nhật Hidden Words và Background MLP.

### 2.7. Cross-Modal Fusion & MLP Head — Giữ nguyên từ v3
- **Text Compression:** `Linear(57, 32)` — nén disentangled embeddings từ 57 → 32 tokens.
- **Cross-Modal Mamba Block:** 2 nhánh SSM độc lập với Shared Gate.
- **Fusion:** `Concat(z_out, t_out)` → `GAP` → `MLP(1536 → 768 → 57)`.
- **Output:** `logits (B, 57)`.

---

## 3. Quá trình Huấn luyện (Training Pipeline)

### 3.1. Learnable vs Frozen Parameters
| Component | Trainable? | Số params ước lượng |
|-----------|-----------|---------------------|
| VMamba-Tiny backbone | ✅ (toàn bộ hoặc một phần) | ~20M |
| Mamba-130M backbone | ❌ Frozen | ~130M |
| Mamba Projection + Norm | ✅ | ~1.2M |
| ResNet18 backbone | ❌ Frozen | ~11M |
| Background MLP | ✅ | ~0.4M |
| Hidden Words (12 groups) | ✅ | ~0.04M |
| Text Compression | ✅ | ~0.002M |
| Cross-Modal Mamba | ✅ | ~4.7M |
| MLP Fusion Head | ✅ | ~1.2M |

### 3.2. Loss Function
- **BCE with Logits** cho multi-label classification.
- Hỗ trợ **Weighted BCE** và **Hybrid Loss** (alpha×BCE + beta×Weighted_BCE).

### 3.3. Chiến lược Tối ưu
1. **Background Images (Offline):** Ảnh inpainted được xử lý sẵn, load song song với ảnh PAR.
2. **DDP + AMP (fp16) + Gradient Accumulation:** Giữ nguyên từ v3.
3. **AdamW + Cosine Annealing:** Chỉ optimize params có `requires_grad=True`.

---

## 4. Cấu trúc Files

```
_my_model_9/
├── config.py                       # Cấu hình (thêm background, HW fields)
├── train.py                        # Script huấn luyện v4
├── evaluate.py                     # Script đánh giá v4
├── train_background.py             # Train ResNet18 background classifier
├── losses.py                       # Hàm loss (BCE, Weighted BCE, Hybrid)
├── overview_architecture.md        # Tài liệu này
│
├── model/
│   ├── __init__.py                 # Export tất cả modules
│   ├── climp_par.py                # ★ CLIMPPAR v4 (model chính)
│   ├── vision_encoder.py           # VMamba-Tiny Vision Encoder
│   ├── text_encoder.py             # Mamba-130M Text Encoder (+ encode_from_embeddings)
│   ├── background_encoder.py       # ★ Background Encoder (ResNet18 + MLP)
│   ├── hidden_words.py             # ★ Hidden Words (12 nhóm learnable)
│   ├── prompt_generator.py         # ★ Prompt Generator (tổ hợp prompts)
│   └── cross_mamba.py              # Cross-Modal Mamba Block
│
└── dataset/
    ├── clip_dataset.py             # PARDataset (+ background image loading)
    └── prompt_templates.py         # Prompt templates (legacy, dùng cho fallback)
```

*★ = File mới hoặc thay đổi lớn trong v4*