# model/__init__.py
# CLIMP-PAR v7: Contrastive Language-Image Mamba + AttributeGCN cho Nhận dạng Thuộc tính Người đi bộ

from .vision_encoder import VMambaVisionEncoder
from .text_encoder import MambaTextEncoder
from .attribute_gcn import AttributeGCN
from .climp_par import CLIMPPAR

__all__ = ['VMambaVisionEncoder', 'MambaTextEncoder', 'AttributeGCN', 'CLIMPPAR']
