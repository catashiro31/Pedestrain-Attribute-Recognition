# model/__init__.py
# CLIMP-PAR: Contrastive Language-Image Mamba cho Nhận dạng Thuộc tính Người đi bộ

from .vision_encoder import VMambaVisionEncoder
from .text_encoder import MambaTextEncoder
from .climp_par import CLIMPPAR

__all__ = ['VMambaVisionEncoder', 'MambaTextEncoder', 'CLIMPPAR']
