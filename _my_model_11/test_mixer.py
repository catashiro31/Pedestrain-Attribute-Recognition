import triton
if not hasattr(triton, 'set_allocator'):
    triton.set_allocator = lambda *args, **kwargs: None
import inspect
from mamba_ssm.models.mixer_seq_simple import MixerModel
print(inspect.signature(MixerModel.forward))
