"""
Re-patch dataset_ms_split_1.pkl and dataset_ms_split_2.pkl
Fix: ensure EasyDict is registered as easydict.EasyDict (not __main__.EasyDict)
so pickle serializes the correct class reference for DDP compatibility.
"""
import pickle
import sys
import types
import numpy as np

np.random.seed(0)

# Create a proper easydict module so pickle stores "easydict.EasyDict"
class EasyDict(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)
    def __setattr__(self, name, value):
        self[name] = value
    def __delattr__(self, name):
        del self[name]

# Register under 'easydict' module — this is the key fix
easydict_module = types.ModuleType('easydict')
easydict_module.EasyDict = EasyDict
sys.modules['easydict'] = easydict_module

# CRITICAL: Set __module__ so pickle stores "easydict.EasyDict" not "__main__.EasyDict"
EasyDict.__module__ = 'easydict'
EasyDict.__qualname__ = 'EasyDict'

VAL_RATIO = 0.165  # ~16.5%, same as random split

for pkl_name in ['dataset_ms_split_1.pkl', 'dataset_ms_split_2.pkl']:
    print(f"\nRe-patching {pkl_name}...")
    
    with open(pkl_name, 'rb') as f:
        dataset = pickle.load(f)
    
    # Convert all dicts to proper EasyDict (with easydict module reference)
    def convert_to_easydict(obj):
        if isinstance(obj, dict):
            new_dict = EasyDict()
            for k, v in obj.items():
                new_dict[k] = convert_to_easydict(v)
            return new_dict
        return obj
    
    dataset = convert_to_easydict(dataset)
    
    partition = dataset['partition']
    
    # Remove old train/val if they exist (from previous bad patch)
    if 'train' in partition:
        del partition['train']
    if 'val' in partition:
        del partition['val']
    if 'weight_train' in dataset:
        del dataset['weight_train']
    
    trainval_indices = partition['trainval'].copy()
    
    # Shuffle and split trainval -> train + val
    shuffled = trainval_indices.copy()
    np.random.shuffle(shuffled)
    val_size = int(len(shuffled) * VAL_RATIO)
    train_size = len(shuffled) - val_size
    
    partition['train'] = np.sort(shuffled[:train_size])
    partition['val'] = np.sort(shuffled[train_size:])
    
    dataset['weight_train'] = np.mean(
        dataset['label'][partition['train']], axis=0
    ).astype(np.float32)
    
    print(f"  trainval: {len(partition['trainval'])}")
    print(f"  train:    {len(partition['train'])}")
    print(f"  val:      {len(partition['val'])}")
    print(f"  test:     {len(partition['test'])}")
    
    # Verify pickle will store "easydict.EasyDict"
    assert type(dataset).__module__ == 'easydict', f"Wrong module: {type(dataset).__module__}"
    
    with open(pkl_name, 'wb') as f:
        pickle.dump(dataset, f)
    
    # Verify we can reload it
    with open(pkl_name, 'rb') as f:
        test_load = pickle.load(f)
    assert 'train' in test_load['partition']
    print(f"  ✅ Saved and verified {pkl_name}")
    print(f"     EasyDict module: {type(test_load).__module__}.{type(test_load).__name__}")

print("\nDone!")
