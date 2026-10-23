"""Verify representative gallery parameter counts with Transformers on the meta device.
No weights or remote Python modules are downloaded/executed. Requires Transformers 5.12.1.
"""
import importlib.util
import json
from pathlib import Path

original = importlib.util.find_spec
importlib.util.find_spec = lambda name, *a, **kw: None if name.split('.')[0] in {'torchvision', 'torchaudio', 'librosa'} else original(name, *a, **kw)
import torch
from transformers import AutoConfig, AutoModelForCausalLM

root = Path(__file__).resolve().parents[1]
reference = json.loads((root / 'docs/gallery/reference-counts.json').read_text())
for identity, expected in reference.items():
    raw = json.loads((root / 'src/data/gallery' / (identity + '.json')).read_text())
    model_type = raw.pop('model_type')
    raw.pop('auto_map', None)
    raw.pop('quantization_config', None)
    config = AutoConfig.for_model(model_type, **raw)
    with torch.device('meta'):
        model = AutoModelForCausalLM.from_config(config)
    actual = sum(p.numel() for p in model.parameters())
    assert actual == expected['total'], (identity, actual, expected['total'])
    print(identity, actual)
