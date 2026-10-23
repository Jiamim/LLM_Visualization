"""Verify logical parameter counts against a local copy of the pinned official inference code.

Usage: python scripts/verify-deepseek-reference.py --reference-dir /path/to/inference
Requires torch, numpy, sympy, Pillow (dependencies of the official reference).
No weights are downloaded; kernels are not executed. All weights use the meta device.
"""
import argparse
import json
import sys
import types
from pathlib import Path

import torch

parser = argparse.ArgumentParser()
parser.add_argument('--reference-dir', type=Path, required=True)
parser.add_argument('--output', type=Path, default=Path('docs/deepseek/reference-counts.json'))
options = parser.parse_args()
sys.path.insert(0, str(options.reference_dir.resolve()))


def forbidden_kernel(*args, **kwargs):
    raise RuntimeError('No kernel execution during meta-device parameter verification')


# The CUDA module cannot import on CPU-only hosts. We do not exercise its behavior.
kernel = types.ModuleType('kernel')
for name in ['act_quant', 'fp4_act_quant', 'fp4_gemm', 'fp8_gemm', 'hc_split_sinkhorn', 'sparse_attn']:
    setattr(kernel, name, forbidden_kernel)
sys.modules['kernel'] = kernel
import model  # noqa: E402; official inference/model.py

args = model.ModelArgs(**json.loads((options.reference_dir / 'config.json').read_text()))
args.dtype = 'bf16'
args.expert_dtype = None
args.max_seq_len = 8
args.max_batch_size = 1
# Tokenizer-dependent hash buffers contain no learnable weights.
model.NgramHashState = lambda *args, **kwargs: None
with torch.device('meta'):
    network = model.Transformer(args)


def count(module):
    # Engram embeddings retain their FP8 scale even when dense weights use BF16.
    return sum(p.numel() for name, p in module.named_parameters() if not name.endswith('.scale'))


counts = {
    'all': count(network),
    'engram': sum(count(layer.engram) for layer in network.layers if layer.engram is not None),
    'vision': count(network.vision) + count(network.aligner) + 3 * args.dim,
    'backbone': count(network.embed) + count(network.norm) + count(network.head)
    + sum(p.numel() for layer in network.layers for name, p in layer.named_parameters() if not name.startswith('engram.')),
    'layers': [{'attention': count(layer.attn), 'moe': count(layer.ffn)} for layer in network.layers],
}
counts['draft'] = counts['all'] - counts['engram'] - counts['vision'] - counts['backbone']
options.output.write_text(json.dumps(counts, indent=2) + '\n')
print(json.dumps({key: value for key, value in counts.items() if key != 'layers'}, indent=2))
