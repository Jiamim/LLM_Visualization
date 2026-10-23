"""Verify Gemma E2B/E4B parameter counts and small-model shapes against Transformers.

Usage: python scripts/verify-gemma-reference.py [--shapes]
No model weights are downloaded. Full configurations instantiate on the meta device;
--shapes runs tiny, randomly initialized text, vision and audio modules on CPU.
Validated with Transformers 5.12.1. Optional media IO backends are excluded during this
process because no raw image/audio decoding is needed (inputs are tensors).
"""
import argparse
import importlib.util
import json
from pathlib import Path

original_find_spec = importlib.util.find_spec

def tensor_only(name, *args, **kwargs):
    if name.split('.')[0] in {'torchvision', 'torchaudio', 'librosa'}:
        return None
    return original_find_spec(name, *args, **kwargs)

importlib.util.find_spec = tensor_only
import torch
import transformers
from transformers import (
    Gemma4Config, Gemma4ForConditionalGeneration, Gemma4TextConfig, Gemma4ForCausalLM,
    Gemma4VisionConfig, Gemma4VisionModel, Gemma4AudioConfig, Gemma4AudioModel,
)

parser = argparse.ArgumentParser()
parser.add_argument('--shapes', action='store_true')
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
count = lambda module: sum(p.numel() for p in module.parameters())
for name in ['e2b', 'e4b']:
    config = Gemma4Config.from_dict(json.loads((root / f'src/data/gemma4-{name}.json').read_text()))
    with torch.device('meta'):
        model = Gemma4ForConditionalGeneration(config)
    text = model.model.language_model
    actual = {
        'transformers': transformers.__version__,
        'total': count(model), 'text': count(text), 'embedding': count(text.embed_tokens),
        'ple_table': count(text.embed_tokens_per_layer),
        'ple_projection': count(text.per_layer_model_projection) + count(text.per_layer_projection_norm),
        'vision': count(model.model.vision_tower) + count(model.model.embed_vision),
        'audio': count(model.model.audio_tower) + count(model.model.embed_audio),
        'layers': [{'attention': count(layer.self_attn), 'mlp': count(layer.mlp), 'all': count(layer)} for layer in text.layers],
    }
    expected = json.loads((root / f'docs/gemma/reference-{name}.json').read_text())
    for key in expected:
        if key != 'transformers':
            assert actual[key] == expected[key], (name, key, actual[key], expected[key])
    print(name, 'parameter counts matched:', actual['total'], flush=True)

if args.shapes:
    torch.manual_seed(7)
    torch.set_num_threads(2)
    config = Gemma4TextConfig(
        vocab_size=64, vocab_size_per_layer_input=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=6, num_attention_heads=2, num_key_value_heads=1,
        head_dim=8, global_head_dim=16, hidden_size_per_layer_input=8,
        num_kv_shared_layers=2, use_double_wide_mlp=True, sliding_window=4,
        layer_types=['sliding_attention', 'full_attention'] * 3,
        final_logit_softcapping=30.0,
    )
    config._attn_implementation = 'eager'
    model = Gemma4ForCausalLM(config).eval()
    ids = torch.randint(2, 64, (2, 7))
    with torch.inference_mode():
        ple = model.model.get_per_layer_inputs(ids, None)
        assert list(ple.shape) == [2, 7, 6, 8]
        output = model(ids, use_cache=True, return_shared_kv_states=True)
        assert list(output.logits.shape) == [2, 7, 64]
        assert len(output.past_key_values.layers) == 4
        for i, cache in enumerate(output.past_key_values.layers):
            assert list(cache.keys.shape) == [2, 1, 3 if i % 2 == 0 else 7, 8 if i % 2 == 0 else 16]
        assert list(output.shared_kv_states['sliding_attention'][0].shape) == [2, 1, 7, 8]
        assert not hasattr(model.model.layers[4].self_attn, 'k_proj')
        assert model.model.layers[4].mlp.intermediate_size == 128
        decoded = model(torch.randint(2, 64, (2, 2)), past_key_values=output.past_key_values,
                        use_cache=True, return_shared_kv_states=True)
        assert list(decoded.shared_kv_states['sliding_attention'][0].shape) == [2, 1, 5, 8]
        assert list(decoded.shared_kv_states['full_attention'][0].shape) == [2, 1, 9, 16]
        assert list(decoded.past_key_values.layers[2].keys.shape) == [2, 1, 3, 8]
    print('Text: PLE, typed KV sharing, double-width MLP, prefill and multi-token decode matched', flush=True)

    vision_config = Gemma4VisionConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=2, head_dim=16, patch_size=2,
        pooling_kernel_size=3, position_embedding_size=32, use_clipped_linears=False)
    vision_config._attn_implementation = 'eager'
    vision = Gemma4VisionModel(vision_config).eval()
    # Grid is 6 x 9, so 54 patches pool to 6 output soft tokens per image.
    ys, xs = torch.meshgrid(torch.arange(6), torch.arange(9), indexing='ij')
    positions = torch.stack([xs.flatten(), ys.flatten()], -1)[None].expand(2, -1, -1)
    with torch.inference_mode():
        features = vision(pixel_values=torch.rand(2, 54, 12), pixel_position_ids=positions)
        # Padding is stripped in the public vision output, collapsing the batch/token axes.
        assert list(features.last_hidden_state.shape) == [12, 32]
    print('Vision: 2 x 54 patches -> 12 flattened valid soft tokens matched', flush=True)

    audio_config = Gemma4AudioConfig(hidden_size=32, num_attention_heads=2, num_hidden_layers=2,
        output_proj_dims=24, subsampling_conv_channels=[8, 4], conv_kernel_size=5,
        attention_chunk_size=4, attention_context_left=5, attention_context_right=0,
        use_clipped_linears=False)
    audio = Gemma4AudioModel(audio_config).eval()
    with torch.inference_mode():
        result = audio(input_features=torch.randn(2, 17, 8), attention_mask=torch.ones(2, 17, dtype=torch.bool))
        assert list(result.last_hidden_state.shape) == [2, 5, 24]
        assert list(result.attention_mask.shape) == [2, 5]
    print('Audio: two stride-2 subsamplings, relative block attention, 17 -> 5 frames matched', flush=True)
