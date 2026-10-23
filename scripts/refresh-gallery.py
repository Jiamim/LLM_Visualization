"""Refresh the explicitly listed configuration snapshots, never download model weights.

Default: check local inventory and hashes without network access.
--fetch updates original config.json/params.json sources from their pinned revision.
Derived configurations and compacted quantization maps are intentionally preserved.
"""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

parser = argparse.ArgumentParser()
parser.add_argument('--fetch', action='store_true')
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
manifest_path = root / 'src/data/gallery-manifest.json'
manifest = json.loads(manifest_path.read_text())
report = []
for entry in manifest['models']:
    path = root / 'src/data/gallery' / entry['configPath']
    raw = json.loads(path.read_text())
    derived = bool(raw.get('_atlas_provenance') or raw.get('_atlas_adapter') or entry.get('snapshotNote'))
    url = entry.get('configUrl', '')
    if args.fetch and not derived and url.endswith(('/config.json', '/params.json')):
        request = urllib.request.Request(url.replace('/blob/', '/raw/'))
        with urllib.request.urlopen(request, timeout=30) as response:
            data = response.read(32 * 1024 * 1024 + 1)
        if len(data) > 32 * 1024 * 1024:
            raise ValueError(f'{entry["id"]}: configuration exceeds 32 MiB')
        raw = json.loads(data)
        path.write_text(json.dumps(raw, ensure_ascii=False, indent=2) + '\n')
    report.append({'id': entry['id'], 'file': entry['configPath'], 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'source': url, 'derived_or_compacted': derived})
assert len(report) == manifest['count'] == len({x['id'] for x in report})
(root / 'docs/gallery/snapshot-hashes.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
print(f'Validated {len(report)} unique gallery entries; hashes written to docs/gallery/snapshot-hashes.json')
