"""Inspect built artifacts and write a non-personal delivery manifest."""
import ast
import hashlib
import json
import tarfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
wheel = ROOT/'dist/neurocore_eeg-0.1.0-py3-none-any.whl'
sdist = ROOT/'dist/neurocore_eeg-0.1.0.tar.gz'
with zipfile.ZipFile(wheel) as archive:
    names = archive.namelist()
    assert any(n.endswith('provenance.json') for n in names)
    assert not any(n.lower().endswith(('.bfn','.ban','.exe','.dll','.wav','.txt')) and not '.dist-info/' in n for n in names)
    assert not any('/local/' in n or '/tests/' in n or '/tools/' in n or '__pycache__' in n for n in names)
    for name in names:
        if name.endswith('.py'):
            ast.parse(archive.read(name).decode('utf-8-sig'), filename=name, feature_version=(3,9))
    count = len(names)
with tarfile.open(sdist) as archive:
    assert not any('/local/' in n or '/.venv/' in n or '/.cache/' in n for n in archive.getnames())
manifest = dict(version='0.1.0', date='2026-11-14', wheel_entries=count,
                python_grammar='3.9', artifacts={})
for path in (wheel,sdist):
    manifest['artifacts'][path.name] = dict(bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
(ROOT/'dist/manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf8')
print(json.dumps(manifest, indent=2))
