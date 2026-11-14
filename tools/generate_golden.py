"""Generate YYG golden JSON from the original, unmodified reference module."""
import contextlib
import importlib.util
import io
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tests'))
from synthetic import signals, write_txt, evaluate_yyg
SOURCE = Path(r'Z:\Work\Trae Output\6ad2f046786dea544f7a4102\YYG_report_tool')
sys.dont_write_bytecode = True
sys.path.insert(0, str(SOURCE))
spec = importlib.util.spec_from_file_location('yyg_reference', SOURCE / 'feature_analysis.py')
reference = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = reference
spec.loader.exec_module(reference)
# Business copy is intentionally supplied by host tools, not bundled with core.
# Freeze the original algorithm's documented no-library fallback for these fixtures.
reference._get_advice_library = lambda: None
out = ROOT / 'tests/golden'
out.mkdir(exist_ok=True)
for age, seed, stress in ((6, 20261114, False), (30, 20261115, False), (65, 20261116, False), (12, 20261117, True)):
    with tempfile.TemporaryDirectory(dir=ROOT / 'local') as folder:
        path = Path(folder) / 'synthetic.txt'
        write_txt(path, signals(seed, stress=stress))
        with contextlib.redirect_stdout(io.StringIO()):
            values = evaluate_yyg(reference, path, age)
    # File paths are not part of the numeric baseline (source has no such fields).
    (out / ('yyg-age%d.json' % age)).write_text(json.dumps(dict(age=age, seed=seed, stress=stress, values=values),
                                                            ensure_ascii=False, allow_nan=False), encoding='utf8')
    print('Golden saved for age', age)
