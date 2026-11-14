"""Run with an isolated interpreter after wheel installation, outside src imports."""
import builtins
import json
from pathlib import Path
import numpy as np

original_import = builtins.__import__


def without_plotly(name, *args, **kwargs):
    if name == 'plotly' or name.startswith('plotly.'):
        raise ImportError('Optional Plotly deliberately unavailable in core smoke test')
    return original_import(name, *args, **kwargs)


builtins.__import__ = without_plotly
import neurocore
from neurocore.analysis import yyg
from neurocore.assessment import bqrequest, bqtester, brainsee
from neurocore.neurofeedback import measurer, monitorr, games
from neurocore.data.protocol import decode_frame
from neurocore.plotting import surface_figure

assert 'site-packages' in str(Path(neurocore.__file__).resolve())
assert neurocore.__version__ == '0.1.0'
assert yyg.compute_dsr_patterns(20,20,20) == []
assert measurer.spectrum(np.full(256,511)).shape == (128,)
grid = brainsee.spectrum_grid(np.full((1,256),511))
try:
    surface_figure(grid)
except ImportError as exc:
    assert 'neurocore-eeg[viz]' in str(exc)
else:
    raise AssertionError('Optional dependency guard failed')
print(json.dumps(dict(version=neurocore.__version__, installed_from_wheel=True,
                      imported_from=str(Path(neurocore.__file__).resolve()), core_without_plotly=True)))
