"""One-time, explicit extraction of the existing YYG algorithms (no runtime dependency)."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(r'Z:\Work\Trae Output\6ad2f046786dea544f7a4102\YYG_report_tool')
TARGET = ROOT / 'src/neurocore/analysis/yyg'


def main():
    TARGET.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for file in (SOURCE / 'eeg_report').glob('*.py'):
        if file.name == '__init__.py':
            continue
        data = file.read_bytes()
        dest = TARGET / 'eeg_report' / file.name
        dest.parent.mkdir(exist_ok=True)
        dest.write_bytes(data)
        manifest['eeg_report/' + file.name] = hashlib.sha256(data).hexdigest()
    (TARGET / 'eeg_report/__init__.py').write_text('"""Preserved YYG signal algorithms."""\n', encoding='utf8')
    file = SOURCE / 'feature_analysis.py'
    source = file.read_text(encoding='utf-8-sig')
    tree = ast.parse(source)
    selected = {
        'get_age_adjusted_tension_threshold', 'get_age_adjusted_emotion_threshold',
        'parse_birthdate', 'parse_scan_date', 'compute_age_from_birthdate',
        'load_and_process', 'load_and_process_dual', '_compute_dsr_patterns',
        '_compute_eq_patterns', '_compute_paf_patterns', '_compute_paf_patterns_from_channels',
        '_get_eq_interpretations', '_get_paf_interpretations', '_get_dsr_interpretations',
    }
    constants = {'DSR_PATTERN_ORDER', 'EQ_PATTERN_ORDER', 'PAF_PATTERN_ORDER'}
    pieces = ['"""Extracted YYG algorithms; numeric behavior preserved. See provenance.json.\n'
              'Historical units/interpretations in source are not physical calibration.\n"""\n']
    pieces.append('from .context import get_advice_library as _get_advice_library')
    for node in tree.body:
        if isinstance(node, ast.Import):
            if all(a.name != 'argparse' for a in node.names):
                pieces.append(ast.get_source_segment(source, node))
        elif isinstance(node, ast.ImportFrom):
            segment = ast.get_source_segment(source, node)
            if node.module and node.module.startswith('eeg_report'):
                segment = segment.replace('from eeg_report', 'from .eeg_report')
                pieces.append(segment)
            elif node.module != 'dynamic_txt':
                pieces.append(segment)
        elif isinstance(node, (ast.ClassDef, ast.FunctionDef)):
            if node.lineno < 1440 or node.name in selected:
                decorators = ''.join('@' + ast.get_source_segment(source, d) + '\n' for d in node.decorator_list)
                pieces.append(decorators + ast.get_source_segment(source, node))
        elif isinstance(node, ast.Assign):
            names = {n.id for t in node.targets for n in ast.walk(t) if isinstance(n, ast.Name)}
            if (node.lineno < 1440 and 'project_root' not in names and 'WECHAT_ID' not in names) or names & constants:
                pieces.append(ast.get_source_segment(source, node))
    (TARGET / '_features.py').write_text('\n\n'.join(pieces) + '\n', encoding='utf8')
    # Original Plotly chart functions remain byte-for-byte source fragments.
    plots = ['from ._features import ChannelSpectrum, get_scaling_array\nfrom .eeg_report.config import MAX_DISPLAY_FREQ\nimport numpy as np\n']
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in {'_build_3d_fft_chart', '_build_3d_fft_chart_single'}:
            plots.append(ast.get_source_segment(source, node))
    plots.append('build_3d_fft_chart = _build_3d_fft_chart\nbuild_3d_fft_chart_single = _build_3d_fft_chart_single\n')
    (TARGET / 'visualization.py').write_text('\n\n'.join(plots), encoding='utf8')
    manifest['feature_analysis.py'] = hashlib.sha256(file.read_bytes()).hexdigest()
    dynamic = SOURCE / 'dynamic_txt.py'
    (TARGET / 'dynamic_txt.py').write_bytes(dynamic.read_bytes())
    manifest['dynamic_txt.py'] = hashlib.sha256(dynamic.read_bytes()).hexdigest()
    public = [node.name for node in ast.parse((TARGET / '_features.py').read_text(encoding='utf8')).body
              if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and not node.name.startswith('_')]
    aliases = {'compute_dsr_patterns': '_compute_dsr_patterns', 'compute_eq_patterns': '_compute_eq_patterns',
               'compute_paf_patterns': '_compute_paf_patterns', 'compute_paf_patterns_from_channels': '_compute_paf_patterns_from_channels',
               'tension_closed_rise_detail': '_tension_closed_rise_detail', 'tension_closed_rise_level': '_tension_closed_rise_level'}
    init = '"""YYG compatibility API; defaults and result containers preserved."""\nfrom ._features import (\n'
    init += ''.join('    ' + n + ',\n' for n in public + sorted(constants) + ['THETA_DROP_ABS_THRESHOLD']) + ')\n'
    init += ''.join('from ._features import ' + old + ' as ' + new + '\n' for new, old in aliases.items())
    init += 'from .context import advice_library\n'
    init += '''
def __getattr__(name):
    legacy = {'_compute_dsr_patterns': 'compute_dsr_patterns',
              '_tension_closed_rise_detail': 'tension_closed_rise_detail',
              '_tension_closed_rise_level': 'tension_closed_rise_level'}
    if name in legacy:
        import warnings
        warnings.warn(name + ' is deprecated; use ' + legacy[name], DeprecationWarning, stacklevel=2)
        return globals()[legacy[name]]
    raise AttributeError(name)
'''
    (TARGET / '__init__.py').write_text(init, encoding='utf8')
    previous = ROOT / 'docs/provenance.json'
    if previous.exists() and json.loads(previous.read_text(encoding='utf8'))['sha256'] != manifest:
        raise ValueError('Source hashes changed: audit and explicitly version the baseline before regenerating')
    monitor_ref = ROOT / 'docs/monitor-reference.json'
    provenance = json.dumps({
        'yyg_declared_baseline': 'da014597054a01f95dc820210a663d85737aeb65',
        'comparison': 'git diff da01459 -- feature_analysis.py dynamic_txt.py eeg_report: no differences (2026-11-14)',
        'extraction': 'local source content identified by SHA-256; no source repository modified',
        'sha256': manifest,
        'monitor_reference': json.loads(monitor_ref.read_text(encoding='utf8')) if monitor_ref.exists() else None,
    }, indent=2) + '\n'
    (ROOT / 'docs/provenance.json').write_text(provenance, encoding='utf8')
    (ROOT / 'src/neurocore/provenance.json').write_text(provenance, encoding='utf8')
    print('Extracted', len(public), 'public YYG functions/classes; source hashes saved.')


if __name__ == '__main__':
    main()
