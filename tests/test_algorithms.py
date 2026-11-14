import json
from pathlib import Path
import numpy as np
import pytest
from synthetic import signals, write_txt, evaluate_yyg
from neurocore._feedback import Threshold, compute_srq, raw_gate, reward_statistics
from neurocore.assessment import bqtester, bqrequest, brainsee
from neurocore.neurofeedback import measurer, monitorr, games


def assert_nested(actual, expected):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for k in expected:
            assert_nested(actual[k], expected[k])
    elif isinstance(expected, list):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected):
            assert_nested(a, b)
    elif isinstance(expected, (float, int)) and not isinstance(expected, bool):
        assert actual == pytest.approx(expected, rel=1e-9, abs=1e-10)
    else:
        assert actual == expected


@pytest.mark.parametrize('age', [6, 30, 65, 12])
def test_yyg_golden(age, tmp_path):
    from neurocore.analysis import yyg
    expected = json.loads((Path(__file__).parent/'golden'/('yyg-age%d.json' % age)).read_text(encoding='utf8'))
    path = tmp_path / 'synthetic.txt'
    write_txt(path, signals(expected['seed'], stress=expected['stress']))
    assert_nested(evaluate_yyg(yyg, path, age), expected['values'])


def test_window_scales_and_no_extra_gain():
    t = np.arange(256)/256
    raw = 511 + 30*np.sin(2*np.pi*10*t)
    # Integer validation reflects device codes; an exact integer tone uses period 4.
    raw = 511 + np.resize([0, 30, 0, -30], 256)
    assert brainsee.spectrum(raw)[64] == pytest.approx(23.4375)
    assert bqtester.spectrum(raw)[64] == pytest.approx(23.4375)
    assert measurer.spectrum(raw)[64] == pytest.approx(11.71875)
    assert bqrequest.spectrum(raw)[64] == pytest.approx(12.65625, abs=1e-8)
    assert brainsee.spectrum(raw)[0] == 0


def test_threshold_phase_and_strict_comparison():
    low, high = Threshold(), Threshold(True)
    assert [low.update(v) for v in [3, 6, 9, 12, 15, 18, 21]] == [0, 0, 0, 4.5, 4.5, 4.5, 13.5]
    assert [high.update(v) for v in [3, 6, 9, 12]] == [0, 0, 0, 7.5]
    e = bqtester.AssessmentEngine()
    out = e.process_raw(np.full(256, 511), np.full(256, 511), 4)
    assert out['reward'] is False


@pytest.mark.parametrize('bad,passed', [(0, True), (127, True), (128, False), (223, False), (224, True), (512, True)])
def test_gate_nonmonotonic_boundary(bad, passed):
    slots = np.full(512, 512)
    slots[:bad] = 0
    assert raw_gate(slots[:256], slots[256:]) == dict(bad=bad, passed=passed)


def test_raw_adapter_tail_and_unknown_gate():
    raw = np.arange(256) + 400
    fft, buffer = monitorr.adapt_channel(raw)
    np.testing.assert_array_equal(fft, (raw[::2]+raw[1::2])//8)
    np.testing.assert_array_equal(buffer[128:], raw[128:])
    with pytest.raises(ValueError):
        raw_gate(fft, fft)
    e = monitorr.MonitorEngine()
    # Gate-less TXT cannot make a confirmed success.
    for pair in signals(seconds=8):
        a, _ = monitorr.adapt_channel(pair[0]); b, _ = monitorr.adapt_channel(pair[1])
        result = e.process(a, b, 4)
        assert result['reward'] in (False, None)


@pytest.mark.parametrize('engine_type', [monitorr.MonitorEngine, monitorr.MeditationEngine])
def test_history_roundtrip_and_profile_rejection(engine_type):
    raw = signals(seconds=10)
    left, right = engine_type(), engine_type()
    for a, b in raw[:5]:
        left.process_raw(a, b, 1)
    state = left.snapshot()
    right.restore(json.loads(json.dumps(state)))
    for a, b in raw[5:]:
        assert_nested(left.process_raw(a, b, 4), right.process_raw(a, b, 4))
    before = right.snapshot()
    state['profile'] = 'wrong'
    with pytest.raises(ValueError):
        right.restore(state)
    assert right.snapshot() == before
    right.reset()
    assert right.frames == 0


def test_statistics_and_cold_start():
    assert reward_statistics([True, True, True, False, True]) == dict(percentage=80, weight=2, successes=4, cells=5)
    with pytest.raises(ValueError):
        reward_statistics([None])
    assert compute_srq([20, 20, 20], [0, 0, 0]) == 60
    raw = signals(seconds=60)
    first = bqtester.evaluate_training(raw, 4)
    assert first['rewards'] == bqtester.evaluate_training(raw, 4)['rewards']
    assessment = bqtester.evaluate_assessment([raw, raw, raw])
    for row, mode in zip(assessment['stages'], (1, 4, 2)):
        assert row['rewards'] == bqtester.evaluate_training(raw, mode)['rewards']


def test_report_protocol_and_fields():
    raw = signals(seconds=90).reshape(3, 30, 2, 256)
    r = bqrequest.compute_report(raw, 30, [20, 30, 40], [2, 3, 4])
    assert r['spectra'].shape == (3, 30, 2, 128)
    assert r['mean_bands'].shape == (3, 2, 6)
    assert r['BQ'] == pytest.approx((sum(r['BRQ'])+r['SRQ']+sum(r['ATQ'])+sum(r['ACQ'])+r['EQ']+sum(r['AntiStress'])+r['CQ'])/11)
    assert r['CQ'] == pytest.approx(r['Symmetry']+25*(r['Synchrony']+1))
    with pytest.raises(ValueError):
        bqrequest.compute_report(raw, 3, [20]*3, [0]*3)
    with pytest.raises(ValueError):
        bqrequest.compute_report(raw[:, :29], 30, [20]*3, [0]*3)


def test_games_boundaries_and_separate_scores():
    arrow = games.ArrowState(energy=2, total=10)
    assert arrow.shoot(2., 11.) == 9  # exact boundary advances to next band
    assert arrow.shoot(12., 11.) == -1
    arrow.energy = 1
    assert arrow.shoot(0., 11.) == -1
    assert games.arrow_reference(0) == 15
    assert games.frc_down_feedback(False, True)['score_delta'] == 0
    assert games.frc_down_feedback(True, True)['score_delta'] == 10
    assert games.meditation_action(True, True) == 'pause'
    assert games.memory_result(9, True)['completed']
    cup = games.CupState(end_enabled=True)
    assert [cup.process(True, True)['score'] for _ in range(4)] == [8, 8, 8, 16]


def test_monitor_against_original_js_golden():
    expected = json.loads((Path(__file__).parent/'golden/monitor-js.json').read_text(encoding='utf8'))
    engine = monitorr.MonitorEngine()
    for i, ((a, b), target) in enumerate(zip(signals(expected['seed'], 183), expected['results'])):
        result = engine.process_raw(a, b, 1 if i < 61 else 4 if i < 122 else 2)
        for key in ('values', 'thresholds', 'flags', 'candidate', 'gate', 'reward'):
            assert_nested(result[key], target[key])
        state = engine.snapshot()
        assert {k: row['counter'] for k, row in state['thresholds'].items()} == target['counters']


def test_yyg_patterns_and_advice_context():
    from neurocore.analysis import yyg
    from neurocore.analysis.yyg.context import get_advice_library
    assert '集中力高于放松力' in yyg.compute_dsr_patterns(20, 22, 25)
    assert '集中力高于放松力' not in yyg.compute_dsr_patterns(20, 22, 35, for_advice=True)
    assert get_advice_library() is None
    with pytest.warns(DeprecationWarning):
        assert yyg._compute_dsr_patterns(20,22,25) == yyg.compute_dsr_patterns(20,22,25)
    host = object()
    with yyg.advice_library(host):
        assert get_advice_library() is host
    assert get_advice_library() is None


@pytest.mark.parametrize('age,expected', [(4,(5,6)),(6,(6,5)),(7,(7,4)),(9,(8,4)),(14,(9,3)),
                                        (45,(10,2)),(59,(9,3)),(69,(8,4)),(79,(7,5)),(99,(6,6))])
def test_age_boundaries(age, expected):
    assert bqrequest.age_parameters(age) == expected


def test_protocol_crc_and_channel_order():
    from neurocore.data.protocol import crc16, decode_frame
    raw = np.column_stack([np.arange(256), np.arange(256)+400]).astype('<u2')
    body = raw.tobytes()+b'\x0d'
    checksum = crc16(body).to_bytes(2, 'big')
    frame = b'\x71\x00\x01'+body+checksum+b'\x00\x00'
    a, b = decode_frame(frame)
    np.testing.assert_array_equal(a, raw[:, 0])
    np.testing.assert_array_equal(b, raw[:, 1])
    corrupt = bytearray(frame); corrupt[50] ^= 1
    with pytest.raises(ValueError):
        decode_frame(corrupt)
    with pytest.raises(ValueError):
        decode_frame(frame[:-1])


def test_optional_plot_and_yyg_legacy_html(tmp_path):
    pytest.importorskip('plotly')
    from neurocore.plotting import surface_figure, write_html
    from neurocore.analysis import yyg
    from neurocore.analysis.yyg.visualization import build_3d_fft_chart_single
    raw = signals(seconds=2)
    g = measurer.spectrum_grid(raw[:, 0])
    original = g.amplitude.copy()
    fig = surface_figure(g)
    assert fig.layout.meta['profile'] == measurer.PROFILE
    assert np.array(fig.data[0].z).shape == (2, 121)
    np.testing.assert_array_equal(g.amplitude, original)
    path = tmp_path/'view.html'
    write_html(g, path)
    assert 'cdn.plot.ly' not in path.read_text(encoding='utf8').split('<script src=')[0][:1000]
    txt = tmp_path/'synthetic.txt'; write_txt(txt, signals())
    a, b, seconds = yyg.load_and_process(str(txt))
    html = build_3d_fft_chart_single(a, seconds, str(txt))
    assert 'plotly' in html.lower()


def test_three_dimensional_first_frame_and_legacy_dual(tmp_path):
    pytest.importorskip('plotly')
    from neurocore.plotting import surface_figure
    from neurocore.analysis import yyg
    from neurocore.analysis.yyg.visualization import build_3d_fft_chart
    data = signals(seconds=1)
    grid = brainsee.spectrum_grid(data[:,0])
    assert np.array(surface_figure(grid).data[0].z).shape == (2,121)
    assert grid.amplitude.shape == (1,128)  # No invented sample in underlying grid
    txt = tmp_path/'synthetic.txt'; write_txt(txt, signals())
    a,b,n = yyg.load_and_process(str(txt))
    assert 'plotly' in build_3d_fft_chart(a,b,n,str(txt)).lower()


def test_meditation_modes_and_transactional_snapshot():
    engine = monitorr.MeditationEngine()
    data = signals(seconds=12)
    for i,(a,b) in enumerate(data):
        result = engine.process_raw(a,b,(6,7,8)[i%3])
        assert result['mode'] in (6,7,8)
        assert result['reward'] is None or type(result['reward']) is bool
    snapshot = engine.snapshot()
    invalid = json.loads(json.dumps(snapshot))
    invalid['meditation']['slots'][0] = float('nan')
    with pytest.raises(ValueError):
        engine.restore(invalid)
    assert engine.snapshot() == snapshot
