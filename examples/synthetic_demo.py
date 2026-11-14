"""Run after wheel installation; create non-personal report/training/3D examples."""
import argparse
import json
from pathlib import Path
import numpy as np
from neurocore.assessment.bqtester import evaluate_assessment
from neurocore.assessment.bqrequest import compute_report
from neurocore.assessment.brainsee import spectrum_grid as report_grid
from neurocore.neurofeedback.measurer import spectrum_grid as measurement_grid
from neurocore.neurofeedback.monitorr import MonitorEngine


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('demo-output'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261114)
    raw = rng.integers(400, 625, size=(270, 2, 256))
    training = evaluate_assessment(raw[90:].reshape(3, 60, 2, 256))
    report = compute_report(raw[:90].reshape(3, 30, 2, 256), 30,
                            [s['percentage'] for s in training['stages']],
                            [s['weight'] for s in training['stages']])
    engine = MonitorEngine()
    for a, b in raw[:60]:
        engine.process_raw(a, b, mode=4)
    (args.output/'summary.json').write_text(json.dumps(dict(
        synthetic=True, BQ=report['BQ'], SRQ=report['SRQ'],
        monitor_frames=engine.frames, report_profile=report['profile']), indent=2), encoding='utf8')
    try:
        from neurocore.plotting import write_html
        write_html(report_grid(raw[:30, 0]), args.output/'brainsee-profile.html')
        write_html(measurement_grid(raw[:30, 0]), args.output/'measurer-profile.html')
    except ImportError:
        print('Install [viz] for optional HTML figures.')
    print('Synthetic demo saved to', args.output.resolve())


if __name__ == '__main__':
    main()
