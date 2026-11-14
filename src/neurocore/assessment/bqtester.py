"""BQtester MonitorR 1cf8c148: N256, no window, per-assessment cold start."""
import numpy as np
from .._spectral import fft256
from .._feedback import FeedbackEngine, raw_gate, reward_statistics, compute_srq
from ..data import blocks

PROFILE = 'bqtester-1cf8c148-n256-rect-v1'


def spectrum(raw):
    return fft256(raw, 'none')


class AssessmentEngine(FeedbackEngine):
    profile = PROFILE
    supported_modes = (1, 4, 2)
    spectrum = staticmethod(spectrum)

    def process_raw(self, a, b, mode, enabled=True):
        gate = raw_gate(a, b)['passed']
        return self.process(a, b, mode, gate=gate, enabled=enabled)


def evaluate_training(raw, mode):
    """Evaluate exactly 60 [A,B] raw blocks with a new engine, never inherited."""
    raw = blocks(raw, integer=True)
    if raw.shape != (60, 2, 256):
        raise ValueError('Expected exactly [60, 2, 256] raw ADC values')
    engine = AssessmentEngine()
    frames = [engine.process_raw(a, b, mode) for a, b in raw]
    rewards = [f['reward'] for f in frames]
    return dict(profile=PROFILE, mode=mode, **reward_statistics(rewards), rewards=rewards, frames=frames)


def evaluate_assessment(training_blocks):
    """Three stages in Alpha/SMR/Beta order, each independently reset."""
    if len(training_blocks) != 3:
        raise ValueError('Expected three training stages')
    results = [evaluate_training(raw, mode) for raw, mode in zip(training_blocks, (1, 4, 2))]
    return dict(profile=PROFILE, stages=results,
                srq=compute_srq([r['percentage'] for r in results], [r['weight'] for r in results]))
