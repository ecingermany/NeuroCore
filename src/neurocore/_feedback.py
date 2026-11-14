"""Shared feature definitions, not shared FFT profiles or implicit histories."""
import copy
import math
import numpy as np
from .data import blocks

KEYS = ('smr_B', 'theta_AB', 'high_frequency_AB', 'beta_A',
        'relax_spectral_support', 'relax_auxiliary', 'beta_inhibit_AB')
UPPER = frozenset(('theta_AB', 'high_frequency_AB', 'relax_auxiliary', 'beta_inhibit_AB'))


def metrics_from_spectra(a, b):
    def mean(p, lo, hi):
        return float(np.mean(p[lo:hi]))
    def spread(p):
        v = p[4:30]
        return .2 * math.sqrt(float(np.sum((v - np.sum(v) / 25) ** 2)))
    support = (mean(a, 9, 19) + mean(b, 9, 19)) / 2
    return dict(smr_B=mean(b, 12, 16), theta_AB=(mean(a, 4, 8) + mean(b, 4, 8)) / 2,
                high_frequency_AB=(mean(a, 19, 49) + mean(b, 19, 49)) / 2,
                beta_A=mean(a, 16, 19), relax_spectral_support=support,
                beta_inhibit_AB=support,
                relax_auxiliary=50 * (math.log(.94 + .03167420814479638 * spread(a)) +
                                     math.log(.94 + .03167420814479638 * spread(b))))


class Threshold:
    def __init__(self, upper=False):
        self.upper = upper
        self.counter = 0
        self.history = [0., 0., 0.]
        self.value = 0.

    def update(self, value):
        if not math.isfinite(value):
            raise ValueError('Nonfinite metric')
        if self.counter < 3:
            self.history[self.counter] = float(value)
            self.counter += 1
        else:
            extreme = max(self.history) if self.upper else min(self.history)
            self.value = (sum(self.history) / 3 + extreme) / 2
            self.history[0] = float(value)
            self.counter = 1
        return self.value


def raw_gate(a, b):
    a = blocks(a, integer=True, maximum=4095)
    b = blocks(b, integer=True, maximum=4095)
    if a.shape != (256,) or b.shape != (256,):
        raise ValueError('Gate requires complete 256-slot buffers for both channels')
    bad = int(np.count_nonzero((a < 10) | (a > 1013)) + np.count_nonzero((b < 10) | (b > 1013)))
    return dict(bad=bad, passed=bad < 128 or bad >= 224)


class FeedbackEngine:
    """Stateful engine for modes 1/4/2/5. Subclasses provide their exact FFT."""
    profile = ''
    supported_modes = (1, 4, 2, 5)

    def __init__(self):
        self.reset()

    def reset(self):
        self.thresholds = {k: Threshold(k in UPPER) for k in KEYS}
        self.frames = 0

    def snapshot(self):
        return dict(profile=self.profile, frames=self.frames, thresholds={
            k: dict(counter=t.counter, history=t.history[:], value=t.value)
            for k, t in self.thresholds.items()})

    def restore(self, state):
        state = copy.deepcopy(state)
        try:
            if state['profile'] != self.profile or type(state['frames']) is not int or state['frames'] < 0:
                raise ValueError()
            proposed = {}
            for k in KEYS:
                row = state['thresholds'][k]
                if type(row['counter']) is not int or not 0 <= row['counter'] <= 3:
                    raise ValueError()
                if len(row['history']) != 3 or not all(math.isfinite(v) for v in row['history'] + [row['value']]):
                    raise ValueError()
                t = Threshold(k in UPPER)
                t.counter, t.history, t.value = row['counter'], row['history'], row['value']
                proposed[k] = t
        except (KeyError, TypeError, ValueError, OverflowError):
            raise ValueError('Invalid or incompatible threshold snapshot') from None
        self.thresholds, self.frames = proposed, state['frames']

    def process(self, a, b, mode, gate=None, enabled=True):
        # Validate before mutating history. Unknown gates remain unknown.
        if mode not in self.supported_modes:
            raise ValueError('Unsupported mode for this profile')
        if gate is not None and type(gate) is not bool:
            raise ValueError('gate must be bool or None')
        values = metrics_from_spectra(self.spectrum(a), self.spectrum(b))
        thresholds = {k: t.update(values[k]) for k, t in self.thresholds.items()}
        flags = {k: values[k] < thresholds[k] if k in UPPER else values[k] > thresholds[k] for k in KEYS}
        required = {1: ('relax_spectral_support', 'relax_auxiliary'),
                    4: ('smr_B', 'theta_AB', 'high_frequency_AB'),
                    2: ('beta_A', 'theta_AB', 'beta_inhibit_AB'),
                    5: ('smr_B', 'beta_A', 'theta_AB', 'beta_inhibit_AB')}[mode]
        candidate = bool(enabled and all(flags[k] for k in required))
        self.frames += 1
        return dict(profile=self.profile, frame=self.frames, mode=mode, values=values,
                    thresholds=thresholds, flags=flags, candidate=candidate, gate=gate,
                    reward=False if not candidate else gate)


def reward_statistics(rewards):
    values = list(rewards)
    if not values or any(type(v) is not bool for v in values):
        raise ValueError('Statistics require a nonempty sequence of known boolean rewards')
    return dict(percentage=100 * sum(values) // len(values),
                weight=sum(a and b for a, b in zip(values, values[1:])),
                successes=sum(values), cells=len(values))


def compute_srq(percentages, weights):
    p, w = np.asarray(percentages, dtype=float), np.asarray(weights, dtype=float)
    if p.shape != (3,) or w.shape != (3,) or not np.isfinite(p).all() or not np.isfinite(w).all():
        raise ValueError('Expected three finite P and three finite W values')
    if (p < 0).any() or (p > 100).any() or (w < 0).any() or (p != np.trunc(p)).any() or (w != np.trunc(w)).any():
        raise ValueError('P must be integer 0..100 and W a nonnegative integer')
    t = p + w
    return float(max(0., t.sum() - t.std() - (t.max() - t.min())))
