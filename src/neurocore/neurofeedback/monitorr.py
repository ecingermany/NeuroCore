"""MonitorR fa398d9a N128 reconstruction profile (not BQtester)."""
import numpy as np
from .._feedback import FeedbackEngine, raw_gate, reward_statistics
from ..data import blocks

PROFILE = 'monitorr-fa398d9a-n128-hann-reconstruction-v1'


def spectrum(processed):
    """Verified metric-reconstruction spectrum; low bins are not a voltage calibration."""
    a = blocks(processed, size=128, integer=True)
    w = .5 - .5 * np.cos(2 * np.pi * np.arange(128) / 128)
    out = np.abs(np.fft.fft((a - 511) * 50 / 512 * w, axis=-1))[..., :64] * .5
    out[..., 0] = 0
    return out


def adapt_channel(raw):
    a = blocks(raw, integer=True, maximum=4095)
    if a.shape != (256,):
        raise ValueError('Expected one 256-point device channel')
    buffer = a.astype(np.int32)
    buffer[:128] = ((buffer[::2] + buffer[1::2]) >> 2) // 2
    return buffer[:128].copy(), buffer


class MonitorEngine(FeedbackEngine):
    """History persists on this instance until reset(); save/restore is explicit."""
    profile = PROFILE
    spectrum = staticmethod(spectrum)

    def process_raw(self, a, b, mode, source_type=0x82000000, enabled=True):
        if source_type not in (0x81000000, 0x82000000):
            raise ValueError('Unverified source type')
        a, gate_a = adapt_channel(a)
        b, gate_b = adapt_channel(b)
        return self.process(a, b, mode, gate=raw_gate(gate_a, gate_b)['passed'], enabled=enabled)


class MeditationEngine(MonitorEngine):
    """Seven-mode static-code extension, including unusual shared helper slots.

    Modes 6/7/8 are static reconstructions, not independently live-validated UI
    sessions. The regular MonitorEngine remains the verified four-mode API.
    """
    profile = 'monitorr-fa398d9a-n128-seven-modes-static-v1'
    supported_modes = (1, 4, 2, 5, 6, 7, 8)

    def reset(self):
        super().reset()
        from .._feedback import Threshold
        self.meditation_x = Threshold(upper=True)
        self.meditation_m = Threshold(upper=True)
        self.helper_slots = [0.] * 9
        self.helper_count = 0

    def _helper(self, value, start):
        h = self.helper_slots
        if self.helper_count < 3:
            h[start+self.helper_count] = value
            self.helper_count += 1
            return 0.
        mean = (h[0] + h[start+1] + h[2]) / 3
        result = (mean + min(h[start:start+3])) / 2
        self.helper_count = 1
        h[start] = value
        return result

    def process(self, a, b, mode, gate=None, enabled=True):
        if mode not in self.supported_modes:
            raise ValueError('Unsupported mode')
        result = super().process(a, b, mode if mode in (1, 4, 2, 5) else 1, gate, enabled)
        pa, pb = self.spectrum(a), self.spectrum(b)
        t, x = result['values']['theta_AB'], result['values']['relax_spectral_support']
        m = float((pa[14:29].mean() + pb[14:29].mean()) / 2)
        alpha = float((pa[8:10].mean() + pb[8:10].mean() + pa[10:13].mean() + pb[10:13].mean()) / 2)
        tx, tm = self.meditation_x.update(x), self.meditation_m.update(m)
        helpers = {6: self._helper(t, 0), 8: self._helper(.5*t, 6), 7: self._helper(.75*t, 3)}
        if mode in (6, 7, 8):
            target = t * {6: 1., 7: .75, 8: .5}[mode]
            candidate = bool(x > tx and target > helpers[mode] and m < tm and target > alpha)
            result.update(mode=mode, candidate=candidate, reward=False if not candidate else gate)
        result['meditation'] = dict(x_threshold=tx, m_threshold=tm, helper_thresholds=helpers,
                                    helper_count=self.helper_count, helper_slots=self.helper_slots[:])
        return result

    def snapshot(self):
        state = super().snapshot()
        state['meditation'] = dict(x=vars(self.meditation_x).copy(), m=vars(self.meditation_m).copy(),
                                   slots=self.helper_slots[:], count=self.helper_count)
        import copy
        return copy.deepcopy(state)

    def restore(self, state):
        # Validate into a temporary engine so rejection cannot partially mutate us.
        import copy
        from .._feedback import Threshold
        temporary = type(self)()
        FeedbackEngine.restore(temporary, state)
        try:
            ext = state['meditation']
            if type(ext['count']) is not int or not 0 <= ext['count'] <= 3:
                raise ValueError()
            if len(ext['slots']) != 9 or not np.isfinite(ext['slots']).all():
                raise ValueError()
            for key in ('x', 'm'):
                item = ext[key]
                if type(item['counter']) is not int or not 0 <= item['counter'] <= 3 or len(item['history']) != 3 or not np.isfinite(item['history'] + [item['value']]).all():
                    raise ValueError()
                threshold = Threshold(upper=True)
                threshold.counter = item['counter']
                threshold.history, threshold.value = item['history'][:], item['value']
                setattr(temporary, 'meditation_' + key, threshold)
            temporary.helper_count, temporary.helper_slots = ext['count'], ext['slots'][:]
        except (KeyError, ValueError, TypeError):
            raise ValueError('Invalid meditation history') from None
        self.__dict__ = copy.deepcopy(temporary.__dict__)

