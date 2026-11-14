"""Pure feedback/score rules recovered from official game code.

These accept explicit game/media state; they do not emulate Windows media,
animation timing, assets or input devices. Static-code parity is not live UI parity.
"""
import math
from dataclasses import dataclass
import numpy as np

SOUNDS = {1: 'Alpha.wav', 4: 'Smr.wav', 2: 'Beta.wav', 5: 'SmrBeta.wav',
          6: 'Theta.wav', 7: 'Theta.wav', 8: 'Theta.wav'}


def reward_sound(mode, reward):
    """One new processing result: sound on every success, not only rising edges."""
    if mode not in SOUNDS:
        raise ValueError('Unsupported mode')
    return SOUNDS[mode] if reward else None


def arrow_rms(spectrum_a, spectrum_b):
    a, b = np.asarray(spectrum_a), np.asarray(spectrum_b)
    if a.ndim != 1 or b.ndim != 1 or min(len(a), len(b)) < 20 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Expected finite spectra with at least 20 bins')
    return float(np.sqrt((np.sum(a[13:20]**2) + np.sum(b[13:20]**2))/14))


def arrow_reference(config_second_byte=0):
    if type(config_second_byte) is not int or not 0 <= config_second_byte <= 11:
        raise ValueError('Only recovered configuration indices 0..11 are supported')
    return (15, 13, 11, 9, 7, 5, 4, 3, 2, 1)[config_second_byte] if config_second_byte < 10 else 0


@dataclass
class ArrowState:
    energy: int = 0
    total: int = 0

    def update(self, reward, blocked=False):
        if not blocked:
            self.energy = min(100, max(0, self.energy + (5 if reward else -1)))
        return self.energy

    def shoot(self, rms, reference):
        """Call at settlement using latest RMS/E and actual configuration B."""
        if not math.isfinite(rms) or rms < 0 or not math.isfinite(reference) or reference < 0:
            raise ValueError('Expected finite nonnegative RMS and reference')
        score = -1
        if self.energy >= 2:
            for k in range(2, 14):
                if rms < k * reference / 11:
                    score = 12 - k
                    break
        self.total = max(0, self.total + score)
        return score


def race_action(reward):
    return 'play' if reward else 'pause'


def meditation_action(reward, prompt_blocked=False):
    return 'play' if reward and not prompt_blocked else 'pause'


def puzzle_tick(position, reward, lower, upper):
    """One animation tick, not one EEG second; task phases remain application-owned."""
    if lower > upper:
        raise ValueError('Invalid animation bounds')
    return min(upper, max(lower, position + (2 if reward else -1)))


def memory_result(length, all_correct, total=0):
    if type(length) is not int or not 2 <= length <= 9:
        raise ValueError('Sequence length must be 2..9')
    return dict(total=total + (length*3 if all_correct else 0),
                next_length=length + int(bool(all_correct)),
                completed=bool(all_correct and length == 9))


def frc_down_feedback(reward, media_ended=False):
    return dict(action='play' if reward else 'pause',
                score_delta=(10 if media_ended else 2) if reward else 0,
                restart=bool(reward and media_ended))


@dataclass
class CupState:
    """CupR 00403170 retained-state reward consumer, excluding media implementation.

    The caller supplies actual media end state when checked. Actions describe
    existing media callbacks; lifecycle is reset by constructing a new state.
    """
    reverse: bool = False
    end_enabled: bool = False
    score: int = 0
    failure_counter: int = 0
    retained: int = 0
    end_counter: int = 0
    was_end: bool = False

    def process(self, reward, media_ended=False, position=0, length=0):
        actions = []
        if reward:
            actions.append('play' if not self.reverse else 'reverse_seek')
            if self.end_enabled and media_ended:
                if self.was_end and self.end_counter != 2:
                    self.end_counter += 1
                    self.reverse, self.retained = False, 1
                    return dict(actions=actions, score=self.score)
                self.was_end = True
                self.score += 8
                self.end_counter = 0
            self.reverse, self.retained = False, 1
        else:
            if self.was_end:
                if self.failure_counter == 1:
                    self.failure_counter = self.end_counter = 0
                    self.was_end = False
                else:
                    self.failure_counter += 1
            if self.retained < 1:
                if self.reverse:
                    actions.append('play')
                else:
                    self.end_enabled = True
                    actions.append('reverse_seek')
                self.retained = 0
                if not self.end_enabled and length // 2 < position:
                    self.end_enabled = True
                self.reverse = True
            self.retained -= 1
        return dict(actions=actions, score=self.score)
