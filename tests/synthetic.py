"""Deterministic, non-personal EEG fixtures; seed and protocol are explicit."""
import numpy as np


def signals(seed=20261114, seconds=120, stress=False):
    rng = np.random.default_rng(seed)
    t = np.arange(seconds*256) / 256
    a = 512 + 27*np.sin(2*np.pi*10*t) + 12*np.sin(2*np.pi*5*t) + rng.normal(0, 5, t.size)
    b = 512 + 23*np.sin(2*np.pi*9*t+.4) + 10*np.sin(2*np.pi*17*t) + rng.normal(0, 5, t.size)
    if stress:
        a += 130*np.sin(2*np.pi*2*t) + 210*np.sin(2*np.pi*5*t)
        b += 100*np.sin(2*np.pi*2*t) + 190*np.sin(2*np.pi*6*t)
    # Include a finite Theta drop and closed-eye modulation.
    a[(t >= 44) & (t < 47)] = 512 + rng.normal(0, 2, np.count_nonzero((t >= 44) & (t < 47)))
    return np.clip(np.rint(np.stack([a, b], axis=1)), 0, 1023).astype(int).reshape(seconds, 256, 2).transpose(0, 2, 1)


def write_txt(path, values):
    rows = ['Raw Data File', '2026-11-14 10:00:00', '2026-11-14 10:02:00',
            'Channel Number: 2', 'Sample Per Line: 256', 'Total Sample: %d' % (len(values)*256)]
    for i, pair in enumerate(values):
        for label, channel in zip('AB', pair):
            rows.extend([label+'  '+str(i*256), ','.join(map(str, channel))])
    path.write_text('\n'.join(rows)+'\n', encoding='utf8')


def normalize(value):
    import dataclasses
    if dataclasses.is_dataclass(value):
        return normalize(dataclasses.asdict(value))
    if isinstance(value, np.ndarray):
        return normalize(value.tolist())
    if isinstance(value, np.generic):
        return normalize(value.item())
    if isinstance(value, dict):
        return {k: normalize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalize(v) for v in value]
    return value


def evaluate_yyg(module, path, age):
    a, b, pa, pb, seconds = module.load_and_process_dual(str(path))
    funcs = ('compute_physiological_tension', 'compute_theta_emotion', 'compute_eq_feature',
             'compute_theta_beta_ratio', 'compute_delta_eyes_change',
             'compute_theta_smr_ratio', 'compute_delta_smr_ratio')
    result = {}
    import inspect
    for name in funcs:
        fn = getattr(module, name)
        args = {'age': age} if 'age' in inspect.signature(fn).parameters else {}
        result[name] = fn(a, b, **args)
    result['compute_paf_feature'] = module.compute_paf_feature(pa, pb, age=age, filepath=str(path))
    result['spectra'] = [a.spectrum, b.spectrum, pa.spectrum, pb.spectrum]
    result['scaled'] = [a.spectrum_scaled, b.spectrum_scaled]
    result['seconds'] = seconds
    return normalize(result)
