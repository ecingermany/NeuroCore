"""YYG compatibility API; defaults and result containers preserved."""
from ._features import (
    get_scaling_array,
    ChannelSpectrum,
    FeatureResult,
    compute_physiological_tension,
    detect_theta_drop_events,
    compute_theta_emotion,
    compute_eq_feature,
    compute_paf_feature,
    get_tbr_threshold,
    compute_theta_beta_ratio,
    compute_delta_eyes_change,
    compute_theta_smr_ratio,
    compute_delta_smr_ratio,
    get_age_adjusted_tension_threshold,
    get_age_adjusted_emotion_threshold,
    parse_birthdate,
    parse_scan_date,
    compute_age_from_birthdate,
    load_and_process,
    load_and_process_dual,
    DSR_PATTERN_ORDER,
    EQ_PATTERN_ORDER,
    PAF_PATTERN_ORDER,
    THETA_DROP_ABS_THRESHOLD,
)
from ._features import _compute_dsr_patterns as compute_dsr_patterns
from ._features import _compute_eq_patterns as compute_eq_patterns
from ._features import _compute_paf_patterns as compute_paf_patterns
from ._features import _compute_paf_patterns_from_channels as compute_paf_patterns_from_channels
from ._features import _tension_closed_rise_detail as tension_closed_rise_detail
from ._features import _tension_closed_rise_level as tension_closed_rise_level
from .context import advice_library

def __getattr__(name):
    legacy = {'_compute_dsr_patterns': 'compute_dsr_patterns',
              '_tension_closed_rise_detail': 'tension_closed_rise_detail',
              '_tension_closed_rise_level': 'tension_closed_rise_level'}
    if name in legacy:
        import warnings
        warnings.warn(name + ' is deprecated; use ' + legacy[name], DeprecationWarning, stacklevel=2)
        return globals()[legacy[name]]
    raise AttributeError(name)
