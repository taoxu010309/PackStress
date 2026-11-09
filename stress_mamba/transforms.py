"""Consistent training-derived stress scale selection for all entry points."""
import json
import math
from pathlib import Path


def percentile_key(percentile):
    value = float(percentile)
    if not math.isfinite(value) or not 0 <= value <= 100:
        raise ValueError('Percentile must be finite and between 0 and 100')
    return 'stress_p' + str(value).removesuffix('.0')


def stress_scale(stats, percentile=70):
    key = percentile_key(percentile)
    if key in stats:
        value = float(stats[key])
    elif float(stats.get('stress_scale_percentile', -1)) == float(percentile):
        value = float(stats['stress_scale'])
    else:
        raise ValueError(f'Missing {key}; recompute training statistics for the selected percentile')
    # Preserve the original non-positive-scale fallback; label error handling is unchanged.
    return value if not value <= 0 else 1.0


def load_stress_stats(data_dir, percentiles=(70,)):
    with (Path(data_dir) / 'transform_stats.json').open() as f:
        stats = json.load(f)
    return {float(p): stress_scale(stats, p) for p in percentiles}, stats.get('stress_scale_percentile')


def transform_settings(config, scales):
    return {'asinh_percentile': float(config.ASINH_PERCENTILE),
            'asinh_scale': scales[float(config.ASINH_PERCENTILE)]}


def validate_checkpoint_transform(checkpoint, settings):
    recorded = checkpoint.get('transform_settings')
    if recorded is not None and recorded != settings:
        raise ValueError('Checkpoint and evaluation asinh percentile/scale differ')
