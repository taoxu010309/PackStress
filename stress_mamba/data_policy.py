"""Whole-model split validation and training-only transform provenance."""
import hashlib
import json
from pathlib import Path


def stats_path(config):
    return Path(config.TRAIN_DATA_DIR).expanduser() / 'transform_stats.json'


def models_in(directory):
    root = Path(directory).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f'Data directory does not exist: {root}')
    # The public interface takes separate roots, each containing model_* directories.
    models = sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith('model_'))
    if not models:
        raise ValueError(f'No model_* directories in {root}')
    records = []
    for model in models:
        features = model / 'input_features.zarr'
        if not features.is_dir():
            raise FileNotFoundError(f'Missing full input volume: {features}')
        meta_path = model / 'metadata.json'
        meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
        # Preserve stable source design IDs if metadata is available (even after renaming).
        model_id = str(meta.get('model_index', model.name.removeprefix('model_')))
        records.append(dict(name=model.name, design_id=model_id, path=str(model.resolve()),
                            input_path=str(features.resolve())))
    return records


def validate_splits(config, test_dir=None):
    directories = {'train':config.TRAIN_DATA_DIR, 'val':config.VAL_DATA_DIR,
                   'test':test_dir if test_dir is not None else config.DATA_DIR}
    roots = [Path(p).expanduser().resolve() for p in directories.values()]
    if len(set(roots)) != len(roots):
        raise ValueError('Train, validation and test must have separate model-level directories')
    splits = {key:models_in(value) for key,value in directories.items()}
    seen_names, seen_ids, seen_paths, seen_inputs = {}, {}, {}, {}
    for split, models in splits.items():
        for model in models:
            for field, seen in [('name',seen_names),('design_id',seen_ids),('path',seen_paths),('input_path',seen_inputs)]:
                value = model[field]
                if value in seen:
                    raise ValueError(f'Duplicate model identity ({field}={value}) in {seen[value]} and {split}; keep every design in one subset')
                seen[value] = split
    return splits


def validate_stats(config, splits):
    path = stats_path(config)
    if not path.is_file():
        raise FileNotFoundError(f'Missing training statistics: {path}; run compute_transform_stats.py --data-config YOUR_CONFIG')
    raw = path.read_bytes()
    stats = json.loads(raw)
    expected = sorted(model['design_id'] for model in splits['train'])
    provenance = stats.get('fit_provenance', {})
    if provenance.get('split') != 'train' or provenance.get('design_ids') != expected:
        raise ValueError('Statistics have missing/different training provenance. Fit statistics with this release on the configured training directory; do not derive them from validation/test data')
    return hashlib.sha256(raw).hexdigest()
