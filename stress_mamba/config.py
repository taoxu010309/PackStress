"""Portable configuration adapter; original attribute names and debug rules retained."""
import os
from datetime import datetime
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]


def read_yaml(path):
    with open(path, encoding='utf-8') as f:
        values = yaml.safe_load(f)
    if not isinstance(values, dict):
        raise ValueError(f'Expected a YAML mapping: {path}')
    return values


class Config:
    def __init__(self):
        self._values = {}
        for section in ('data', 'model', 'train'):
            path = os.environ.get(f'STRESS_MAMBA_{section.upper()}_CONFIG', ROOT/'configs'/f'{section}.yaml')
            values = read_yaml(path)
            duplicate = self._values.keys() & values.keys()
            if duplicate:
                raise ValueError(f'Duplicate config keys: {sorted(duplicate)}')
            self._values.update(values)
        if 'REAL_LOSS_WEIGHT' in self._values:
            raise ValueError('REAL_LOSS_WEIGHT was removed; use the supplied core-method train configuration')
        # YAML 1.1 may interpret 5e-5 (without a decimal point) as a string.
        for key in ('LEARNING_RATE', 'WEIGHT_DECAY', 'GRAD_CLIP_NORM',
                    'VONMISES_LOSS_WEIGHT', 'GRAD_WEIGHT',
                    'LABEL_WEIGHT_ALPHA', 'LABEL_WEIGHT_GAMMA', 'LABEL_WEIGHT_CLIP_C',
                    'ASINH_PERCENTILE'):
            self._values[key] = float(self._values[key])
        self.DEBUG_MODE = self._values['DEBUG_MODE']
        self.LOG_DIR = os.path.join(self.LOG_BASE_DIR, f"{self.EXPERIMENT_NAME}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        self.MODEL_SAVE_PATH = os.path.join(self.LOG_DIR, 'model')

    def __getattr__(self, name):
        values = self.__dict__.get('_values', {})
        if name in values:
            return values[name]
        raise AttributeError(name)

    @property
    def BATCH_SIZE(self): return 2 if self.DEBUG_MODE else self._values['BATCH_SIZE']
    @property
    def MAX_EPOCHS(self): return 5 if self.DEBUG_MODE else self._values['MAX_EPOCHS']
    @property
    def VAL_EVERY(self): return 1 if self.DEBUG_MODE else self._values['VAL_EVERY']
    @property
    def STATS_DATA_DIR(self): return self.TRAIN_DATA_DIR


config = Config()
