"""Consume configuration-file flags before handing off to original CLI code."""
import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def configure(mode):
    import yaml
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--data-config', default=str(ROOT/'configs/data.yaml'))
    parser.add_argument('--model-config', default=str(ROOT/'configs/model.yaml'))
    parser.add_argument('--train-config', default=str(ROOT/'configs/train.yaml'))
    if mode == 'evaluate':
        parser.add_argument('--eval-config', default=str(ROOT/'configs/evaluate.yaml'))
    args, remaining = parser.parse_known_args()
    for section in ('data','model','train'):
        path = Path(getattr(args,section+'_config')).resolve()
        if not path.is_file(): parser.error(f'Configuration file missing: {path}')
        os.environ[f'STRESS_MAMBA_{section.upper()}_CONFIG'] = str(path)
    # Always use the shipped custom Python sources; compiled CUDA modules are installed separately.
    for name in ('mamba','causal-conv1d'):
        sys.path.insert(0,str(ROOT/'third_party'/name))
    if '--help' in remaining or '-h' in remaining:
        print('Additional release options: --data-config PATH --model-config PATH --train-config PATH')
        if mode=='evaluate': print('                            --eval-config PATH')
    if mode == 'evaluate':
        with open(args.eval_config,encoding='utf-8') as f: values=yaml.safe_load(f)
        supplied={arg.split('=')[0] for arg in remaining}
        allowed={'checkpoint','data_dir','gpu','batch_size','num_workers','output_dir','plot_scatter','sample_ratio','max_points'}
        if set(values)-allowed: parser.error(f'Unknown evaluation keys: {sorted(set(values)-allowed)}')
        for key,value in values.items():
            flag='--'+key
            if flag in supplied or value is None: continue
            if key=='plot_scatter':
                if value: remaining.append(flag)
            else: remaining.extend([flag,str(value)])
    sys.argv=[sys.argv[0],*remaining]
