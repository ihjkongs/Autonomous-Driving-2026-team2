"""Resolve numbered experiments with descriptive names or legacy numeric names."""
from pathlib import Path
import re


def experiment_path(results_root, name, index, label=None):
    parent = Path(results_root) / name
    number = str(int(index))
    candidates = ([parent / number] if (parent / number).is_dir() else [])
    candidates += sorted(p for p in parent.glob(number + '_*') if p.is_dir())
    if len(candidates) > 1:
        raise ValueError('Multiple directories for experiment %s/%s: %s' % (name, number, candidates))
    if candidates:
        return candidates[0]
    if label:
        if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_-]*', label):
            raise ValueError('Experiment label must use letters, numbers, underscores or hyphens')
        return parent / (number + '_' + label)
    return parent / number
