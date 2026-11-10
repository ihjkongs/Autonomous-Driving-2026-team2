"""One source of truth for expert collection and validation."""
from pathlib import Path
from ruamel.yaml import YAML

ROOT = Path(__file__).resolve().parents[1]
MAPS = ['Sochi', 'Spa', 'Nuerburgring', 'Monza', 'Melbourne', 'Austin',
        'Silverstone', 'Sakhir', 'IMS', 'Budapest', 'Montreal', 'Sepang',
        'Oschersleben', 'YasMarina', 'MoscowRaceway', 'Zandvoort', 'Catalunya',
        'BrandsHatch', 'Shanghai', 'Hockenheim', 'SaoPaulo', 'Spielberg', 'MexicoCity']


def load_expert_config():
    with (ROOT / 'configs/expert.yaml').open() as stream:
        return YAML(typ='safe').load(stream)
