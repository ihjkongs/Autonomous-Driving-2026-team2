"""Fixed procedural closed tracks reserved for evaluation, never training."""
import argparse
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
from ruamel.yaml import YAML
from utils.expert_config import ROOT

CASES = [
    ('OfflineNovelThreeLobes', 18., .24, 3, .08, 2.5),
    ('OfflineNovelFourLobes', 22., .19, 4, .06, 3.0),
    ('OfflineNovelFiveLobes', 20., .22, 5, .04, 2.2),
]


def create():
    records = []
    for name, radius, amplitude, frequency, asymmetry, width in CASES:
        directory = ROOT/'installation/maps'/name
        directory.mkdir(parents=True, exist_ok=False)
        theta = np.linspace(0., 2*np.pi, 1200, endpoint=False)
        r = radius*(1.+amplitude*np.cos(frequency*theta)+asymmetry*np.sin(2*theta))
        points = np.column_stack([r*np.cos(theta), r*np.sin(theta)])
        resolution, margin = .05, 4.
        origin = points.min(axis=0)-margin
        size = np.ceil((points.max(axis=0)-origin+margin)/resolution).astype(int)+1
        image = np.zeros((size[1], size[0]), dtype=np.uint8)
        pixels = np.rint((points-origin)/resolution).astype(np.int32)
        pixels[:, 1] = size[1]-1-pixels[:, 1]
        cv2.polylines(image, [pixels.reshape(-1, 1, 2)], True, 255, round(width/resolution), cv2.LINE_8)
        image_path = directory/(name+'_map.png')
        cv2.imwrite(str(image_path), image)
        with (directory/(name+'_map.yaml')).open('w') as stream:
            YAML().dump(dict(image=image_path.name, resolution=resolution,
                origin=[float(origin[0]), float(origin[1]), 0.], negate=0,
                occupied_thresh=.65, free_thresh=.196, width=width), stream)
        np.savetxt(directory/(name+'_centerline.csv'), points, delimiter=',', header='x,y')
        records.append(dict(name=name, radius=radius, amplitude=amplitude, frequency=frequency,
            asymmetry=asymmetry, road_width_m=width, resolution_m=resolution,
            files={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.iterdir()},
            role='Evaluation only: never added to training, normalization or checkpoint selection'))
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    cli = parser.parse_args()
    cli.manifest.parent.mkdir(parents=True, exist_ok=True)
    cli.manifest.write_text(json.dumps(dict(tracks=create()), indent=2)+'\n')
    print(cli.manifest)


if __name__ == '__main__':
    main()
