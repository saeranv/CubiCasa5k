"""Generates an abbreviated floorplan for BEM extraction in DeepRad."""
import os
import sys
import json
from pathlib import Path

import numpy as np
from floortrans.loaders.svg_loader import FloorplanSVG

# Constants
ROOM_CLASSES = ["Background", "Outdoor", "Wall", "Kitchen", "Living Room",
                "Bed Room", "Bath", "Entry", "Railing", "Storage", "Garage", "Undefined"]
# Path to svg data directory of cubicasa5k
CUBICASA5K_DATA_DIR = os.path.abspath(
    os.path.join(os.getcwd(), 'data', 'cubicasa5k'))

# Path to all models in deep_rad
DEEPRAD_DATA_DIR = os.path.abspath(os.path.join(
    os.getcwd(), '..', 'deeprad/data/models/'))


def abbreviated_floorplan_dict(floorplan_dict):
    """Creates a lightweight dictionary representation of floor image and vector data.

    Args:
        floorplan_dict: Raw floorplan dictionary from FloorplanSVG class.
            .. code-block:: python
                {
                'image': img,
                'label': label,
                'folder': self.folders[index],
                'heatmaps': heatmaps,
                'scale': coef_width,
                'house': house
                }

    Returns:
        Abbreviated dictionary of floorplan data for BEM generation.
            .. code-block:: python
                {
                'scale': Float to scale unscaled image to normalized scaled image for CNN input.
                'img': Original floorplan image.
                'label': Labeled floorplan image.
                'wall_vecs': Wall vector data.
                'window_vecs': Window vector data.
                'door_vecs': Door vector data.
                }
        """

    hdata = floorplan_dict

    # Convert tensors to lists
    img = hdata['image'].permute(1, 2, 0)
    img = img.data.cpu().tolist()
    label = hdata['label'].data.cpu().tolist()[0]

    # hdata['scale']: img_orig / img_scaled, so multiply by scale goes back to orig
    abbrev_floorplan_dict = {
        'id': 'floorplan_{}'.format(hdata['folder'].split('/')[2]),
        'scale': hdata['scale'],
        'img': img,
        'label': label,
        # get vector representations for scaling
        'walls': hdata['house'].representation['walls'],
        'windowss': hdata['house'].representation['windows'],
        'doors': hdata['house'].representation['doors']
    }

    return abbrev_floorplan_dict


def write_json(abbrev_floorplan_dict, dest_fpath):
    """Writes hdict to json."""
    with open(dest_fpath, 'w') as fp:
        json.dump(abbrev_floorplan_dict, fp, indent=4)


def make_dir_safely(dest_dir):
    Path(dest_dir).mkdir(parents=True, exist_ok=True)


def main(data_num, dest_dir, verbose=True):
    """Generate data_num amount of abbreviated floorplan_bem json at dest_dir."""

    if dest_dir is None:
        dest_dir = DEEPRAD_DATA_DIR

    test_fpath = os.path.join(CUBICASA5K_DATA_DIR, 'test.txt')

    with open(test_fpath, 'r') as fp:
        ids = fp.readlines()

    n_ids = len(ids)
    print('Total data available: {}\n'.format(n_ids))
    data_num = min(data_num, n_ids)

    # Load floorplan_svgs
    hdatas = FloorplanSVG(CUBICASA5K_DATA_DIR, '/test.txt',
                          format='txt', original_size=True)

    for i in range(data_num):
        hdata = hdatas.get_data(i)
        hdict = abbreviated_floorplan_dict(hdata)
        id = hdict['id']

        # Make new filepaths
        dest_id_dir = os.path.join(dest_dir, id)
        dest_id_fpath = os.path.join(dest_id_dir, 'abbrev_{}.json'.format(id))
        make_dir_safely(dest_id_dir)

        # write json and report
        write_json(hdict, dest_id_fpath)
        if verbose:
            print('{}/{}: wrote {} at {}'.format(i +
                  1, data_num, id, dest_id_fpath))

# Dump polygons as numpy arrays
# for i, poly_np in enumerate(poly_np_arr):
#     deeprad_fpath = os.path.join(
#         deeprad_model_dir, floorplan_id, '{}_{}.npy'.format(floorplan_id, i))
#     np.save(deeprad_fpath, poly_np)


if __name__ == "__main__":

    # Set defaults
    data_num, dest_dir, verbose = 1, None, True

    if len(sys.argv) > 1:
        argv = sys.argv[1:]
        if '--data_num' in argv:
            data_num = int(argv[1])
        if '--dest_dir' in argv:
            dest_dir = argv[3]
        if '--silent' in argv:
            verbose = False

    main(data_num, dest_dir, verbose)
