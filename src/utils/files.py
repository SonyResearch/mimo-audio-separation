import os
import typing as tp
import csv

import numpy as np


def fast_scandir(dir: str, ext: tp.List[str]):
    """ Very fast `glob` alternative. from https://stackoverflow.com/a/59803793/4259243

    fast_scandir implementation by Scott Hawley originally in https://github.com/zqevans/audio-diffusion/blob/main/dataset/dataset.py

    Args:
        dir (str): top-level directory at which to begin scanning.
        ext (tp.List[str]): list of allowed file extensions.
    """
    subfolders, files = [], []
    # add starting period to extensions if needed
    ext = ['.' + x if x[0] != '.' else x for x in ext]

    try:  # hope to avoid 'permission denied' by this try
        for f in os.scandir(dir):
            try:  # 'hope to avoid too many levels of symbolic links' error
                if f.is_dir():
                    subfolders.append(f.path)
                elif f.is_file():
                    is_hidden = os.path.basename(f.path).startswith(".")
                    has_ext = os.path.splitext(f.name)[1].lower() in ext

                    if has_ext and (not is_hidden):
                        files.append(f.path)
            except Exception:
                pass
    except Exception:
        pass

    for dir in list(subfolders):
        sf, f = fast_scandir(dir, ext)
        subfolders.extend(sf)
        files.extend(f)

    return subfolders, files


def get_info_from_csv(
    csv_path: str, main_tag: str = 'filepath',
    other_info_tags: tp.List[str] = ['sample_rate', 'num_frames', 'num_channels']
):
    main_data = []
    meta_dicts = []
    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            main_data.append(row[main_tag])
            meta = {k: int(row[k]) for k in other_info_tags}
            meta_dicts.append(meta)

    # sort by file path
    sorted_indices = np.argsort(main_data)
    main_data = [main_data[i] for i in sorted_indices]
    meta_dicts = [meta_dicts[i] for i in sorted_indices]

    return main_data, meta_dicts
