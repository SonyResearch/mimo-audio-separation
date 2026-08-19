import os
import argparse
from tqdm import tqdm
import csv

import torchaudio
import soundfile as sf


def get_audio_metadata(filepath):
    info_ = sf.info(filepath)
    sample_rate = info_.samplerate
    num_channels = info_.channels
    num_frames = info_.frames

    info = {
        'sample_rate': sample_rate,
        'num_frames': num_frames,
        'num_channels': num_channels
    }

    return info


def make_metadata_csv(dir: str, repr_audio='vocals.wav'):

    # Find all 1-lev sub-directories under root dir
    dirs = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]

    print(f"Found {len(dirs)} sub-directories under {dir}")

    csv_path = os.path.join(dir, "metadata.csv")
    rows = []
    for d in tqdm(dirs):
        p = os.path.join(dir, d, repr_audio)
        id = d.replace('-', '').replace('  ', ' ').replace(' ', '_').lower()

        info = get_audio_metadata(p)
        row = {
            "id": id,
            "path": d,
            "sample_rate": info["sample_rate"],
            "num_frames": info["num_frames"],
            "num_channels": info["num_channels"]
        }
        rows.append(row)

        print(row)

    rows.sort(key=lambda x: x["id"])
    with open(csv_path, mode='w', newline='') as csv_file:
        fieldnames = ["id", "path", "sample_rate", "num_frames", "num_channels"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"Saved metadata to {csv_path}")


def main():
    args = argparse.ArgumentParser()
    args.add_argument('--root-dir', type=str, required=True, help="A root directory of audio dataset.")
    args = args.parse_args()

    root_dir = args.root_dir

    print(root_dir)
    make_metadata_csv(root_dir)


if __name__ == "__main__":
    main()
