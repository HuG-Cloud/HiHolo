import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import mytools


# Modify these values directly before running this script.
IMAGE1_PATH = "../frc_calc/object.h5"
IMAGE2_PATH = "../frc_calc/object_1.h5"
DATASET1 = "phasedata"
DATASET2 = "phasedata"
BETA = 8.0
OUTPUT_PATH = "frc_plot.png"
SHOW_PLOT = True


def _read_array(path, dataset=None):
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".npy":
        data = np.load(path)
    elif suffix in {".h5", ".hdf5"}:
        with h5py.File(path, "r") as handle:
            if dataset is None:
                keys = list(handle.keys())
                if not keys:
                    raise ValueError(f"HDF5 file {path} is empty")
                dataset = keys[0]
            if dataset not in handle:
                raise ValueError(f"Dataset '{dataset}' not found in {path}")
            data = np.asarray(handle[dataset])
    else:
        data = read_tiff_to_float(path)

    data = np.asarray(data)
    if data.ndim != 2:
        raise ValueError(f"{path} must contain a 2D image, got shape {data.shape}")
    return data.astype(np.float64, copy=False)


def main():
    im1 = _read_array(IMAGE1_PATH, DATASET1)
    im2 = _read_array(IMAGE2_PATH, DATASET2)
    im1 = mytools.downsample_data(im1, 512)
    im2 = mytools.downsample_data(im2, 512)

    frc_values, freq, half_bit, full_bit = mytools.frc(im1, im2, beta=BETA)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(freq, frc_values, label="Fourier-ring-correlation", linewidth=1.8)
    ax.plot(freq, half_bit, label="1/2 bit threshold-curve", linewidth=1.4)
    ax.plot(freq, full_bit, label="1 bit threshold-curve", linewidth=1.4)
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Correlation")
    ax.set_title("Fourier Ring Correlation")
    ax.set_xlim(freq[0], freq[-1])
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=200)

    print(f"FRC points: {len(frc_values)}")
    print(f"Plot saved to: {OUTPUT_PATH}")
    if SHOW_PLOT:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
