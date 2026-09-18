import argparse
import os
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


def read_h5_dataset(file_path, dataset_name=None):
    with h5py.File(file_path, "r") as f:
        if dataset_name is None:
            keys = list(f.keys())
            if not keys:
                raise ValueError(f"No datasets found in {file_path}")
            dataset_name = keys[0]
        if dataset_name not in f:
            raise ValueError(f"Dataset '{dataset_name}' not found in {file_path}")
        data = f[dataset_name][()].astype(np.float32)
    return dataset_name, data


def build_design_matrix(height, width, degree):
    yy, xx = np.mgrid[0:height, 0:width]
    x = np.linspace(-1.0, 1.0, width, dtype=np.float64)[None, :].repeat(height, axis=0)
    y = np.linspace(-1.0, 1.0, height, dtype=np.float64)[:, None].repeat(width, axis=1)

    terms = [np.ones_like(x)]
    if degree >= 1:
        terms.extend([x, y])
    if degree >= 2:
        terms.extend([x * x, x * y, y * y])
    if degree >= 3:
        terms.extend([x * x * x, x * x * y, x * y * y, y * y * y])

    matrix = np.stack([term.reshape(-1) for term in terms], axis=1)
    return matrix, yy, xx


def fit_polynomial_background(image, degree):
    height, width = image.shape
    design_matrix, _, _ = build_design_matrix(height, width, degree)
    coeffs, _, _, _ = np.linalg.lstsq(design_matrix, image.reshape(-1), rcond=None)
    background = (design_matrix @ coeffs).reshape(height, width).astype(np.float32)
    corrected = (image - background).astype(np.float32)
    return background, corrected, coeffs.astype(np.float32)


def save_output_h5(output_path, dataset_name, original, background, corrected, degree):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as f:
        main_ds = f.create_dataset(dataset_name, data=corrected, dtype=np.float32)
        f.create_dataset("background", data=background, dtype=np.float32)
        f.create_dataset("original", data=original, dtype=np.float32)
        main_ds.attrs["processing"] = "polynomial_background_removal"
        main_ds.attrs["polynomial_degree"] = degree


def save_corrected_png(output_path, corrected):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(output_path, corrected, cmap="gray")


def build_default_paths(input_path):
    root, _ = os.path.splitext(input_path)
    return root + "_detrended.h5", root + "_detrended.png"


def collect_h5_files(input_path, recursive=False):
    input_path = Path(input_path)
    if input_path.is_file():
        return [input_path.resolve()]

    glob_func = input_path.rglob if recursive else input_path.glob
    files = sorted(path.resolve() for path in glob_func("*.h5"))
    files.extend(sorted(path.resolve() for path in glob_func("*.H5")))

    unique_files = []
    seen = set()
    for file_path in files:
        if file_path not in seen:
            unique_files.append(file_path)
            seen.add(file_path)
    return unique_files


def resolve_output_paths(input_file, input_root, output_h5_arg, output_png_arg, batch_mode):
    default_h5, default_png = build_default_paths(str(input_file))
    if not batch_mode:
        output_h5 = os.path.abspath(output_h5_arg) if output_h5_arg is not None else default_h5
        output_png = os.path.abspath(output_png_arg) if output_png_arg is not None else default_png
        return output_h5, output_png

    rel_parent = input_file.parent.relative_to(input_root)
    detrended_name = input_file.stem + "_detrended"

    if output_h5_arg is None:
        output_h5 = default_h5
    else:
        output_h5 = str((Path(output_h5_arg).resolve() / rel_parent / (detrended_name + ".h5")))

    if output_png_arg is None:
        output_png = default_png
    else:
        output_png = str((Path(output_png_arg).resolve() / rel_parent / (detrended_name + ".png")))

    return output_h5, output_png


def process_single_file(input_h5, dataset, degree, output_h5, output_png):
    dataset_name, image = read_h5_dataset(input_h5, dataset)
    if image.ndim != 2:
        raise ValueError(f"Expected a 2D phase image, got shape {image.shape}")

    background, corrected, coeffs = fit_polynomial_background(image, degree)
    save_output_h5(output_h5, dataset_name, image, background, corrected, degree)
    save_corrected_png(output_png, corrected)

    print(f"Input file: {input_h5}")
    print(f"Dataset: {dataset_name}")
    print(f"Input shape: {image.shape}")
    print(f"Polynomial degree: {degree}")
    print(f"Background coeff count: {len(coeffs)}")
    print(
        f"Original stats: min={image.min():.6f} max={image.max():.6f} "
        f"mean={image.mean():.6f} std={image.std():.6f}"
    )
    print(
        f"Corrected stats: min={corrected.min():.6f} max={corrected.max():.6f} "
        f"mean={corrected.mean():.6f} std={corrected.std():.6f}"
    )
    print(f"Saved H5: {output_h5}")
    print(f"Saved PNG: {output_png}")

    return {
        "input_file": str(input_h5),
        "dataset_name": dataset_name,
        "output_h5": output_h5,
        "output_png": output_png,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Remove low-frequency background from a phase H5 file or a directory of H5 files."
    )
    parser.add_argument("input_h5", help="Path to the input H5 file or a directory containing H5 files")
    parser.add_argument(
        "--dataset",
        default=None,
        help="Dataset name in the input H5 file. Default: use the first dataset",
    )
    parser.add_argument(
        "--degree",
        type=int,
        default=2,
        choices=[1, 2, 3],
        help="Polynomial degree used to fit the low-frequency background",
    )
    parser.add_argument(
        "--output-h5",
        default=None,
        help="Single-file mode: output H5 path. Directory mode: output H5 directory",
    )
    parser.add_argument(
        "--output-png",
        default=None,
        help="Single-file mode: output PNG path. Directory mode: output PNG directory",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="When the input is a directory, recursively process all H5 files",
    )
    args = parser.parse_args()

    input_path = Path(args.input_h5).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    batch_mode = input_path.is_dir()
    if batch_mode and args.output_h5 is not None and Path(args.output_h5).suffix.lower() == ".h5":
        raise ValueError("Directory mode expects --output-h5 to be a directory path")
    if batch_mode and args.output_png is not None and Path(args.output_png).suffix.lower() == ".png":
        raise ValueError("Directory mode expects --output-png to be a directory path")

    input_files = collect_h5_files(input_path, recursive=args.recursive)
    if not input_files:
        raise ValueError(f"No H5 files found in {input_path}")

    if not batch_mode:
        output_h5, output_png = resolve_output_paths(
            input_files[0], input_path.parent, args.output_h5, args.output_png, batch_mode=False
        )
        process_single_file(str(input_files[0]), args.dataset, args.degree, output_h5, output_png)
        return

    print(f"Found {len(input_files)} H5 files in directory: {input_path}")
    success_count = 0
    failures = []

    for input_file in input_files:
        output_h5, output_png = resolve_output_paths(
            input_file, input_path, args.output_h5, args.output_png, batch_mode=True
        )
        print("=" * 80)
        try:
            process_single_file(str(input_file), args.dataset, args.degree, output_h5, output_png)
            success_count += 1
        except Exception as exc:
            failures.append((str(input_file), str(exc)))
            print(f"Failed to process {input_file}: {exc}")

    print("=" * 80)
    print(f"Batch processing completed: {success_count}/{len(input_files)} files succeeded")
    if failures:
        print("Failed files:")
        for file_path, error_msg in failures:
            print(f"  {file_path}: {error_msg}")


if __name__ == "__main__":
    main()
