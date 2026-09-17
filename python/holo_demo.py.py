#!/usr/bin/env python3
"""Minimal NumPy implementation of AP holographic reconstruction.

This script is intentionally self-contained.  It does not use the hiholo
extension or CUDA.  Input holograms are intensity images stored in an HDF5
dataset with shape (H, W) or (number_of_distances, H, W).

The propagation model is fixed to the Fourier Fresnel kernel and measurements
are combined with the averaged projection used by the original AP solver.
"""

from pathlib import Path

import h5py
import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:  # The reconstruction itself does not require Matplotlib.
    plt = None


# ============================================================================
# Parameters to change for a demonstration
# ============================================================================
INPUT_H5 = "/home/hug/Downloads/Holo_logo_data.h5"
INPUT_DATASET = "data/holo"
FRESNEL_DATASET = "data/FN"

#PROBE_H5 = "/home/hug/Downloads/data/logo/holo_probe.h5"
#PROBE_DATASET = "probe_data_p0_flatcorr_org"

OUTPUT_H5 = "ap_recons.h5"

ITERATIONS = 30
DISPLAY_INTERVAL = 10
DISPLAY_REALTIME = True  # Show the phase image while iterations are running.

# Object-domain constraints.  Use +/- np.inf to disable a phase bound.
AMPLITUDE_LIMITS = (1.0, 1.0)
PHASE_LIMITS = (-np.inf, np.inf)

# ============================================================================


def list_h5_datasets(h5_file):
    """Return all dataset paths in an HDF5 file."""
    datasets = []
    h5_file.visititems(
        lambda name, item: datasets.append(name)
        if isinstance(item, h5py.Dataset)
        else None
    )
    return datasets


def read_h5_dataset(path, dataset):
    """Read a dataset from an HDF5 file, including nested paths such as data/holo."""
    with h5py.File(path, "r") as h5_file:
        if dataset not in h5_file:
            available = ", ".join(list_h5_datasets(h5_file))
            raise KeyError(f"Dataset {dataset!r} not found. Available: {available}")
        return np.asarray(h5_file[dataset], dtype=np.float64)


def rotate_images_for_reconstruction(data):
    """Rotate each input image by 180 degrees, then flip it left-right."""
    rotated = np.rot90(data, k=2, axes=(-2, -1))
    return np.flip(rotated, axis=-1)


def read_h5_image(path, dataset):
    """Read one 2-D hologram or an N-by-H-by-W stack from an HDF5 file."""
    data = read_h5_dataset(path, dataset)

    if data.ndim == 2:
        data = data[np.newaxis, ...]
    if data.ndim != 3:
        raise ValueError("The hologram dataset must have shape (H, W) or (N, H, W).")
    if not np.all(np.isfinite(data)):
        raise ValueError("The hologram dataset contains NaN or infinity.")
    if np.any(data < 0):
        raise ValueError("Hologram intensity must be non-negative.")
    return rotate_images_for_reconstruction(data)


def read_fresnel_numbers(path, dataset):
    """Read one positive Fresnel number for each measurement."""
    values = read_h5_dataset(path, dataset).reshape(-1)
    if values.size < 1:
        raise ValueError("The Fresnel-number dataset must not be empty.")
    if not np.all(np.isfinite(values)):
        raise ValueError("The Fresnel-number dataset contains NaN or infinity.")
    if np.any(values <= 0):
        raise ValueError("Each Fresnel number must be positive.")
    return values



def fourier_fresnel_kernel(shape, fresnel_number):
    """Build the Fourier propagation kernel used by CUDAPropKernel::Fourier."""
    rows, cols = shape
    if fresnel_number <= 0:
        raise ValueError("Each Fresnel number must be positive.")

    row_frequency = np.fft.fftfreq(rows)
    col_frequency = np.fft.fftfreq(cols)
    frequency_squared = row_frequency[:, np.newaxis] ** 2 + col_frequency[np.newaxis, :] ** 2
    # CUDA's genFFTFreq returns angular frequencies (2*pi*fftfreq), and its
    # Fourier kernel is exp(-i * omega_squared / (4*pi*FresnelNumber)).
    return np.exp(-1j * np.pi * frequency_squared / fresnel_number)


def propagate(field, kernel):
    return np.fft.ifft2(np.fft.fft2(field) * kernel)


def apply_object_constraints(field, amplitude_limits, phase_limits):
    """Apply the phase constraint first, then the amplitude constraint."""
    min_phase, max_phase = phase_limits
    min_amplitude, max_amplitude = amplitude_limits
    if max_phase < min_phase:
        raise ValueError("PHASE_LIMITS maximum cannot be smaller than its minimum.")
    if max_amplitude < min_amplitude:
        raise ValueError("AMPLITUDE_LIMITS maximum cannot be smaller than its minimum.")

    amplitude = np.abs(field)
    phase = np.clip(np.angle(field), min_phase, max_phase)
    amplitude = np.clip(amplitude, min_amplitude, max_amplitude)
    return amplitude * np.exp(1j * phase)


def reconstruct_ap(measured_amplitudes, kernels, iterations, display_interval,
                   amplitude_limits, phase_limits, show_realtime=False):
    """Run alternating projections with fixed averaged measurement projection."""
    if iterations < 1:
        raise ValueError("ITERATIONS must be at least 1.")
    if display_interval < 1:
        raise ValueError("DISPLAY_INTERVAL must be at least 1.")

    field = np.ones(measured_amplitudes.shape[1:], dtype=np.complex128)
    residuals = []
    inverse_kernels = np.conj(kernels)
    figure = axis = image = None

    if show_realtime and plt is None:
        print("Matplotlib is not installed; continuing without image display.")

    for iteration in range(1, iterations + 1):
        propagated = np.fft.ifft2(np.fft.fft2(field)[np.newaxis, ...] * kernels, axes=(-2, -1))
        predicted_amplitudes = np.abs(propagated)
        residual = np.linalg.norm(predicted_amplitudes - measured_amplitudes) / np.sqrt(measured_amplitudes.size)
        residuals.append(residual)

        # Keep detector-plane phase, replace detector-plane amplitude, then
        # back-propagate every measurement and average the object estimates.
        constrained = measured_amplitudes * np.exp(1j * np.angle(propagated))
        back_propagated = np.fft.ifft2(
            np.fft.fft2(constrained, axes=(-2, -1)) * inverse_kernels,
            axes=(-2, -1),
        )
        field = apply_object_constraints(
            np.mean(back_propagated, axis=0), amplitude_limits, phase_limits
        )

        if iteration % display_interval == 0 or iteration == iterations:
            phase = np.angle(field)
            print(
                f"Iteration {iteration:4d}/{iterations}: "
                f"amplitude residual = {residual:.6e}, "
                f"phase range = [{phase.min():.4f}, {phase.max():.4f}]"
            )
            if show_realtime and plt is not None:
                if image is None:
                    plt.ion()
                    figure, axis = plt.subplots(figsize=(7, 6))
                    image = axis.imshow(phase, cmap="viridis_r")
                    figure.colorbar(image, ax=axis, label="phase (rad)")
                    axis.set_xlabel("x (pixel)")
                    axis.set_ylabel("y (pixel)")
                    figure.tight_layout()
                else:
                    image.set_data(phase)
                    image.autoscale()
                axis.set_title(f"AP phase after {iteration} iterations")
                figure.canvas.draw_idle()
                # Process GUI events so the image updates during the loop.
                plt.pause(1)

    if figure is not None:
        plt.ioff()
        # Keep the final reconstruction visible without blocking file output.
        plt.show(block=False)

    return field, np.asarray(residuals, dtype=np.float32)


def main():
    input_path = Path(INPUT_H5)
    #probe_path = Path(PROBE_H5)
    output_path = Path(OUTPUT_H5)
    hologram_intensities = read_h5_image(input_path, INPUT_DATASET)
    fresnel_numbers = read_fresnel_numbers(input_path, FRESNEL_DATASET)
    #probe_intensities = read_h5_image(probe_path, PROBE_DATASET)
    #hologram_intensities = object_intensities / probe_intensities
    if len(fresnel_numbers) != hologram_intensities.shape[0]:
        raise ValueError(
            f"{FRESNEL_DATASET!r} must contain exactly one value for each hologram: "
            f"expected {hologram_intensities.shape[0]}, got {len(fresnel_numbers)}."
        )

    measured_amplitudes = np.sqrt(hologram_intensities)
    kernels = np.asarray(
        [fourier_fresnel_kernel(measured_amplitudes.shape[-2:], value) for value in fresnel_numbers]
    )

    print(f"Input: {hologram_intensities.shape}")
    print(f"Reconstruction shape: {measured_amplitudes.shape[-2:]}")
    field, _residuals = reconstruct_ap(
        measured_amplitudes, kernels, ITERATIONS, DISPLAY_INTERVAL,
        AMPLITUDE_LIMITS, PHASE_LIMITS, show_realtime=DISPLAY_REALTIME,
    )

    phase = np.angle(field)
    amplitude = np.abs(field)

    with h5py.File(output_path, "w") as h5_file:
        h5_file.create_dataset("phasedata", data=phase.astype(np.float32))
        h5_file.create_dataset("amplitude", data=amplitude.astype(np.float32))
        h5_file.attrs["algorithm"] = "AP (averaged projection, Fourier Fresnel kernel)"
        h5_file.attrs["fresnel_numbers"] = fresnel_numbers

    print(f"Saved phase and amplitude to: {output_path}")


if __name__ == "__main__":
    main()
