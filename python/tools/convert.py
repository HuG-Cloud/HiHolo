import os
import h5py


TARGET_FILES = [
    "../recons_data/probe_gr.h5",
    "../recons_data/obj_gr.h5",
]
TARGET_DATASET = "phasedata"


def rename_dataset_to_phasedata(file_path: str) -> None:
    abs_path = os.path.abspath(os.path.join(os.path.dirname(__file__), file_path))
    if not os.path.exists(abs_path):
        print(f"[Skip] File not found: {abs_path}")
        return

    with h5py.File(abs_path, "r+") as f:
        if TARGET_DATASET in f:
            print(f"[OK] {os.path.basename(abs_path)} already has dataset '{TARGET_DATASET}'")
            return

        keys = list(f.keys())
        if not keys:
            print(f"[Skip] No dataset in {os.path.basename(abs_path)}")
            return

        src_key = keys[0]
        f.copy(src_key, TARGET_DATASET)
        del f[src_key]
        print(f"[Done] {os.path.basename(abs_path)}: '{src_key}' -> '{TARGET_DATASET}'")


if __name__ == "__main__":
    for rel_path in TARGET_FILES:
        rename_dataset_to_phasedata(rel_path)
