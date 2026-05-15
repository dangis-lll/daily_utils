from __future__ import annotations

import argparse
import sys
from pathlib import Path

import SimpleITK as sitk


def iter_candidate_dirs(root_dir: Path) -> list[Path]:
    """Return directories that may contain a DICOM series."""
    return [root_dir, *sorted(path for path in root_dir.rglob("*") if path.is_dir())]


def collect_series(root_dir: Path) -> dict[str, list[str]]:
    """Recursively collect DICOM series under one first-level folder."""
    series_files: dict[str, list[str]] = {}

    for current_dir in iter_candidate_dirs(root_dir):
        try:
            series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(current_dir))
        except RuntimeError as exc:
            print(f"[WARN] Failed to inspect {current_dir}: {exc}")
            continue

        if not series_ids:
            continue

        for series_id in series_ids:
            try:
                file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(
                    str(current_dir),
                    series_id,
                )
            except RuntimeError as exc:
                print(f"[WARN] Failed to read series in {current_dir}: {exc}")
                continue

            if file_names:
                series_files[f"{current_dir}__{series_id}"] = list(file_names)

    return series_files


def choose_primary_series(series_files: dict[str, list[str]]) -> tuple[str, list[str]]:
    """Pick the series with the most slices."""
    return max(series_files.items(), key=lambda item: len(item[1]))


def convert_one_first_level_folder(folder: Path, output_dir: Path, overwrite: bool = False) -> bool:
    """Convert the largest DICOM series under one first-level folder."""
    output_path = output_dir / f"{folder.name}.nii.gz"

    if output_path.exists() and not overwrite:
        print(f"[SKIP] {folder.name}: output exists -> {output_path.name}")
        return False

    series_files = collect_series(folder)
    if not series_files:
        print(f"[SKIP] {folder.name}: no DICOM series found")
        return False

    selected_key, selected_files = choose_primary_series(series_files)
    if len(series_files) > 1:
        series_sizes = sorted(
            ((key, len(files)) for key, files in series_files.items()),
            key=lambda item: item[1],
            reverse=True,
        )
        summary = ", ".join(f"{Path(key.split('__', 1)[0]).name}:{count}" for key, count in series_sizes)
        print(
            f"[INFO] {folder.name}: found {len(series_files)} series, "
            f"using the largest one ({len(selected_files)} slices). {summary}"
        )

    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(selected_files)
    image = reader.Execute()
    sitk.WriteImage(image, str(output_path), useCompression=True)

    print(
        f"[OK] {folder.name}: {len(selected_files)} slices -> {output_path.name} "
        f"(source: {Path(selected_key.split('__', 1)[0])})"
    )
    return True


def convert_dicom_series_to_nifti(root_path: str | Path, overwrite: bool = False) -> tuple[int, int]:
    """
    Convert DICOM series under every first-level subfolder of root_path.

    For each first-level folder, all nested DICOM series are found recursively.
    Only the series with the most slices is kept and written as
    <first_level_folder_name>.nii.gz directly under root_path.

    Returns:
        A tuple of (converted_count, skipped_count).
    """
    root = Path(root_path).expanduser()

    if not root.is_dir():
        raise NotADirectoryError(f"Root directory does not exist: {root}")

    first_level_dirs = sorted(path for path in root.iterdir() if path.is_dir())
    if not first_level_dirs:
        raise ValueError(f"No first-level subfolders found under: {root}")

    converted = 0
    skipped = 0

    for folder in first_level_dirs:
        try:
            if convert_one_first_level_folder(folder, output_dir=root, overwrite=overwrite):
                converted += 1
            else:
                skipped += 1
        except Exception as exc:  # noqa: BLE001
            skipped += 1
            print(f"[ERROR] {folder.name}: {exc}")

    return converted, skipped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recursively find DICOM series in each first-level subfolder and save "
            "the largest series as <subfolder_name>.nii.gz under the given root directory."
        )
    )
    parser.add_argument(
        "root",
        nargs="?",
        default=".",
        help="Root directory whose first-level subfolders will be processed. Default: current directory.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing <subfolder_name>.nii.gz files.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root).expanduser()

    try:
        converted, skipped = convert_dicom_series_to_nifti(root, overwrite=args.overwrite)
    except (NotADirectoryError, ValueError) as exc:
        print(f"[ERROR] {exc}")
        return 1

    print(f"\nDone. Converted: {converted}, Skipped: {skipped}, Total: {converted + skipped}")
    return 0


if __name__ == "__main__":
    convert_dicom_series_to_nifti(r"C:\baidunetdiskdownload\60testCBCT", overwrite=False)
