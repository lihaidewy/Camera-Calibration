"""Offline checkerboard camera-intrinsics calibration for one fixed camera mode."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import cv2
    import numpy as np
except ImportError:  # Keeps --help useful when dependencies are absent.
    cv2 = None
    np = None


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
MINIMUM_VALID_IMAGES = 3
EXPECTED_RESOLUTION = (1920, 1080)


class CalibrationError(RuntimeError):
    """Raised for input or calibration failures that should not produce parameters."""


@dataclass(frozen=True)
class CheckerboardConfig:
    columns: int = 9
    rows: int = 6
    square_size_m: float = 0.025

    def validate(self) -> None:
        if self.columns < 2 or self.rows < 2:
            raise CalibrationError("Checkerboard columns and rows must both be at least 2.")
        if not math.isfinite(self.square_size_m) or self.square_size_m <= 0:
            raise CalibrationError("Checkerboard square size must be a finite value greater than zero.")


@dataclass
class ImageRecord:
    path: Path
    status: str
    reason: str = ""
    error_px: float | None = None


def require_opencv() -> None:
    if cv2 is None or np is None:
        raise CalibrationError(
            "OpenCV and NumPy are required. Install dependencies with: "
            "python -m pip install -r requirements.txt"
        )


def build_object_points(config: CheckerboardConfig) -> Any:
    """Return Z=0 checkerboard points in OpenCV's (columns, rows) order."""
    require_opencv()
    config.validate()
    points = np.zeros((config.columns * config.rows, 3), np.float32)
    points[:, :2] = np.mgrid[0 : config.columns, 0 : config.rows].T.reshape(-1, 2)
    points[:, :2] *= config.square_size_m
    return points


def discover_images(input_directory: Path) -> list[Path]:
    if not input_directory.exists():
        raise CalibrationError(f"Input directory does not exist: {input_directory}")
    if not input_directory.is_dir():
        raise CalibrationError(f"Input path is not a directory: {input_directory}")
    images = sorted(
        (path for path in input_directory.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS),
        key=lambda path: path.name.lower(),
    )
    if not images:
        allowed = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise CalibrationError(f"No supported images found in {input_directory} (supported: {allowed}).")
    return images


def detect_corners(image: Any, config: CheckerboardConfig) -> Any | None:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(
        gray,
        (config.columns, config.rows),
        cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE,
    )
    if not found:
        return None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    return cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)


def reprojection_error(
    object_points: Any,
    image_points: Any,
    rvec: Any,
    tvec: Any,
    camera_matrix: Any,
    distortion_coefficients: Any,
) -> float:
    projected, _ = cv2.projectPoints(
        object_points, rvec, tvec, camera_matrix, distortion_coefficients
    )
    # calibrateCamera may return float32 detections while projectPoints returns
    # float64 coordinates. Normalise both shape and dtype before measuring.
    residuals = image_points.reshape(-1, 2).astype(np.float64) - projected.reshape(-1, 2)
    return float(np.linalg.norm(residuals) / len(projected))


def serialise_calibration(
    *,
    config: CheckerboardConfig,
    image_size: tuple[int, int],
    rms_error: float,
    camera_matrix: Any,
    distortion_coefficients: Any,
    used_records: list[ImageRecord],
    rejected_records: list[ImageRecord],
) -> dict[str, Any]:
    return {
        "model": "pinhole",
        "image_width": image_size[0],
        "image_height": image_size[1],
        "checkerboard": {
            "inner_corners_cols": config.columns,
            "inner_corners_rows": config.rows,
            "square_size_m": config.square_size_m,
        },
        "camera_matrix": camera_matrix.tolist(),
        "fx": float(camera_matrix[0, 0]),
        "fy": float(camera_matrix[1, 1]),
        "cx": float(camera_matrix[0, 2]),
        "cy": float(camera_matrix[1, 2]),
        "distortion_coefficients": distortion_coefficients.reshape(-1).tolist(),
        "rms_error": float(rms_error),
        "mean_reprojection_error": float(np.mean([record.error_px for record in used_records])),
        "min_reprojection_error": float(min(record.error_px for record in used_records)),
        "max_reprojection_error": float(max(record.error_px for record in used_records)),
        "opencv_version": cv2.__version__,
        "number_of_candidate_images": len(used_records) + len(rejected_records),
        "number_of_used_images": len(used_records),
        "number_of_rejected_images": len(rejected_records),
        "used_images": [record.path.name for record in used_records],
        "rejected_images": [
            {"image": record.path.name, "reason": record.reason} for record in rejected_records
        ],
        "per_image_reprojection_errors": {
            record.path.name: record.error_px for record in used_records
        },
    }


def save_calibration(calibration: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(calibration, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def load_calibration(input_path: Path) -> dict[str, Any]:
    with input_path.open("r", encoding="utf-8") as handle:
        calibration = json.load(handle)
    required = {"model", "camera_matrix", "distortion_coefficients", "fx", "fy", "cx", "cy"}
    missing = required.difference(calibration)
    if missing:
        raise CalibrationError(f"Calibration file is missing required fields: {', '.join(sorted(missing))}")
    return calibration


def write_report(records: list[ImageRecord], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image", "status", "error_px", "reason"])
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "image": record.path.name,
                    "status": record.status,
                    "error_px": "" if record.error_px is None else f"{record.error_px:.8f}",
                    "reason": record.reason,
                }
            )


def calibrate(input_directory: Path, output_directory: Path, config: CheckerboardConfig) -> dict[str, Any]:
    """Calibrate one pinhole camera mode and write all reviewable output artifacts."""
    require_opencv()
    config.validate()
    images = discover_images(input_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    object_template = build_object_points(config)
    object_points: list[Any] = []
    image_points: list[Any] = []
    used_records: list[ImageRecord] = []
    rejected_records: list[ImageRecord] = []
    reference_size: tuple[int, int] | None = None

    for image_path in images:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            rejected_records.append(ImageRecord(image_path, "rejected", "image could not be read"))
            continue

        image_size = (image.shape[1], image.shape[0])
        if reference_size is None:
            reference_size = image_size
        if image_size != reference_size:
            rejected_records.append(
                ImageRecord(
                    image_path,
                    "rejected",
                    f"resolution {image_size[0]}x{image_size[1]} differs from "
                    f"calibration resolution {reference_size[0]}x{reference_size[1]}",
                )
            )
            continue

        corners = detect_corners(image, config)
        if corners is None:
            rejected_records.append(ImageRecord(image_path, "rejected", "checkerboard corners not detected"))
            continue
        object_points.append(object_template.copy())
        image_points.append(corners)
        used_records.append(ImageRecord(image_path, "used"))

    if reference_size is None:
        write_report(used_records + rejected_records, output_directory / "reprojection_errors.csv")
        raise CalibrationError("No input image could be read; calibration parameters were not created.")
    if len(used_records) < MINIMUM_VALID_IMAGES:
        write_report(used_records + rejected_records, output_directory / "reprojection_errors.csv")
        raise CalibrationError(
            f"Only {len(used_records)} valid checkerboard images found; at least "
            f"{MINIMUM_VALID_IMAGES} are required. Calibration parameters were not created."
        )

    try:
        rms_error, camera_matrix, distortion_coefficients, rvecs, tvecs = cv2.calibrateCamera(
            object_points, image_points, reference_size, None, None
        )
    except cv2.error as error:
        raise CalibrationError(f"OpenCV pinhole calibration failed: {error}") from error
    for index, record in enumerate(used_records):
        record.error_px = reprojection_error(
            object_points[index],
            image_points[index],
            rvecs[index],
            tvecs[index],
            camera_matrix,
            distortion_coefficients,
        )

    calibration = serialise_calibration(
        config=config,
        image_size=reference_size,
        rms_error=rms_error,
        camera_matrix=camera_matrix,
        distortion_coefficients=distortion_coefficients,
        used_records=used_records,
        rejected_records=rejected_records,
    )
    undistorted_directory = output_directory / "undistorted"
    undistorted_directory.mkdir(exist_ok=True)
    for record in used_records:
        image = cv2.imread(str(record.path), cv2.IMREAD_COLOR)
        if image is None:
            raise CalibrationError(f"Could not re-read valid image for undistortion: {record.path}")
        undistorted = cv2.undistort(image, camera_matrix, distortion_coefficients)
        destination = undistorted_directory / record.path.name
        if not cv2.imwrite(str(destination), undistorted):
            raise CalibrationError(f"Could not write undistorted validation image: {destination}")

    write_report(used_records + rejected_records, output_directory / "reprojection_errors.csv")
    save_calibration(calibration, output_directory / "calibration.json")
    return calibration


def parse_args(arguments: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline pinhole camera-intrinsics calibration from checkerboard images."
    )
    parser.add_argument("--input", required=True, type=Path, help="Directory containing JPG/JPEG/PNG images")
    parser.add_argument("--output", required=True, type=Path, help="Directory for calibration outputs")
    parser.add_argument("--columns", type=int, default=9, help="Checkerboard internal-corner columns (default: 9)")
    parser.add_argument("--rows", type=int, default=6, help="Checkerboard internal-corner rows (default: 6)")
    parser.add_argument(
        "--square-size-m", type=float, default=0.025, help="Checkerboard square size in metres (default: 0.025)"
    )
    parser.add_argument(
        "--model",
        choices=["pinhole"],
        default="pinhole",
        help="Camera model. Only explicit pinhole calibration is implemented in this version.",
    )
    return parser.parse_args(arguments)


def main(arguments: list[str] | None = None) -> int:
    args = parse_args(arguments)
    config = CheckerboardConfig(args.columns, args.rows, args.square_size_m)
    try:
        calibration = calibrate(args.input, args.output, config)
    except CalibrationError as error:
        print(f"Calibration failed: {error}", file=sys.stderr)
        return 2

    actual_resolution = (calibration["image_width"], calibration["image_height"])
    print(f"Found {calibration['number_of_candidate_images']} candidate images.")
    print(
        f"Used {calibration['number_of_used_images']} images; "
        f"rejected {calibration['number_of_rejected_images']} images."
    )
    print(f"Model: {calibration['model']}; resolution: {actual_resolution[0]}x{actual_resolution[1]}")
    if actual_resolution != EXPECTED_RESOLUTION:
        print(
            f"Warning: expected operating resolution is {EXPECTED_RESOLUTION[0]}x{EXPECTED_RESOLUTION[1]}, "
            f"but calibration used {actual_resolution[0]}x{actual_resolution[1]}",
            file=sys.stderr,
        )
    print(f"RMS error: {calibration['rms_error']:.6f} px")
    print(f"Mean reprojection error: {calibration['mean_reprojection_error']:.6f} px")
    print(f"Parameters: {args.output / 'calibration.json'}")
    print(f"Validation images: {args.output / 'undistorted'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
