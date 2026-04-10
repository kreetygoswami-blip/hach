"""
scanner.py
==========
Handles all barcode / QR-code detection.

Supports:
  - Decode from an image file path
  - Decode from a raw OpenCV frame (numpy array)
  - Live camera capture (returns first code found)

Gracefully degrades if OpenCV is not installed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── optional imports ──────────────────────────────────────────────────────────
try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logger.warning("opencv-python not installed – camera scanning disabled.")

_QR = None
_BAR = None
if CV2_AVAILABLE:
    try:
        _QR = cv2.QRCodeDetector()
    except Exception:
        _QR = None
    try:
        _BAR = cv2.barcode_BarcodeDetector()
    except Exception:
        _BAR = None

SCAN_READY = CV2_AVAILABLE


# ── result dataclass ──────────────────────────────────────────────────────────

@dataclass
class ScanResult:
    """Holds the result of a single barcode scan."""
    raw: str                    # decoded string (e.g. "3017620422003")
    symbology: str = "UNKNOWN"  # e.g. "EAN13", "QRCODE"
    success: bool = True

    @property
    def barcode(self) -> str:
        return self.raw.strip()


# ── internal helpers ──────────────────────────────────────────────────────────

def _decode_frame(frame) -> Optional[ScanResult]:
    """
    Attempt to decode barcodes from an OpenCV BGR frame.
    Returns the first ScanResult found, or None.
    """
    if not CV2_AVAILABLE or frame is None:
        return None
    try:
        # 1) QR / DataMatrix-like payloads
        try:
            if _QR is not None:
                data, _, _ = _QR.detectAndDecode(frame)
                if data:
                    return ScanResult(raw=data, symbology="QRCODE")
        except Exception:
            pass

        # 2) 1D/2D retail barcodes (EAN/UPC etc) — requires opencv-contrib-python
        try:
            if _BAR is not None:
                ok, decoded_info, _, _ = _BAR.detectAndDecode(frame)
                if ok and decoded_info:
                    for s in decoded_info:
                        if s:
                            return ScanResult(raw=s, symbology="BARCODE")
        except Exception:
            pass
    except Exception as exc:
        logger.error("OpenCV decode error: %s", exc)
    return None


def _preprocess(frame):
    """
    Apply light preprocessing to improve decode rate on blurry / low-contrast
    images.
    """
    if not CV2_AVAILABLE:
        return frame
    try:
        if len(frame.shape) == 3:
            grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            grey = frame
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        sharp = cv2.filter2D(grey, -1, kernel)
        return sharp
    except Exception:
        return frame


# ── public API ────────────────────────────────────────────────────────────────

def scan_from_file(image_path: str | Path) -> Optional[ScanResult]:
    """
    Decode the first barcode/QR found in an image file.

    Args:
        image_path: Path to a JPEG, PNG, BMP, etc.

    Returns:
        ScanResult or None if nothing found / libs missing.
    """
    if not SCAN_READY:
        logger.error("scan_from_file: OpenCV not available.")
        return None

    path = Path(image_path)
    if not path.exists():
        logger.error("Image file not found: %s", path)
        return None

    frame = cv2.imread(str(path))
    if frame is None:
        logger.error("cv2.imread failed for: %s", path)
        return None

    # Try raw first, then preprocessed
    result = _decode_frame(frame) or _decode_frame(_preprocess(frame))
    if result:
        logger.info("Scanned from file %s → %s (%s)",
                    path.name, result.raw, result.symbology)
    else:
        logger.info("No barcode found in %s", path.name)
    return result


def scan_from_frame(frame) -> Optional[ScanResult]:
    """
    Decode from a raw OpenCV BGR numpy array (e.g. from VideoCapture).

    Args:
        frame: numpy ndarray (H x W x 3, BGR)

    Returns:
        ScanResult or None.
    """
    if not SCAN_READY:
        return None
    # Try raw, preprocessed, and a center crop (barcodes are usually centered).
    try:
        res = _decode_frame(frame) or _decode_frame(_preprocess(frame))
        if res:
            return res
        h, w = frame.shape[:2]
        x1, x2 = int(w * 0.15), int(w * 0.85)
        y1, y2 = int(h * 0.20), int(h * 0.85)
        crop = frame[y1:y2, x1:x2]
        return _decode_frame(crop) or _decode_frame(_preprocess(crop))
    except Exception:
        return _decode_frame(frame) or _decode_frame(_preprocess(frame))


def scan_from_camera(camera_index: int = 0,
                     timeout_seconds: float = 15.0) -> Optional[ScanResult]:
    """
    Open the default camera and scan until a barcode is found or timeout.

    Args:
        camera_index:    OpenCV camera index (0 = default webcam).
        timeout_seconds: Give up after this many seconds.

    Returns:
        ScanResult or None.
    """
    if not SCAN_READY:
        logger.error("scan_from_camera: OpenCV not available.")
        return None

    import time
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        logger.error("Could not open camera index %d", camera_index)
        return None

    logger.info("Camera opened – scanning for up to %.1fs…", timeout_seconds)
    start = time.time()
    result = None

    try:
        while time.time() - start < timeout_seconds:
            ret, frame = cap.read()
            if not ret:
                continue
            result = scan_from_frame(frame)
            if result:
                break
    finally:
        cap.release()

    if result:
        logger.info("Camera scan → %s (%s)", result.raw, result.symbology)
    else:
        logger.info("Camera scan timed out – no barcode found.")
    return result


def decode_barcode_bytes(data: bytes) -> Optional[ScanResult]:
    """
    Decode a barcode from raw image bytes (e.g. from a file upload).

    Args:
        data: Raw image bytes (JPEG / PNG / etc.)

    Returns:
        ScanResult or None.
    """
    if not SCAN_READY:
        return None
    try:
        arr   = np.frombuffer(data, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return None
        return scan_from_frame(frame)
    except Exception as exc:
        logger.error("decode_barcode_bytes error: %s", exc)
        return None
