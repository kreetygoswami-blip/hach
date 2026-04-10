"""
FoodScan  –  Mobile-Friendly Food Nutritional Scanner  (v4)
============================================================
All 7 feature modules + Smart Analysis Engine:

  ❶  Data Fetching       – OpenFoodFacts API: ingredients, calories,
                           full nutriments, additives_tags, allergens_tags
  ❷  Smart Analysis      – rule-based verdict engine:
                             • High sugar / trans fat  → Moderate / Avoid
                             • Harmful E-number detection (23 codes)
                             • Ingredient keyword scanning
                             • Plain-English recommendations generated
                               e.g. "High sugar content", "Contains
                               preservatives", "Good protein source"
  ❸  UI Display          – product image, Nutri-Score badge, colour-coded
                           verdict banner, recommendation cards with icons
  ❹  Allergy Alerts      – 15 allergens (gluten, nuts, dairy, soy, eggs…)
  ❺  Diet Compatibility  – Vegan, Vegetarian, Keto, Diabetic-Friendly,
                           Halal, Gluten-Free, Low-Sodium
  ❻  History & Favourites– scan history (last 30), star favourites,
                           persistent JSON storage, history screen
  ❼  Desktop QR scanner – optional OpenCV window inside the app: live
                           camera feed, switch camera index (e.g. 0 vs 1),
                           pick an image file to decode; any QR payload —
                           http(s) opens in the browser, long digit codes
                           as barcodes, other text as name search (OFF API).

Dependencies (core):
    pip install kivy requests

Optional (desktop QR / camera preview):
    pip install opencv-python
    # If QR still won’t read, try: pip install pyzbar  (+ macOS: brew install zbar)

Run:
    python agriculture.py
"""

from __future__ import annotations

import functools
import json
import os
import subprocess
import sys
import threading
import webbrowser
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import requests

try:
    import cv2

    QR_SCAN_AVAILABLE = True
except ImportError:
    cv2 = None  # type: ignore[misc, assignment]
    QR_SCAN_AVAILABLE = False

try:
    from PIL import Image as PILImage

    _PIL_OK = True
except ImportError:
    _PIL_OK = False


def _ensure_libzbar_load_path() -> None:
    """
    On macOS, Homebrew puts libzbar in /opt/homebrew/lib, but
    ``ctypes.util.find_library('zbar')`` often returns None, so ``pyzbar``
    fails with “Unable to find zbar shared library” even after
    ``brew install zbar``. Prepend that path to DYLD_FALLBACK_LIBRARY_PATH
    before any pyzbar import.
    """
    if sys.platform != "darwin":
        return
    for d in ("/opt/homebrew/lib", "/usr/local/lib"):
        if os.path.isfile(os.path.join(d, "libzbar.dylib")):
            key = "DYLD_FALLBACK_LIBRARY_PATH"
            cur = os.environ.get(key, "")
            parts = [p for p in cur.split(os.pathsep) if p]
            if d not in parts:
                os.environ[key] = d + (os.pathsep + cur if cur else "")
            break


_ensure_libzbar_load_path()

os.environ.setdefault("KIVY_NO_ENV_CONFIG", "1")

from kivy.app import App
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.graphics import Color, RoundedRectangle
from kivy.metrics import dp, sp
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.filechooser import FileChooserListView
from kivy.uix.image import AsyncImage, Image as KivyImage
from kivy.uix.label import Label
from kivy.uix.popup import Popup
from kivy.uix.progressbar import ProgressBar
from kivy.uix.scrollview import ScrollView
from kivy.uix.screenmanager import ScreenManager, Screen, SlideTransition
from kivy.uix.textinput import TextInput
from kivy.utils import get_color_from_hex
from kivy.utils import platform as kivy_platform

Window.size = (400, 800)

C = {
    "bg":       "#F5F7FA",
    "primary":  "#2ECC71",
    "blue":     "#3498DB",
    "danger":   "#E74C3C",
    "warning":  "#F39C12",
    "dark":     "#2C3E50",
    "card":     "#FFFFFF",
    "text":     "#2C3E50",
    "sub":      "#7F8C8D",
    "safe":     "#27AE60",
    "moderate": "#F39C12",
    "avoid":    "#E74C3C",
    "border":   "#ECF0F1",
    "purple":   "#9B59B6",
    "teal":     "#1ABC9C",
    "gold":     "#F1C40F",
    "info":     "#2980B9",
}
Window.clearcolor = get_color_from_hex(C["bg"] + "FF")


# ══════════════════════════════════════════════════════════════════════════════
# ❶  DATA FETCHING  (OpenFoodFacts)
# ══════════════════════════════════════════════════════════════════════════════

_OFF_BARCODE = "https://world.openfoodfacts.org/api/v0/product/{}.json"
_OFF_SEARCH  = "https://world.openfoodfacts.org/cgi/search.pl"
_OFF_FIELDS  = (
    "product_name,brands,image_url,code,nutriments,"
    "ingredients_text,allergens_tags,labels_tags,"
    "categories_tags,additives_tags,quantity,nutriscore_grade"
)

_SESSION = requests.Session()
_SESSION.headers.update({"User-Agent": "FoodScan/4.0 (educational)"})


# ══════════════════════════════════════════════════════════════════════════════
#  Desktop QR decode (OpenCV) — live feed + image file
# ══════════════════════════════════════════════════════════════════════════════

_QR_DETECTOR = None


@functools.lru_cache(maxsize=1)
def _pyzbar_import_ok() -> bool:
    try:
        import pyzbar.pyzbar  # noqa: F401
        return True
    except Exception:
        return False


def _qr_detector():
    global _QR_DETECTOR
    if not QR_SCAN_AVAILABLE:
        return None
    if _QR_DETECTOR is None:
        _QR_DETECTOR = cv2.QRCodeDetector()
    return _QR_DETECTOR


def _make_video_capture(index: int):
    """OpenCV capture with a Windows-friendly backend when needed."""
    if not QR_SCAN_AVAILABLE:
        return None
    if sys.platform == "win32":
        return cv2.VideoCapture(index, getattr(cv2, "CAP_DSHOW", 700))
    return cv2.VideoCapture(index)


def _pyzbar_decode_one(img) -> str:
    """Single-image pyzbar attempt; needs ``pip install pyzbar`` + system libzbar."""
    if not _pyzbar_import_ok():
        return ""
    from pyzbar import pyzbar
    if img is None or img.size == 0:
        return ""
    try:
        for sym in pyzbar.decode(img):
            if sym.data:
                return sym.data.decode("utf-8", errors="replace").strip()
    except Exception:
        return ""
    return ""


def _pyzbar_scan_image(bgr, deep: bool) -> str:
    """Try pyzbar on grayscale / RGB variants (invert, CLAHE, upscale, …)."""
    if bgr is None or bgr.size == 0:
        return ""
    if not _pyzbar_import_ok():
        return ""

    if len(bgr.shape) == 2:
        gray = bgr
    else:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    quick = (
        gray,
        cv2.flip(gray, 1),
        cv2.bitwise_not(gray),
        cv2.bitwise_not(cv2.flip(gray, 1)),
    )
    for im in quick:
        s = _pyzbar_decode_one(im)
        if s:
            return s
    if not deep:
        return ""

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    eq = clahe.apply(gray)
    for im in (eq, cv2.flip(eq, 1), cv2.bitwise_not(eq)):
        s = _pyzbar_decode_one(im)
        if s:
            return s
    try:
        _, ot = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        for im in (ot, cv2.flip(ot, 1), cv2.bitwise_not(ot)):
            s = _pyzbar_decode_one(im)
            if s:
                return s
    except cv2.error:
        pass

    h, w = gray.shape[:2]
    m = max(h, w)
    if m < 1400:
        sc = min(3.0, 1600 / max(m, 1))
        nh, nw = int(h * sc), int(w * sc)
        if nh > 8 and nw > 8:
            up = cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_CUBIC)
            for im in (up, cv2.flip(up, 1), cv2.bitwise_not(up)):
                s = _pyzbar_decode_one(im)
                if s:
                    return s

    if len(bgr.shape) == 3:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        s = _pyzbar_decode_one(rgb)
        if s:
            return s
    return ""


def load_image_bgr(path: str):
    """
    Load a BGR image for OpenCV. ``cv2.imread`` fails on many Unicode paths and
    some formats; we use bytes + imdecode and Pillow as fallbacks.
    """
    if not path or not QR_SCAN_AVAILABLE:
        return None
    path = str(path).strip()
    try:
        path = os.fsdecode(os.fsencode(path))
    except Exception:
        pass

    try:
        with open(path, "rb") as f:
            blob = f.read()
        if blob:
            buf = np.frombuffer(blob, dtype=np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if img is not None:
                return img
    except Exception:
        pass

    try:
        img = cv2.imread(path)
        if img is not None:
            return img
    except Exception:
        pass

    if _PIL_OK:
        try:
            with PILImage.open(path) as im:
                im = im.convert("RGB")
                rgb = np.asarray(im, dtype=np.uint8)
                return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        except Exception:
            pass
    return None


def decode_qr_bgr(bgr, deep: bool = True) -> str:
    """
    Decode first QR in a BGR frame. Tries several preprocessings because
    OpenCV often misses mirrored (selfie cam), dim, or small codes.

    When ``deep`` is False, only the fast variants run (good for every
    video frame); when True, also CLAHE, upscale, multi-decode, pyzbar.
    """
    if bgr is None or not QR_SCAN_AVAILABLE or bgr.size == 0:
        return ""
    det = _qr_detector()
    if det is None:
        return ""

    def try_one(img) -> str:
        if img is None or img.size == 0:
            return ""
        try:
            data, _, _ = det.detectAndDecode(img)
        except cv2.error:
            return ""
        return (data or "").strip()

    def try_multi(img) -> str:
        if img is None or img.size == 0:
            return ""
        try:
            ok, infos, _, _ = det.detectAndDecodeMulti(img)
        except cv2.error:
            return ""
        if not ok or infos is None:
            return ""
        for x in infos:
            if x:
                return str(x).strip()
        return ""

    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    s = _pyzbar_scan_image(bgr, deep=deep)
    if s:
        return s

    for img in (bgr, gray, cv2.flip(bgr, 1), cv2.flip(gray, 1)):
        s = try_one(img)
        if s:
            return s

    if not deep:
        return ""

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    eq = clahe.apply(gray)
    for img in (eq, cv2.flip(eq, 1)):
        s = try_one(img)
        if s:
            return s

    try:
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        s = try_one(otsu)
        if s:
            return s
    except cv2.error:
        pass

    m = max(h, w)
    if m < 1000:
        scale = min(2.0, 1400 / max(m, 1))
        nh, nw = int(h * scale), int(w * scale)
        if nh > 0 and nw > 0:
            up_bgr = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_CUBIC)
            up_gray = cv2.cvtColor(up_bgr, cv2.COLOR_BGR2GRAY)
            for img in (up_bgr, up_gray, cv2.flip(up_gray, 1)):
                s = try_one(img)
                if s:
                    return s

    for img in (bgr, gray, eq):
        s = try_multi(img)
        if s:
            return s

    return ""


def _decode_qr_still_image_extra(bgr) -> str:
    """Extra multi-scale OpenCV attempts for uploaded photos (no pyzbar required)."""
    if bgr is None or not QR_SCAN_AVAILABLE or bgr.size == 0:
        return ""
    det = _qr_detector()
    if det is None:
        return ""

    def try_one(img) -> str:
        if img is None or img.size == 0:
            return ""
        try:
            data, _, _ = det.detectAndDecode(img)
            if data:
                return data.strip()
        except cv2.error:
            pass
        return ""

    def try_multi(img) -> str:
        if img is None or img.size == 0:
            return ""
        try:
            ok, infos, _, _ = det.detectAndDecodeMulti(img)
            if ok and infos:
                for x in infos:
                    if x:
                        return str(x).strip()
        except cv2.error:
            pass
        return ""

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if len(bgr.shape) == 3 else bgr
    h, w = gray.shape[:2]
    for scale in (0.35, 0.5, 0.75, 1.25, 1.8, 2.8, 4.0):
        nh, nw = max(20, int(h * scale)), max(20, int(w * scale))
        if nh * nw > 14_000_000:
            continue
        resized = cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_CUBIC)
        for im in (resized, cv2.flip(resized, 1), cv2.bitwise_not(resized)):
            s = try_one(im) or try_multi(im)
            if s:
                return s
    return ""


def decode_qr_image_path(path: str) -> str:
    if not path or not QR_SCAN_AVAILABLE:
        return ""
    img = load_image_bgr(path)
    if img is None:
        return ""
    s = decode_qr_bgr(img, deep=True)
    if s:
        return s
    return _decode_qr_still_image_extra(img)


def pyzbar_usable() -> bool:
    """True if pyzbar can load (Python package + system libzbar)."""
    return _pyzbar_import_ok()


def probe_video_capture_indices(max_idx: int = 4) -> list[int]:
    """Return indices that yield at least one frame (for laptop 0 / 1 “cameras”)."""
    if not QR_SCAN_AVAILABLE:
        return []
    found: list[int] = []
    for i in range(max_idx):
        cap = _make_video_capture(i)
        if cap is None:
            continue
        try:
            if not cap.isOpened():
                continue
            ok, _ = cap.read()
            if ok:
                found.append(i)
        finally:
            cap.release()
    return found if found else [0]


def open_video_capture(index: int):
    cap = _make_video_capture(index)
    return cap if cap and cap.isOpened() else None


def classify_qr_payload(raw: str) -> tuple[str, str]:
    """
    Classify decoded QR content for routing.
    Returns (kind, value) where kind is url | barcode | text | empty.
    """
    s = (raw or "").strip()
    if not s:
        return "empty", ""
    parsed = urlparse(s)
    if parsed.scheme in ("http", "https"):
        return "url", s
    compact = "".join(ch for ch in s if ch.isdigit())
    alnum = "".join(ch for ch in s if ch.isalnum())
    if len(compact) >= 8 and (not alnum or len(compact) / max(len(alnum), 1) >= 0.85):
        return "barcode", compact
    return "text", s


def fetch_by_barcode(barcode: str) -> dict | None:
    try:
        r = _SESSION.get(_OFF_BARCODE.format(barcode.strip()), timeout=12)
        d = r.json()
        return d["product"] if d.get("status") == 1 else None
    except Exception:
        return None


def search_by_name(query: str, page_size: int = 10) -> list[dict]:
    try:
        r = _SESSION.get(_OFF_SEARCH, params={
            "search_terms": query, "search_simple": 1,
            "action": "process", "json": 1,
            "page_size": page_size, "fields": _OFF_FIELDS,
        }, timeout=12)
        return [p for p in r.json().get("products", [])
                if p.get("product_name")]
    except Exception:
        return []


# ══════════════════════════════════════════════════════════════════════════════
# ❷  SMART ANALYSIS ENGINE
# ══════════════════════════════════════════════════════════════════════════════

# ── 3a. Nutrient thresholds & rating ─────────────────────────────────────────

_THRESHOLDS: dict[str, tuple[float, float]] = {
    # key            (low=safe, high=avoid)  per 100 g
    "sugars":        (5.0,  22.5),
    "saturated-fat": (1.5,   5.0),
    "salt":          (0.3,   1.5),
    "sodium":        (0.1,   0.6),
    "fat":           (3.0,  17.5),
    "energy-kcal":   (150,  400),
    "fiber":         (3.0,   6.0),   # inverted – higher = better
    "proteins":      (5.0,  10.0),   # inverted
    "trans-fat":     (0.0,   0.5),   # any trans fat → moderate; >0.5 → avoid
}
_INVERT = {"fiber", "proteins"}


def rate_nutrient(key: str, per100: float) -> str:
    """Return 'safe' | 'moderate' | 'avoid' | 'neutral'."""
    k = key.lower().replace(" ", "-")
    if k not in _THRESHOLDS:
        return "neutral"
    lo, hi = _THRESHOLDS[k]
    if k in _INVERT:
        return "safe" if per100 >= hi else ("moderate" if per100 >= lo else "avoid")
    return "safe" if per100 <= lo else ("avoid" if per100 >= hi else "moderate")


def _fval(nutriments: dict, *keys) -> float:
    for k in keys:
        v = nutriments.get(k)
        if v is not None:
            try:
                return float(v)
            except (ValueError, TypeError):
                pass
    return 0.0


def energy_kcal_per100(nutriments: dict) -> float:
    """
    OpenFoodFacts commonly provides energy as:
      - energy-kcal_100g (kcal)  OR
      - energy_100g (kJ)

    This helper returns kcal/100g with a kJ→kcal fallback.
    """
    kcal = _fval(nutriments, "energy-kcal_100g", "energy-kcal")
    if kcal > 0:
        return kcal
    kj = _fval(nutriments, "energy_100g", "energy")
    if kj > 0:
        return kj / 4.184
    return 0.0


def build_nutrient_insights(nutriments: dict) -> list[dict]:
    checks = [
        ("sugars",        "Sugar"),
        ("saturated-fat", "Saturated Fat"),
        ("salt",          "Salt"),
        ("fat",           "Total Fat"),
        ("fiber",         "Dietary Fiber"),
        ("proteins",      "Protein"),
    ]
    out = []
    for key, label in checks:
        val = nutriments.get(f"{key}_100g") or nutriments.get(key)
        if val is not None:
            try:
                fv = float(val)
                out.append({"label": label,
                             "rating": rate_nutrient(key, fv),
                             "value": f"{fv:.1f} g"})
            except (ValueError, TypeError):
                pass
    return out


# ── 3b. Harmful additive detection ───────────────────────────────────────────

HARMFUL_ADDITIVES: dict[str, tuple[str, str]] = {
    "e102":  ("Tartrazine (E102)",           "avoid"),
    "e104":  ("Quinoline Yellow (E104)",     "moderate"),
    "e110":  ("Sunset Yellow (E110)",        "avoid"),
    "e120":  ("Carmine / Cochineal (E120)",  "moderate"),
    "e122":  ("Carmoisine (E122)",           "avoid"),
    "e124":  ("Ponceau 4R (E124)",           "avoid"),
    "e129":  ("Allura Red AC (E129)",        "avoid"),
    "e150d": ("Caramel IV (E150d)",          "moderate"),
    "e211":  ("Sodium Benzoate (E211)",      "avoid"),
    "e212":  ("Potassium Benzoate (E212)",   "avoid"),
    "e220":  ("Sulphur Dioxide (E220)",      "moderate"),
    "e249":  ("Potassium Nitrite (E249)",    "avoid"),
    "e250":  ("Sodium Nitrite (E250)",       "avoid"),
    "e320":  ("BHA (E320)",                  "avoid"),
    "e321":  ("BHT (E321)",                  "avoid"),
    "e407":  ("Carrageenan (E407)",          "moderate"),
    "e412":  ("Guar Gum (E412)",             "moderate"),
    "e621":  ("MSG / Glutamate (E621)",      "moderate"),
    "e951":  ("Aspartame (E951)",            "moderate"),
    "e952":  ("Cyclamate (E952)",            "avoid"),
    "e954":  ("Saccharin (E954)",            "moderate"),
    "e955":  ("Sucralose (E955)",            "moderate"),
    "e1422": ("Acetylated Distarch (E1422)", "moderate"),
}


def detect_additives(product: dict) -> list[dict]:
    found: dict[str, dict] = {}
    for tag in product.get("additives_tags", []):
        code = tag.split(":")[-1].lower().replace("-", "")
        if code in HARMFUL_ADDITIVES and code not in found:
            name, sev = HARMFUL_ADDITIVES[code]
            found[code] = {"name": name, "severity": sev,
                           "icon": "🚫" if sev == "avoid" else "⚠️"}
    raw = (product.get("ingredients_text") or "").upper()
    for code, (name, sev) in HARMFUL_ADDITIVES.items():
        if code.upper() in raw and code not in found:
            found[code] = {"name": name, "severity": sev,
                           "icon": "🚫" if sev == "avoid" else "⚠️"}
    return list(found.values())


# ── 3c. Ingredient keyword concerns ──────────────────────────────────────────

_CONCERN_KEYWORDS: list[tuple[str, str, str]] = [
    ("high fructose corn syrup", "High-Fructose Corn Syrup",      "avoid"),
    ("hydrogenated",             "Hydrogenated Oils (Trans Fat)",  "avoid"),
    ("partially hydrogenated",   "Partially Hydrogenated Oil",     "avoid"),
    ("artificial flavour",       "Artificial Flavours",            "moderate"),
    ("artificial flavor",        "Artificial Flavours",            "moderate"),
    ("artificial color",         "Artificial Colours",             "moderate"),
    ("artificial colour",        "Artificial Colours",             "moderate"),
    ("artificial sweetener",     "Artificial Sweeteners",          "moderate"),
    ("monosodium glutamate",     "MSG",                            "moderate"),
    ("sodium nitrate",           "Sodium Nitrate",                 "avoid"),
    ("potassium bromate",        "Potassium Bromate",              "avoid"),
    ("brominated vegetable",     "Brominated Vegetable Oil",       "avoid"),
    ("carrageenan",              "Carrageenan",                    "moderate"),
    ("acesulfame",               "Acesulfame-K",                   "moderate"),
    ("saccharin",                "Saccharin",                      "moderate"),
    ("aspartame",                "Aspartame",                      "moderate"),
]


def analyze_ingredients(product: dict) -> list[dict]:
    raw   = (product.get("ingredients_text") or "").lower()
    found, seen = [], set()
    for kw, label, sev in _CONCERN_KEYWORDS:
        if kw in raw and label not in seen:
            found.append({"label": label, "severity": sev,
                          "icon": "🚫" if sev == "avoid" else "⚠️"})
            seen.add(label)
    return found


# ── 3d. SMART RECOMMENDATION ENGINE ──────────────────────────────────────────
#
#  Generates plain-English, consumer-friendly recommendations.
#  Each recommendation is a dict:
#    { "text": str, "type": "positive"|"warning"|"danger", "icon": str }

def generate_recommendations(product: dict) -> list[dict]:
    """
    Core smart analysis logic.  Rules applied (in priority order):

    DANGER (avoid):
      • Sugar > 22.5 g/100g  → "High sugar content – limit intake"
      • Trans fat detected    → "Contains trans fat – linked to heart disease"
      • Sodium Nitrite/BHA/BHT found → "Contains [additive] – a harmful preservative"
      • Any 'avoid'-rated additive   → "Contains [name]"

    WARNING (moderate):
      • Sugar 5–22.5 g       → "Moderate sugar content – consume in moderation"
      • Saturated fat > 5 g  → "High saturated fat – may raise cholesterol"
      • Salt > 1.5 g         → "High salt content – watch your sodium intake"
      • Calories > 400 kcal  → "High calorie density – watch portion size"
      • Any 'moderate' additive      → "Contains preservatives / additives"
      • HFCS / artificial colours    → specific message

    POSITIVE (safe):
      • Protein ≥ 10 g       → "Good protein source"
      • Fiber ≥ 6 g          → "High in dietary fiber – good for digestion"
      • Fiber 3–6 g          → "Contains dietary fiber"
      • Sugar ≤ 5 g          → "Low in sugar"
      • Saturated fat ≤ 1.5g → "Low in saturated fat"
      • Salt ≤ 0.3 g         → "Low in salt"
      • No additives found   → "No harmful additives detected"
      • Calories ≤ 150 kcal  → "Low calorie option"
    """
    nm   = product.get("nutriments", {})
    ingr = (product.get("ingredients_text") or "").lower()

    sugar   = _fval(nm, "sugars_100g",        "sugars")
    sat_fat = _fval(nm, "saturated-fat_100g", "saturated-fat")
    salt    = _fval(nm, "salt_100g",          "salt")
    fiber   = _fval(nm, "fiber_100g",         "fiber")
    protein = _fval(nm, "proteins_100g",      "proteins")
    kcal    = energy_kcal_per100(nm)
    sodium  = _fval(nm, "sodium_100g",        "sodium")

    has_trans = ("hydrogenated" in ingr or
                 "partially hydrogenated" in ingr)
    has_hfcs  = "high fructose corn syrup" in ingr

    additives = detect_additives(product)
    avoid_adds    = [a for a in additives if a["severity"] == "avoid"]
    moderate_adds = [a for a in additives if a["severity"] == "moderate"]

    recs: list[dict] = []

    # ── DANGER rules ──────────────────────────────────────────────────────────
    if sugar > 22.5:
        recs.append({
            "text": f"High sugar content ({sugar:.1f} g/100g) – limit intake",
            "type": "danger", "icon": "🚫",
        })
    if has_trans:
        recs.append({
            "text": "Contains trans fat (hydrogenated oil) – linked to heart disease",
            "type": "danger", "icon": "🚫",
        })
    if has_hfcs:
        recs.append({
            "text": "Contains high-fructose corn syrup – strongly linked to obesity",
            "type": "danger", "icon": "🚫",
        })
    for a in avoid_adds:
        recs.append({
            "text": f"Contains {a['name']} – a harmful additive",
            "type": "danger", "icon": "🚫",
        })

    # ── WARNING rules ─────────────────────────────────────────────────────────
    if 5 < sugar <= 22.5:
        recs.append({
            "text": f"Moderate sugar content ({sugar:.1f} g/100g) – consume in moderation",
            "type": "warning", "icon": "⚠️",
        })
    if sat_fat > 5:
        recs.append({
            "text": f"High saturated fat ({sat_fat:.1f} g/100g) – may raise cholesterol",
            "type": "warning", "icon": "⚠️",
        })
    elif 1.5 < sat_fat <= 5:
        recs.append({
            "text": f"Moderate saturated fat ({sat_fat:.1f} g/100g)",
            "type": "warning", "icon": "⚠️",
        })
    if salt > 1.5:
        recs.append({
            "text": f"High salt content ({salt:.1f} g/100g) – watch your sodium intake",
            "type": "warning", "icon": "⚠️",
        })
    elif 0.3 < salt <= 1.5:
        recs.append({
            "text": f"Moderate salt content ({salt:.1f} g/100g)",
            "type": "warning", "icon": "⚠️",
        })
    if kcal > 400:
        recs.append({
            "text": f"High calorie density ({kcal:.0f} kcal/100g) – watch portion size",
            "type": "warning", "icon": "⚠️",
        })
    if moderate_adds:
        names = ", ".join(a["name"].split("(")[0].strip()
                          for a in moderate_adds[:3])
        recs.append({
            "text": f"Contains preservatives / additives: {names}",
            "type": "warning", "icon": "⚠️",
        })
    if "artificial color" in ingr or "artificial colour" in ingr:
        recs.append({
            "text": "Contains artificial colours – may affect behaviour in children",
            "type": "warning", "icon": "⚠️",
        })

    # ── POSITIVE rules ────────────────────────────────────────────────────────
    if protein >= 10:
        recs.append({
            "text": f"Good protein source ({protein:.1f} g/100g) – supports muscle health",
            "type": "positive", "icon": "✅",
        })
    elif 5 <= protein < 10:
        recs.append({
            "text": f"Contains protein ({protein:.1f} g/100g)",
            "type": "positive", "icon": "✅",
        })
    if fiber >= 6:
        recs.append({
            "text": f"High in dietary fiber ({fiber:.1f} g/100g) – great for digestion",
            "type": "positive", "icon": "✅",
        })
    elif 3 <= fiber < 6:
        recs.append({
            "text": f"Contains dietary fiber ({fiber:.1f} g/100g) – good for gut health",
            "type": "positive", "icon": "✅",
        })
    if sugar <= 5 and sugar > 0:
        recs.append({
            "text": f"Low in sugar ({sugar:.1f} g/100g) – a healthier choice",
            "type": "positive", "icon": "✅",
        })
    if sat_fat <= 1.5 and sat_fat > 0:
        recs.append({
            "text": "Low in saturated fat – heart-friendly",
            "type": "positive", "icon": "✅",
        })
    if salt <= 0.3 and salt > 0:
        recs.append({
            "text": "Low in salt – good for blood pressure",
            "type": "positive", "icon": "✅",
        })
    if kcal <= 150 and kcal > 0:
        recs.append({
            "text": f"Low calorie option ({kcal:.0f} kcal/100g)",
            "type": "positive", "icon": "✅",
        })
    if not additives:
        recs.append({
            "text": "No harmful additives detected – clean label",
            "type": "positive", "icon": "✅",
        })

    return recs


def smart_overall_verdict(product: dict,
                           insights: list[dict],
                           recs: list[dict]) -> str:
    """
    Combine nutrient insights + smart recommendations into one final verdict.
    Returns 'safe' | 'moderate' | 'avoid'.
    """
    danger_count   = sum(1 for r in recs if r["type"] == "danger")
    warning_count  = sum(1 for r in recs if r["type"] == "warning")
    positive_count = sum(1 for r in recs if r["type"] == "positive")

    avoid_n    = sum(1 for i in insights if i["rating"] == "avoid")
    moderate_n = sum(1 for i in insights if i["rating"] == "moderate")

    if danger_count >= 2 or avoid_n >= 2:
        return "avoid"
    if danger_count >= 1 or avoid_n >= 1 or warning_count >= 3 or moderate_n >= 3:
        return "moderate"
    if warning_count >= 1 or moderate_n >= 1:
        return "moderate"
    return "safe"


# ══════════════════════════════════════════════════════════════════════════════
# ❹  ALLERGY ALERTS
# ══════════════════════════════════════════════════════════════════════════════

ALLERGEN_MAP: dict[str, tuple[str, str]] = {
    "gluten":    ("🌾", "Gluten"),
    "wheat":     ("🌾", "Wheat"),
    "milk":      ("🥛", "Dairy / Milk"),
    "nuts":      ("🥜", "Tree Nuts"),
    "peanuts":   ("🥜", "Peanuts"),
    "eggs":      ("🥚", "Eggs"),
    "soy":       ("🫘", "Soy"),
    "shellfish": ("🦐", "Shellfish"),
    "fish":      ("🐟", "Fish"),
    "sesame":    ("🌿", "Sesame"),
    "celery":    ("🥬", "Celery"),
    "mustard":   ("🌱", "Mustard"),
    "lupin":     ("🌼", "Lupin"),
    "molluscs":  ("🐚", "Molluscs"),
    "sulphites": ("⚗️",  "Sulphites"),
}


def detect_allergens(product: dict) -> list[dict]:
    tags     = " ".join(product.get("allergens_tags", [])).lower()
    raw      = (product.get("ingredients_text") or "").lower()
    combined = tags + " " + raw
    found, seen = [], set()
    for key, (icon, label) in ALLERGEN_MAP.items():
        if key in combined and label not in seen:
            found.append({"icon": icon, "label": label})
            seen.add(label)
    return found


# ══════════════════════════════════════════════════════════════════════════════
# ❺  DIET COMPATIBILITY
# ══════════════════════════════════════════════════════════════════════════════

_NON_VEGAN = ["milk", "egg", "honey", "gelatin", "meat", "fish",
              "chicken", "beef", "pork", "lard", "whey", "casein",
              "lactose", "butter", "cream", "anchovy"]
_NON_HALAL = ["pork", "lard", "gelatin", "alcohol", "wine", "beer",
              "rum", "whisky", "vodka", "bacon", "ham"]
_GLUTEN_KW = ["wheat", "gluten", "barley", "rye", "spelt",
              "kamut", "triticale", "malt"]
_NON_VEG   = ["meat", "chicken", "beef", "pork", "fish", "seafood",
              "gelatin", "lard", "anchovy", "bacon", "ham"]


def check_diet_compatibility(product: dict) -> list[dict]:
    labels   = " ".join(product.get("labels_tags", [])).lower()
    cats     = " ".join(product.get("categories_tags", [])).lower()
    ingr     = (product.get("ingredients_text") or "").lower()
    combined = labels + " " + cats + " " + ingr
    nm       = product.get("nutriments", {})

    def _f(key: str) -> float:
        try:
            return float(nm.get(key) or 0)
        except (ValueError, TypeError):
            return 0.0

    carbs  = _f("carbohydrates_100g")
    fiber  = _f("fiber_100g")
    sugar  = _f("sugars_100g")
    sodium = _f("sodium_100g")
    net    = carbs - fiber
    results = []

    if "vegan" in labels:
        v_ok, v_note = True, "Labelled vegan"
    elif any(w in combined for w in _NON_VEGAN):
        v_ok, v_note = False, "Contains animal-derived ingredients"
    else:
        v_ok, v_note = None, "No animal ingredients detected (unverified)"
    results.append({"diet": "Vegan", "icon": "🌱",
                    "compatible": v_ok, "note": v_note})

    if "vegetarian" in labels:
        vg_ok, vg_note = True, "Labelled vegetarian"
    elif any(w in combined for w in _NON_VEG):
        vg_ok, vg_note = False, "Contains meat / fish"
    else:
        vg_ok, vg_note = None, "No meat detected (unverified)"
    results.append({"diet": "Vegetarian", "icon": "🥦",
                    "compatible": vg_ok, "note": vg_note})

    if carbs > 0:
        k_ok, k_note = net <= 5, f"Net carbs: {net:.1f} g / 100 g"
    else:
        k_ok, k_note = None, "Carbohydrate data unavailable"
    results.append({"diet": "Keto", "icon": "🥑",
                    "compatible": k_ok, "note": k_note})

    if sugar > 0:
        d_ok, d_note = sugar <= 5, f"Sugar: {sugar:.1f} g / 100 g"
    else:
        d_ok, d_note = None, "Sugar data unavailable"
    results.append({"diet": "Diabetic-Friendly", "icon": "💉",
                    "compatible": d_ok, "note": d_note})

    if "halal" in labels:
        h_ok, h_note = True, "Labelled halal"
    elif any(w in combined for w in _NON_HALAL):
        h_ok, h_note = False, "Contains non-halal ingredients"
    else:
        h_ok, h_note = None, "No non-halal ingredients detected (unverified)"
    results.append({"diet": "Halal", "icon": "☪️",
                    "compatible": h_ok, "note": h_note})

    if "gluten-free" in labels or "sans-gluten" in labels:
        gf_ok, gf_note = True, "Labelled gluten-free"
    elif any(w in combined for w in _GLUTEN_KW):
        gf_ok, gf_note = False, "Contains gluten sources"
    else:
        gf_ok, gf_note = None, "No gluten detected (unverified)"
    results.append({"diet": "Gluten-Free", "icon": "🌾",
                    "compatible": gf_ok, "note": gf_note})

    if sodium > 0:
        ls_ok   = sodium <= 0.12
        ls_note = f"Sodium: {sodium * 1000:.0f} mg / 100 g"
    else:
        ls_ok, ls_note = None, "Sodium data unavailable"
    results.append({"diet": "Low-Sodium", "icon": "🧂",
                    "compatible": ls_ok, "note": ls_note})

    return results


# ══════════════════════════════════════════════════════════════════════════════
# ❻  HISTORY & FAVOURITES
# ══════════════════════════════════════════════════════════════════════════════

_DATA_FILE   = Path("foodscan_data.json")
_MAX_HISTORY = 30


def _load_data() -> dict:
    if _DATA_FILE.exists():
        try:
            return json.loads(_DATA_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"history": [], "favourites": []}


def _save_data(data: dict) -> None:
    try:
        _DATA_FILE.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


def _make_entry(product: dict) -> dict:
    nm   = product.get("nutriments", {})
    kcal = energy_kcal_per100(nm)
    return {
        "barcode":    product.get("code", ""),
        "name":       product.get("product_name") or "Unknown",
        "brand":      product.get("brands", ""),
        "image_url":  product.get("image_url", ""),
        "calories":   round(kcal, 1),
        "scanned_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }


def history_add(product: dict) -> None:
    data  = _load_data()
    entry = _make_entry(product)
    data["history"] = [h for h in data["history"]
                       if h.get("barcode") != entry["barcode"]]
    data["history"].insert(0, entry)
    data["history"] = data["history"][:_MAX_HISTORY]
    _save_data(data)


def history_get() -> list[dict]:
    return _load_data().get("history", [])


def favourites_toggle(product: dict) -> bool:
    data    = _load_data()
    entry   = _make_entry(product)
    barcode = entry["barcode"]
    favs    = data.get("favourites", [])
    if any(f.get("barcode") == barcode for f in favs):
        data["favourites"] = [f for f in favs if f.get("barcode") != barcode]
        _save_data(data)
        return False
    favs.insert(0, entry)
    data["favourites"] = favs
    _save_data(data)
    return True


def is_favourite(barcode: str) -> bool:
    return any(f.get("barcode") == barcode
               for f in _load_data().get("favourites", []))


def favourites_get() -> list[dict]:
    return _load_data().get("favourites", [])


# ══════════════════════════════════════════════════════════════════════════════
# ❸  UI WIDGETS
# ══════════════════════════════════════════════════════════════════════════════

class RoundedButton(Button):
    def __init__(self, bg_color=None, text_color=None, radius=12, **kwargs):
        super().__init__(**kwargs)
        self.bg_hex   = bg_color   or C["primary"]
        self.text_hex = text_color or "#FFFFFF"
        self.radius   = radius
        self.background_normal = ""
        self.background_color  = (0, 0, 0, 0)
        self.color             = get_color_from_hex(self.text_hex)
        self.font_size         = sp(14)
        self.bold              = True
        self.bind(pos=self._draw, size=self._draw)

    def _draw(self, *_):
        self.canvas.before.clear()
        with self.canvas.before:
            Color(*get_color_from_hex(self.bg_hex + "FF"))
            RoundedRectangle(pos=self.pos, size=self.size,
                             radius=[dp(self.radius)])


class Card(BoxLayout):
    def __init__(self, **kwargs):
        kwargs.setdefault("orientation", "vertical")
        kwargs.setdefault("padding",     dp(14))
        kwargs.setdefault("spacing",     dp(8))
        super().__init__(**kwargs)
        self.bind(pos=self._draw, size=self._draw)

    def _draw(self, *_):
        self.canvas.before.clear()
        with self.canvas.before:
            Color(*get_color_from_hex(C["card"] + "FF"))
            RoundedRectangle(pos=self.pos, size=self.size, radius=[dp(14)])


class SectionTitle(Label):
    def __init__(self, text, **kwargs):
        super().__init__(
            text=text, font_size=sp(13), bold=True,
            color=get_color_from_hex(C["dark"] + "FF"),
            halign="left", valign="middle",
            size_hint_y=None, height=dp(26), **kwargs,
        )
        self.bind(size=lambda w, s: setattr(w, "text_size", s))


class NutrientRow(BoxLayout):
    def __init__(self, name, value, rating="neutral", **kwargs):
        super().__init__(orientation="horizontal",
                         size_hint_y=None, height=dp(34), **kwargs)
        dot_c = {"safe": C["safe"], "moderate": C["moderate"],
                 "avoid": C["avoid"], "neutral": C["sub"]}.get(rating, C["sub"])
        self.add_widget(Label(
            text="●", color=get_color_from_hex(dot_c + "FF"),
            size_hint_x=None, width=dp(18), font_size=sp(9),
        ))
        n = Label(text=name, color=get_color_from_hex(C["text"] + "FF"),
                  halign="left", valign="middle", font_size=sp(12))
        n.bind(size=lambda w, s: setattr(w, "text_size", s))
        self.add_widget(n)
        v = Label(text=str(value), color=get_color_from_hex(dot_c + "FF"),
                  halign="right", valign="middle",
                  size_hint_x=None, width=dp(80), font_size=sp(12), bold=True)
        v.bind(size=lambda w, s: setattr(w, "text_size", s))
        self.add_widget(v)


class InsightBadge(BoxLayout):
    _MAP = {
        "safe":     (C["safe"],     "✔ Safe"),
        "moderate": (C["moderate"], "⚠ Moderate"),
        "avoid":    (C["avoid"],    "✘ Avoid"),
    }

    def __init__(self, label, rating, **kwargs):
        super().__init__(size_hint_y=None, height=dp(38),
                         padding=[dp(10), dp(4)], **kwargs)
        hex_c, prefix = self._MAP.get(rating, (C["sub"], ""))
        self._hex = hex_c
        self.bind(pos=self._draw, size=self._draw)
        self.add_widget(Label(
            text=f"{prefix}  {label}",
            color=get_color_from_hex("#FFFFFFFF"),
            font_size=sp(11), bold=True,
        ))

    def _draw(self, *_):
        self.canvas.before.clear()
        with self.canvas.before:
            Color(*get_color_from_hex(self._hex + "CC"))
            RoundedRectangle(pos=self.pos, size=self.size, radius=[dp(20)])


class RecommendationRow(BoxLayout):
    """
    A single plain-English recommendation line.
    type: 'positive' → green  |  'warning' → amber  |  'danger' → red
    """
    _TYPE_COLOR = {
        "positive": C["safe"],
        "warning":  C["warning"],
        "danger":   C["danger"],
    }

    def __init__(self, text: str, rec_type: str, icon: str, **kwargs):
        super().__init__(orientation="horizontal",
                         size_hint_y=None, height=dp(44),
                         padding=[dp(10), dp(4)], spacing=dp(8), **kwargs)
        hex_c = self._TYPE_COLOR.get(rec_type, C["sub"])
        self._hex = hex_c
        self.bind(pos=self._draw, size=self._draw)

        self.add_widget(Label(
            text=icon, font_size=sp(16),
            size_hint_x=None, width=dp(26),
            color=get_color_from_hex("#FFFFFFFF"),
        ))
        lbl = Label(
            text=text,
            font_size=sp(11), bold=True,
            color=get_color_from_hex("#FFFFFFFF"),
            halign="left", valign="middle",
        )
        lbl.bind(size=lambda w, s: setattr(w, "text_size", s))
        self.add_widget(lbl)

    def _draw(self, *_):
        self.canvas.before.clear()
        with self.canvas.before:
            Color(*get_color_from_hex(self._hex + "CC"))
            RoundedRectangle(pos=self.pos, size=self.size, radius=[dp(10)])


class VerdictBanner(BoxLayout):
    """Large coloured verdict banner shown at the top of the detail view."""
    _CFG = {
        "safe":     (C["safe"],    "✔  Generally Safe to Consume"),
        "moderate": (C["warning"], "⚠  Consume in Moderation"),
        "avoid":    (C["danger"],  "✘  Limit or Avoid This Product"),
        "neutral":  (C["sub"],     "ℹ  Insufficient Data"),
    }

    def __init__(self, verdict: str, **kwargs):
        super().__init__(size_hint_y=None, height=dp(52),
                         padding=[dp(14), dp(6)], **kwargs)
        hex_c, text = self._CFG.get(verdict, self._CFG["neutral"])
        self._hex = hex_c
        self.bind(pos=self._draw, size=self._draw)
        self.add_widget(Label(
            text=text, font_size=sp(15), bold=True,
            color=get_color_from_hex("#FFFFFFFF"),
            halign="center",
        ))

    def _draw(self, *_):
        self.canvas.before.clear()
        with self.canvas.before:
            Color(*get_color_from_hex(self._hex + "FF"))
            RoundedRectangle(pos=self.pos, size=self.size, radius=[dp(12)])


class AllergenChip(BoxLayout):
    def __init__(self, icon, label, **kwargs):
        super().__init__(size_hint=(None, None), size=(dp(112), dp(32)),
                         padding=[dp(8), dp(4)], **kwargs)
        self.bind(pos=self._draw, size=self._draw)
        self.add_widget(Label(
            text=f"{icon} {label}",
            color=get_color_from_hex("#FFFFFFFF"),
            font_size=sp(10), bold=True,
        ))

    def _draw(self, *_):
        self.canvas.before.clear()
        with self.canvas.before:
            Color(*get_color_from_hex(C["danger"] + "DD"))
            RoundedRectangle(pos=self.pos, size=self.size, radius=[dp(16)])


class DietChip(BoxLayout):
    def __init__(self, icon, label, compatible, **kwargs):
        super().__init__(size_hint=(None, None), size=(dp(132), dp(32)),
                         padding=[dp(8), dp(4)], **kwargs)
        if compatible is True:
            hex_c, tick = C["safe"],   "✔"
        elif compatible is False:
            hex_c, tick = C["danger"], "✘"
        else:
            hex_c, tick = C["sub"],    "?"
        self._hex = hex_c
        self.bind(pos=self._draw, size=self._draw)
        self.add_widget(Label(
            text=f"{icon} {tick} {label}",
            color=get_color_from_hex("#FFFFFFFF"),
            font_size=sp(10), bold=True,
        ))

    def _draw(self, *_):
        self.canvas.before.clear()
        with self.canvas.before:
            Color(*get_color_from_hex(self._hex + "CC"))
            RoundedRectangle(pos=self.pos, size=self.size, radius=[dp(16)])


class AdditiveBadge(BoxLayout):
    def __init__(self, name, severity, **kwargs):
        super().__init__(size_hint_y=None, height=dp(34),
                         padding=[dp(10), dp(4)], **kwargs)
        hex_c    = C["avoid"] if severity == "avoid" else C["warning"]
        icon     = "🚫" if severity == "avoid" else "⚠️"
        self._hex = hex_c
        self.bind(pos=self._draw, size=self._draw)
        self.add_widget(Label(
            text=f"{icon}  {name}",
            color=get_color_from_hex("#FFFFFFFF"),
            font_size=sp(11), bold=True, halign="left",
        ))

    def _draw(self, *_):
        self.canvas.before.clear()
        with self.canvas.before:
            Color(*get_color_from_hex(self._hex + "CC"))
            RoundedRectangle(pos=self.pos, size=self.size, radius=[dp(10)])


class NutriScoreBadge(BoxLayout):
    _GRADE_COLORS = {
        "a": "#1E8F4E", "b": "#6DB33F",
        "c": "#F5A623", "d": "#E07B39", "e": "#D63B2F",
    }

    def __init__(self, grade: str, **kwargs):
        # Allow callers to override size/size_hint without hitting
        # "got multiple values for keyword argument" errors.
        kwargs.setdefault("size_hint", (None, None))
        kwargs.setdefault("size", (dp(56), dp(56)))
        super().__init__(**kwargs)
        g     = grade.lower()
        hex_c = self._GRADE_COLORS.get(g, C["sub"])
        self._hex = hex_c
        self.bind(pos=self._draw, size=self._draw)
        self.add_widget(Label(
            text=grade.upper(), font_size=sp(22), bold=True,
            color=get_color_from_hex("#FFFFFFFF"),
        ))

    def _draw(self, *_):
        self.canvas.before.clear()
        with self.canvas.before:
            Color(*get_color_from_hex(self._hex + "FF"))
            RoundedRectangle(pos=self.pos, size=self.size, radius=[dp(10)])


def _wrap_chips(chips_data, chip_factory, wrap_width: int = 360) -> BoxLayout:
    container = BoxLayout(orientation="vertical",
                          size_hint_y=None, spacing=dp(6))
    row   = BoxLayout(size_hint_y=None, height=dp(36), spacing=dp(6))
    row_w = 0
    rows  = [row]
    for data in chips_data:
        chip   = chip_factory(*data)
        chip_w = chip.width
        if row_w + chip_w > dp(wrap_width) and row_w > 0:
            row   = BoxLayout(size_hint_y=None, height=dp(36), spacing=dp(6))
            rows.append(row)
            row_w = 0
        row.add_widget(chip)
        row_w += chip_w + dp(6)
    for r in rows:
        container.add_widget(r)
    container.height = len(rows) * dp(42)
    return container


# ══════════════════════════════════════════════════════════════════════════════
# SCREENS
# ══════════════════════════════════════════════════════════════════════════════

class HomeScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        root = BoxLayout(orientation="vertical", padding=dp(18), spacing=dp(12))

        hdr = BoxLayout(orientation="vertical", size_hint_y=None,
                        height=dp(86), spacing=dp(2))
        hdr.add_widget(Label(
            text="🥦 FoodScan", font_size=sp(28), bold=True,
            color=get_color_from_hex(C["primary"] + "FF"),
            size_hint_y=None, height=dp(46),
        ))
        hdr.add_widget(Label(
            text="Scan · Understand · Decide",
            font_size=sp(13), color=get_color_from_hex(C["sub"] + "FF"),
            size_hint_y=None, height=dp(22),
        ))
        root.add_widget(hdr)

        sc = Card(size_hint_y=None, height=dp(118))
        sc.add_widget(SectionTitle("🔍  Search by food name"))
        self.search_in = TextInput(
            hint_text="e.g. Nutella, Coca-Cola, Oreo…",
            multiline=False, size_hint_y=None, height=dp(42),
            font_size=sp(13), padding=[dp(10), dp(10)],
            background_color=get_color_from_hex(C["border"] + "FF"),
            foreground_color=get_color_from_hex(C["text"] + "FF"),
            cursor_color=get_color_from_hex(C["primary"] + "FF"),
        )
        self.search_in.bind(on_text_validate=self._do_search)
        sb = RoundedButton(text="Search", size_hint_y=None, height=dp(38),
                           bg_color=C["blue"])
        sb.bind(on_release=self._do_search)
        sc.add_widget(self.search_in)
        sc.add_widget(sb)
        root.add_widget(sc)

        bc = Card(size_hint_y=None, height=dp(118))
        bc.add_widget(SectionTitle("🔢  Enter barcode / QR number"))
        self.barcode_in = TextInput(
            hint_text="e.g. 3017620422003",
            multiline=False, size_hint_y=None, height=dp(42),
            font_size=sp(13), padding=[dp(10), dp(10)],
            input_filter="int",
            background_color=get_color_from_hex(C["border"] + "FF"),
            foreground_color=get_color_from_hex(C["text"] + "FF"),
            cursor_color=get_color_from_hex(C["primary"] + "FF"),
        )
        self.barcode_in.bind(on_text_validate=self._do_barcode)
        bb = RoundedButton(text="Look Up Barcode", size_hint_y=None,
                           height=dp(38), bg_color=C["primary"])
        bb.bind(on_release=self._do_barcode)
        bc.add_widget(self.barcode_in)
        bc.add_widget(bb)
        root.add_widget(bc)

        action_row = BoxLayout(size_hint_y=None, height=dp(52), spacing=dp(6))
        qr_btn = RoundedButton(
            text="📷  QR", bg_color=C["primary"], size_hint_x=1, font_size=sp(11),
        )
        qr_btn.bind(on_release=self._open_qr_scan)
        hist_btn = RoundedButton(
            text="🕘  History", bg_color=C["dark"], size_hint_x=1, font_size=sp(11),
        )
        hist_btn.bind(on_release=self._open_history)
        fav_btn = RoundedButton(
            text="⭐  Fav", bg_color=C["purple"], size_hint_x=1, font_size=sp(11),
        )
        fav_btn.bind(on_release=self._open_favourites)
        action_row.add_widget(qr_btn)
        action_row.add_widget(hist_btn)
        action_row.add_widget(fav_btn)
        root.add_widget(action_row)

        demo = Card(size_hint_y=None, height=dp(96))
        demo.add_widget(Label(
            text="Try a demo product:",
            font_size=sp(11), color=get_color_from_hex(C["sub"] + "FF"),
            size_hint_y=None, height=dp(18),
        ))
        chips = BoxLayout(spacing=dp(8), size_hint_y=None, height=dp(40))
        for name, code in [("Nutella",  "3017620422003"),
                            ("Pepsi",    "012000001765"),
                            ("Oreo",     "7622210951137"),
                            ("Pringles", "5053990101536")]:
            b = RoundedButton(text=name, bg_color=C["dark"],
                              size_hint_x=1, height=dp(38), radius=20,
                              font_size=sp(11))
            b.barcode = code
            b.bind(on_release=self._demo_tap)
            chips.add_widget(b)
        demo.add_widget(chips)
        root.add_widget(demo)

        root.add_widget(BoxLayout())
        self.add_widget(root)

    def _do_search(self, *_):
        q = self.search_in.text.strip()
        if not q:
            return
        rs = self.manager.get_screen("results")
        rs._prev = "home"
        rs.load_search(q)
        self.manager.transition = SlideTransition(direction="left")
        self.manager.current = "results"

    def _do_barcode(self, *_):
        code = self.barcode_in.text.strip()
        if not code:
            return
        self._go_detail(code)

    def _demo_tap(self, btn):
        self._go_detail(btn.barcode)

    def _go_detail(self, barcode: str):
        detail = self.manager.get_screen("detail")
        detail._prev = "home"
        detail.load_barcode(barcode)
        self.manager.transition = SlideTransition(direction="left")
        self.manager.current = "detail"

    def _open_qr_scan(self, *_):
        self.manager.transition = SlideTransition(direction="left")
        self.manager.current = "qrscan"

    def _open_history(self, *_):
        self.manager.get_screen("history").load(mode="history")
        self.manager.transition = SlideTransition(direction="left")
        self.manager.current = "history"

    def _open_favourites(self, *_):
        self.manager.get_screen("history").load(mode="favourites")
        self.manager.transition = SlideTransition(direction="left")
        self.manager.current = "history"


class ResultsScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._prev = "home"
        lay = BoxLayout(orientation="vertical", padding=dp(14), spacing=dp(10))

        hdr = BoxLayout(size_hint_y=None, height=dp(48), spacing=dp(10))
        back = RoundedButton(text="← Back", size_hint_x=None, width=dp(76),
                             bg_color=C["dark"], radius=20)
        back.bind(on_release=self._go_back)
        self.title_lbl = Label(
            text="Search Results", font_size=sp(17), bold=True,
            color=get_color_from_hex(C["dark"] + "FF"),
        )
        hdr.add_widget(back)
        hdr.add_widget(self.title_lbl)
        lay.add_widget(hdr)

        self.progress = ProgressBar(max=100, value=0,
                                    size_hint_y=None, height=dp(5))
        lay.add_widget(self.progress)

        sv = ScrollView()
        self.box = BoxLayout(orientation="vertical",
                             spacing=dp(10), size_hint_y=None)
        self.box.bind(minimum_height=self.box.setter("height"))
        sv.add_widget(self.box)
        lay.add_widget(sv)
        self.add_widget(lay)

    def load_search(self, query: str):
        self.title_lbl.text = f'Results: "{query}"'
        self.box.clear_widgets()
        self.progress.value = 20
        self.box.add_widget(Label(
            text="Searching…", font_size=sp(13),
            color=get_color_from_hex(C["sub"] + "FF"),
            size_hint_y=None, height=dp(40),
        ))

        def _worker():
            try:
                found = search_by_name(query)
            except Exception:
                found = []
            Clock.schedule_once(lambda dt: self._show(found), 0)

        threading.Thread(target=_worker, daemon=True).start()

    def _show(self, products):
        self.progress.value = 100
        self.box.clear_widgets()
        if not products:
            self.box.add_widget(Label(
                text="No results found. Try a different name.",
                font_size=sp(13), color=get_color_from_hex(C["sub"] + "FF"),
                size_hint_y=None, height=dp(50),
            ))
            return
        for p in products:
            self.box.add_widget(self._make_card(p))

    def _make_card(self, p):
        name    = p.get("product_name") or "Unknown Product"
        brand   = p.get("brands", "")
        img_url = p.get("image_url", "")
        grade   = (p.get("nutriscore_grade") or "").lower()

        # Quick smart verdict for the list card
        insights = build_nutrient_insights(p.get("nutriments", {}))
        recs     = generate_recommendations(p)
        verdict  = smart_overall_verdict(p, insights, recs)
        v_dot    = {"safe": "🟢", "moderate": "🟡", "avoid": "🔴"}.get(verdict, "⚪")

        card = Card(size_hint_y=None, height=dp(92),
                    orientation="horizontal", spacing=dp(10))
        if img_url:
            card.add_widget(AsyncImage(source=img_url, size_hint_x=None,
                                       width=dp(68), allow_stretch=True,
                                       keep_ratio=True))
        else:
            card.add_widget(Label(text="🍽", font_size=sp(28),
                                  size_hint_x=None, width=dp(60)))

        info = BoxLayout(orientation="vertical", spacing=dp(2))
        name_row = BoxLayout(size_hint_y=None, height=dp(22), spacing=dp(6))
        name_row.add_widget(Label(
            text=name[:30] + ("…" if len(name) > 30 else ""),
            font_size=sp(12), bold=True,
            color=get_color_from_hex(C["dark"] + "FF"),
            halign="left", size_hint_y=None, height=dp(20),
        ))
        if grade in ("a", "b", "c", "d", "e"):
            name_row.add_widget(NutriScoreBadge(
                grade, size_hint_x=None, size=(dp(28), dp(22))))
        info.add_widget(name_row)
        info.add_widget(Label(
            text=f"{v_dot}  {brand[:24] if brand else '—'}",
            font_size=sp(11), color=get_color_from_hex(C["sub"] + "FF"),
            halign="left", size_hint_y=None, height=dp(16),
        ))
        btn = RoundedButton(text="View Details →", size_hint_y=None,
                            height=dp(28), bg_color=C["blue"], radius=8)
        btn.product = p
        btn.bind(on_release=self._open)
        info.add_widget(btn)
        card.add_widget(info)
        return card

    def _open(self, btn):
        detail = self.manager.get_screen("detail")
        detail._prev = "results"
        detail.load_product(btn.product)
        self.manager.transition = SlideTransition(direction="left")
        self.manager.current = "detail"

    def _go_back(self, *_):
        self.manager.transition = SlideTransition(direction="right")
        self.manager.current = getattr(self, "_prev", "home")


class DetailScreen(Screen):
    """Full product view with smart analysis engine output."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._prev    = "home"
        self._product = None

        root = BoxLayout(orientation="vertical", padding=dp(14), spacing=dp(8))

        hdr = BoxLayout(size_hint_y=None, height=dp(48), spacing=dp(8))
        back = RoundedButton(text="← Back", size_hint_x=None, width=dp(72),
                             bg_color=C["dark"], radius=20)
        back.bind(on_release=self._go_back)
        self.hdr_title = Label(
            text="Product Detail", font_size=sp(14), bold=True,
            color=get_color_from_hex(C["dark"] + "FF"),
        )
        self.fav_btn = RoundedButton(
            text="☆", size_hint_x=None, width=dp(44),
            bg_color=C["sub"], radius=22,
        )
        self.fav_btn.bind(on_release=self._toggle_fav)
        hdr.add_widget(back)
        hdr.add_widget(self.hdr_title)
        hdr.add_widget(self.fav_btn)
        root.add_widget(hdr)

        self.status_lbl = Label(
            text="", font_size=sp(12),
            color=get_color_from_hex(C["sub"] + "FF"),
            size_hint_y=None, height=dp(20),
        )
        root.add_widget(self.status_lbl)

        sv = ScrollView()
        self.content = BoxLayout(orientation="vertical",
                                 spacing=dp(10), size_hint_y=None,
                                 padding=[0, 0, 0, dp(20)])
        self.content.bind(minimum_height=self.content.setter("height"))
        sv.add_widget(self.content)
        root.add_widget(sv)
        self.add_widget(root)

    def load_barcode(self, barcode: str):
        self.content.clear_widgets()
        self.status_lbl.text = f"Fetching {barcode}…"

        def _worker():
            try:
                product = fetch_by_barcode(barcode)
            except Exception:
                product = None
            Clock.schedule_once(lambda dt: self._render(product), 0)

        threading.Thread(target=_worker, daemon=True).start()

    def load_product(self, product: dict):
        Clock.schedule_once(lambda dt: self._render(product), 0)

    def _toggle_fav(self, *_):
        if not self._product:
            return
        now_fav = favourites_toggle(self._product)
        self.fav_btn.text   = "★" if now_fav else "☆"
        self.fav_btn.bg_hex = C["gold"] if now_fav else C["sub"]
        self.fav_btn._draw()

    def _update_fav_btn(self, barcode: str):
        fav = is_favourite(barcode)
        self.fav_btn.text   = "★" if fav else "☆"
        self.fav_btn.bg_hex = C["gold"] if fav else C["sub"]
        self.fav_btn._draw()

    def _render(self, product):
        self.content.clear_widgets()
        self.status_lbl.text = ""

        if not product:
            self.content.add_widget(Label(
                text="❌  Product not found.\nCheck the barcode and try again.",
                font_size=sp(14), halign="center",
                color=get_color_from_hex(C["danger"] + "FF"),
                size_hint_y=None, height=dp(80),
            ))
            return

        self._product = product
        history_add(product)
        self._update_fav_btn(product.get("code", ""))

        name        = product.get("product_name") or "Unknown Product"
        brand       = product.get("brands", "Unknown Brand")
        img_url     = product.get("image_url", "")
        ingredients = product.get("ingredients_text", "") or ""
        nutriments  = product.get("nutriments", {})
        grade       = (product.get("nutriscore_grade") or "").lower()
        quantity    = product.get("quantity", "")

        self.hdr_title.text = name[:24] + ("…" if len(name) > 24 else "")

        # ── Run smart analysis ────────────────────────────────────────────────
        insights = build_nutrient_insights(nutriments)
        recs     = generate_recommendations(product)
        verdict  = smart_overall_verdict(product, insights, recs)

        # ── Section 1: Verdict banner (full-width, prominent) ─────────────────
        self.content.add_widget(VerdictBanner(verdict))

        # ── Section 2: Hero card (image + name + Nutri-Score) ─────────────────
        hero = Card(size_hint_y=None, height=dp(148),
                    orientation="horizontal", spacing=dp(12))
        if img_url:
            hero.add_widget(AsyncImage(source=img_url, size_hint_x=None,
                                       width=dp(110), allow_stretch=True,
                                       keep_ratio=True))
        else:
            hero.add_widget(Label(text="🍽", font_size=sp(44),
                                  size_hint_x=None, width=dp(90)))

        meta = BoxLayout(orientation="vertical", spacing=dp(4))
        meta.add_widget(Label(
            text=name, font_size=sp(13), bold=True,
            color=get_color_from_hex(C["dark"] + "FF"),
            halign="left", valign="top",
            text_size=(dp(195), None), size_hint_y=None, height=dp(44),
        ))
        meta.add_widget(Label(
            text=f"🏷  {brand}" + (f"  ·  {quantity}" if quantity else ""),
            font_size=sp(10), color=get_color_from_hex(C["sub"] + "FF"),
            halign="left", size_hint_y=None, height=dp(18),
        ))
        if grade in ("a", "b", "c", "d", "e"):
            ns_row = BoxLayout(size_hint_y=None, height=dp(40), spacing=dp(8))
            ns_row.add_widget(Label(
                text="Nutri-Score:", font_size=sp(11),
                color=get_color_from_hex(C["sub"] + "FF"),
                size_hint_x=None, width=dp(80),
            ))
            ns_row.add_widget(NutriScoreBadge(grade, size_hint_x=None,
                                              size=(dp(38), dp(38))))
            ns_row.add_widget(BoxLayout())
            meta.add_widget(ns_row)
        hero.add_widget(meta)
        self.content.add_widget(hero)

        # ── Section 3: Calorie highlight ──────────────────────────────────────
        kcal = energy_kcal_per100(nutriments)
        cal_card = Card(size_hint_y=None, height=dp(62),
                        orientation="horizontal", spacing=dp(8))
        cal_card.add_widget(Label(text="🔥", font_size=sp(26),
                                  size_hint_x=None, width=dp(44)))
        ci = BoxLayout(orientation="vertical")
        ci.add_widget(Label(
            text=f"{kcal:.0f} kcal", font_size=sp(20), bold=True,
            color=get_color_from_hex(C["primary"] + "FF"), halign="left",
        ))
        ci.add_widget(Label(
            text="per 100 g / 100 ml", font_size=sp(10),
            color=get_color_from_hex(C["sub"] + "FF"), halign="left",
        ))
        cal_card.add_widget(ci)
        self.content.add_widget(cal_card)

        # ── Section 4: SMART RECOMMENDATIONS (the new centrepiece) ────────────
        if recs:
            # Sort: danger first, then warning, then positive
            order = {"danger": 0, "warning": 1, "positive": 2}
            recs_sorted = sorted(recs, key=lambda r: order.get(r["type"], 3))

            rec_card = Card(size_hint_y=None,
                            height=dp(38 + len(recs_sorted) * 48))
            rec_card.add_widget(SectionTitle("💡  Smart Recommendations"))
            for r in recs_sorted:
                rec_card.add_widget(
                    RecommendationRow(r["text"], r["type"], r["icon"])
                )
            self.content.add_widget(rec_card)

        # ── Section 5: Dietary insights (per-nutrient badges) ─────────────────
        if insights:
            ins_card = Card(size_hint_y=None,
                            height=dp(36 + len(insights) * 42))
            ins_card.add_widget(SectionTitle("📊  Dietary Insights"))
            for ins in insights:
                ins_card.add_widget(InsightBadge(
                    label=f"{ins['label']}  ({ins['value']})",
                    rating=ins["rating"],
                ))
            self.content.add_widget(ins_card)

        # ── Section 6: Nutrition facts table ──────────────────────────────────
        nutrient_keys = [
            ("fat_100g",           "Total Fat",     "fat"),
            ("saturated-fat_100g", "Saturated Fat", "saturated-fat"),
            ("carbohydrates_100g", "Carbohydrates", "neutral"),
            ("sugars_100g",        "Sugars",        "sugars"),
            ("fiber_100g",         "Dietary Fiber", "fiber"),
            ("proteins_100g",      "Protein",       "proteins"),
            ("salt_100g",          "Salt",          "salt"),
            ("sodium_100g",        "Sodium",        "sodium"),
        ]
        rows = []
        for key, label, rkey in nutrient_keys:
            val = nutriments.get(key)
            if val is not None:
                try:
                    fv = float(val)
                    rows.append((label, f"{fv:.2f} g", rate_nutrient(rkey, fv)))
                except (ValueError, TypeError):
                    pass
        if rows:
            tbl = Card(size_hint_y=None, height=dp(36 + len(rows) * 36))
            tbl.add_widget(SectionTitle("🧪  Nutrition Facts (per 100 g)"))
            for label, val, rating in rows:
                tbl.add_widget(NutrientRow(label, val, rating))
            self.content.add_widget(tbl)

        # ── Section 7: Harmful additives ──────────────────────────────────────
        additives = detect_additives(product)
        if additives:
            add_card = Card(size_hint_y=None,
                            height=dp(36 + len(additives) * 38))
            add_card.add_widget(SectionTitle("🚫  Harmful Additives Detected"))
            for a in additives:
                add_card.add_widget(AdditiveBadge(a["name"], a["severity"]))
            self.content.add_widget(add_card)
        else:
            ok = Card(size_hint_y=None, height=dp(50))
            ok.add_widget(Label(
                text="✅  No harmful additives detected",
                font_size=sp(12), bold=True,
                color=get_color_from_hex(C["safe"] + "FF"), halign="left",
            ))
            self.content.add_widget(ok)

        # ── Section 7b: Ingredient keyword concerns ───────────────────────────
        concerns = analyze_ingredients(product)
        if concerns:
            con_card = Card(size_hint_y=None,
                            height=dp(36 + len(concerns) * 38))
            con_card.add_widget(SectionTitle("🔬  Ingredient Concerns"))
            for c in concerns:
                con_card.add_widget(AdditiveBadge(c["label"], c["severity"]))
            self.content.add_widget(con_card)

        # ── Section 8: Allergy alerts ─────────────────────────────────────────
        allergens = detect_allergens(product)
        if allergens:
            chips_w = _wrap_chips([(a["icon"], a["label"])
                                   for a in allergens], AllergenChip)
            alg = Card(size_hint_y=None, height=dp(46) + chips_w.height)
            alg.add_widget(SectionTitle("⚠️  Allergy Alerts"))
            alg.add_widget(chips_w)
            self.content.add_widget(alg)
        else:
            no_alg = Card(size_hint_y=None, height=dp(50))
            no_alg.add_widget(Label(
                text="✅  No common allergens detected",
                font_size=sp(12), bold=True,
                color=get_color_from_hex(C["safe"] + "FF"), halign="left",
            ))
            self.content.add_widget(no_alg)

        # ── Section 9: Diet compatibility ─────────────────────────────────────
        diets      = check_diet_compatibility(product)
        diet_chips = _wrap_chips(
            [(d["icon"], d["diet"], d["compatible"]) for d in diets], DietChip)
        diet_card = Card(size_hint_y=None, height=dp(46) + diet_chips.height)
        diet_card.add_widget(SectionTitle("🥗  Diet Compatibility"))
        diet_card.add_widget(diet_chips)
        self.content.add_widget(diet_card)

        # ── Section 10: Ingredients ───────────────────────────────────────────
        if ingredients:
            ing_card = Card(size_hint_y=None, height=dp(110))
            ing_card.add_widget(SectionTitle("📋  Ingredients"))
            ing_lbl = Label(
                text=ingredients[:440] + ("…" if len(ingredients) > 440 else ""),
                font_size=sp(10),
                color=get_color_from_hex(C["sub"] + "FF"),
                halign="left", valign="top",
            )
            ing_lbl.bind(size=lambda w, s: setattr(w, "text_size", s))
            ing_card.add_widget(ing_lbl)
            self.content.add_widget(ing_card)

    def _go_back(self, *_):
        self.manager.transition = SlideTransition(direction="right")
        self.manager.current = getattr(self, "_prev", "home")


class QrScanScreen(Screen):
    """
    Desktop-oriented QR scanner: OpenCV capture blitted into a Kivy Image.
    Switch VideoCapture index for another physical/virtual camera; optional
    image file decode. Avoids separate cv2.imshow window.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        lay = BoxLayout(orientation="vertical", padding=dp(12), spacing=dp(6))

        hdr = BoxLayout(size_hint_y=None, height=dp(44), spacing=dp(8))
        back = RoundedButton(text="Back", size_hint_x=None, width=dp(72),
                             bg_color=C["dark"], radius=20, font_size=sp(12))
        back.bind(on_release=self._go_back)
        hdr.add_widget(back)
        hdr.add_widget(Label(
            text="QR scanner", font_size=sp(16), bold=True,
            color=get_color_from_hex(C["dark"] + "FF"),
        ))
        lay.add_widget(hdr)

        self.cam_info = Label(
            text="", font_size=sp(10),
            color=get_color_from_hex(C["sub"] + "FF"),
            size_hint_y=None, height=dp(18),
        )
        lay.add_widget(self.cam_info)

        self.preview = KivyImage(
            allow_stretch=True, keep_ratio=True, size_hint_y=1,
        )
        lay.add_widget(self.preview)

        self.status = Label(
            text="",
            font_size=sp(11), color=get_color_from_hex(C["sub"] + "FF"),
            size_hint_y=None, height=dp(52), valign="top",
        )
        self.status.bind(
            size=lambda w, s: setattr(w, "text_size", (s[0] - dp(4), None))
        )
        lay.add_widget(self.status)

        row1 = BoxLayout(size_hint_y=None, height=dp(44), spacing=dp(6))
        self._switch_btn = RoundedButton(
            text="Switch camera", bg_color=C["blue"], font_size=sp(11),
        )
        self._switch_btn.bind(on_release=self._switch_camera)
        photo_btn = RoundedButton(
            text="Photo / file", bg_color=C["purple"], font_size=sp(11),
        )
        photo_btn.bind(on_release=self._pick_image)
        reset_btn = RoundedButton(
            text="New QR", bg_color=C["teal"], font_size=sp(11),
        )
        reset_btn.bind(on_release=self._reset_lock)
        row1.add_widget(self._switch_btn)
        row1.add_widget(photo_btn)
        row1.add_widget(reset_btn)
        lay.add_widget(row1)

        lay.add_widget(Label(
            text="Any QR · http(s) → browser · digits → barcode · else name search",
            font_size=sp(9),
            color=get_color_from_hex(C["sub"] + "AA"),
            size_hint_y=None, height=dp(28), halign="center",
        ))

        self.add_widget(lay)
        self._cap = None
        self._frame_ev = None
        self._frame_i = 0
        self._camera_indices = [0]
        self._camera_slot = 0
        self._last_handled_payload: str | None = None

    def _reset_lock(self, *_):
        self._last_handled_payload = None
        self.status.text = "Ready for the next QR code."

    def on_enter(self):
        self._last_handled_payload = None
        if not QR_SCAN_AVAILABLE:
            self.cam_info.text = ""
            self.status.text = "Install OpenCV:  pip install opencv-python"
            return
        self.status.text = "Starting camera…"
        Clock.schedule_once(lambda _dt: self._open_after_probe(), 0)

    def _open_after_probe(self):
        self._camera_indices = probe_video_capture_indices()
        self._camera_slot = 0
        self._start_camera_session()

    def _start_camera_session(self):
        self._stop_preview_only()
        if not self._camera_indices:
            self._camera_indices = [0]
        dev_i = self._camera_indices[self._camera_slot % len(self._camera_indices)]
        self._cap = open_video_capture(dev_i)
        if not self._cap or not self._cap.isOpened():
            self.status.text = "Camera unavailable (busy, denied, or none)."
            self.cam_info.text = ""
            return
        try:
            self._cap.set(getattr(cv2, "CAP_PROP_FRAME_WIDTH", 3), 1280)
            self._cap.set(getattr(cv2, "CAP_PROP_FRAME_HEIGHT", 4), 720)
            self._cap.set(getattr(cv2, "CAP_PROP_AUTOFOCUS", 39), 1)
        except Exception:
            pass
        # Discard frames while exposure / autofocus settle.
        for _ in range(12):
            self._cap.read()
        n = len(self._camera_indices)
        self.cam_info.text = (
            f"Device index {dev_i}  ({self._camera_slot + 1}/{n})"
            if n > 1
            else f"Device index {dev_i}"
        )
        self.status.text = "Point the camera at a QR — hold steady & well lit."
        self._frame_i = 0
        self._frame_ev = Clock.schedule_interval(self._update_frame, 1 / 15)

    def _stop_preview_only(self):
        if self._frame_ev:
            self._frame_ev.cancel()
            self._frame_ev = None
        if self._cap:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None

    def on_leave(self):
        self._stop_preview_only()

    def _switch_camera(self, *_):
        if not QR_SCAN_AVAILABLE:
            return
        if len(self._camera_indices) < 2:
            self.status.text = "Only one camera index responded. Try another USB camera."
            return
        self._camera_slot = (self._camera_slot + 1) % len(self._camera_indices)
        self._last_handled_payload = None
        self.status.text = "Switching camera…"
        self._start_camera_session()

    def _update_frame(self, _dt):
        if not self._cap or not self._cap.isOpened():
            return
        from kivy.graphics.texture import Texture

        ok, frame = self._cap.read()
        if not ok or frame is None:
            return
        self._frame_i += 1
        # Alternate deep decode so the UI thread stays responsive.
        deep = self._frame_i % 2 == 0
        data = decode_qr_bgr(frame, deep=deep)
        if data and data != self._last_handled_payload:
            self._last_handled_payload = data
            Clock.schedule_once(lambda dt, d=data: self._dispatch(d), 0)
        elif not data and self._frame_i % 30 == 0 and self._last_handled_payload is None:
            self.status.text = "No QR seen yet — move closer, add light, or Switch camera."

        flipped = cv2.flip(frame, 0)
        buf = flipped.tobytes()
        h, w = frame.shape[:2]
        tex = Texture.create(size=(w, h), colorfmt="bgr")
        tex.blit_buffer(buf, colorfmt="bgr", bufferfmt="ubyte")
        self.preview.texture = tex

    def _dispatch(self, data: str):
        kind, value = classify_qr_payload(data)
        if kind == "empty":
            self._last_handled_payload = None
            return
        if kind == "url":
            self.status.text = "Opening link in browser…"
            try:
                webbrowser.open(value)
            except Exception:
                self.status.text = "Could not open URL."
                self._last_handled_payload = None
                return
            self.status.text = f"Opened: {value[:56]}…" if len(value) > 56 else f"Opened: {value}"
            return

        if kind == "barcode":
            self.on_leave()
            detail = self.manager.get_screen("detail")
            detail._prev = "qrscan"
            detail.load_barcode(value)
            self.manager.transition = SlideTransition(direction="left")
            self.manager.current = "detail"
            return

        self.status.text = f"Looking up “{value[:32]}…”"

        def _worker():
            try:
                found = search_by_name(value.strip(), page_size=15)
            except Exception:
                found = []
            Clock.schedule_once(
                lambda dt, q=value, plist=found: self._apply_name_search(q, plist),
                0,
            )

        threading.Thread(target=_worker, daemon=True).start()

    def _apply_name_search(self, query: str, products: list):
        if not products:
            self.status.text = (
                f'No foods for “{query[:34]}{"…" if len(query) > 34 else ""}”'
            )
            self._last_handled_payload = None
            return
        self.on_leave()
        if len(products) == 1:
            detail = self.manager.get_screen("detail")
            detail._prev = "qrscan"
            detail.load_product(products[0])
            self.manager.transition = SlideTransition(direction="left")
            self.manager.current = "detail"
        else:
            rs = self.manager.get_screen("results")
            rs._prev = "qrscan"
            rs.load_search(query)
            self.manager.transition = SlideTransition(direction="left")
            self.manager.current = "results"

    def _pick_image(self, *_):
        if not QR_SCAN_AVAILABLE:
            self.status.text = "Install OpenCV first: pip install opencv-python"
            return
        self.status.text = "Choose an image…"
        try:
            if kivy_platform == "macosx":
                self._pick_macos_native()
                return
            self._pick_filechooser_popup()
        except Exception as e:
            self.status.text = f"Picker error: {e}"

    def _pick_macos_native(self):
        def work():
            path = None
            try:
                proc = subprocess.run(
                    [
                        "osascript", "-e",
                        'POSIX path of (choose file with prompt '
                        '"Select an image containing a QR code")',
                    ],
                    capture_output=True,
                    text=True,
                    timeout=300,
                )
                if proc.returncode == 0 and proc.stdout.strip():
                    cand = proc.stdout.strip()
                    if os.path.isfile(cand):
                        path = cand
            except Exception:
                path = None

            def apply_path(_dt):
                if path:
                    self._decode_file(path)
                else:
                    self._pick_filechooser_popup()

            Clock.schedule_once(apply_path, 0)

        threading.Thread(target=work, daemon=True).start()

    def _pick_filechooser_popup(self):
        root = os.path.expanduser("~")
        fc = FileChooserListView(
            path=root,
            filters=["*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp"],
        )
        content = BoxLayout(orientation="vertical", spacing=dp(6), padding=dp(8))
        content.add_widget(fc)
        btn_row = BoxLayout(size_hint_y=None, height=dp(44), spacing=dp(8))
        pop = Popup(title="Select image", size_hint=(0.94, 0.88))

        def use_sel(*_a):
            if fc.selection:
                pop.dismiss()
                self._decode_file(fc.selection[0])
            else:
                self.status.text = "Select a file, then Use image."

        def cancel(*_a):
            pop.dismiss()

        cancel_btn = Button(text="Cancel", size_hint_x=0.5)
        ok_btn = Button(text="Use image", size_hint_x=0.5)
        cancel_btn.bind(on_release=cancel)
        ok_btn.bind(on_release=use_sel)
        btn_row.add_widget(cancel_btn)
        btn_row.add_widget(ok_btn)
        content.add_widget(btn_row)
        pop.content = content
        pop.open()

    def _decode_file(self, path: str):
        if not QR_SCAN_AVAILABLE:
            self.status.text = "Install OpenCV: pip install opencv-python"
            return
        img = load_image_bgr(path)
        if img is None:
            self.status.text = (
                "Could not read that file. Use PNG or JPG, or pick another path."
            )
            self._last_handled_payload = None
            return
        data = decode_qr_bgr(img, deep=True)
        if not data:
            if not pyzbar_usable():
                tip = (
                    " ZBar library missing: Mac run `brew install zbar`, "
                    "then restart the app. Also: `pip install pyzbar`. "
                )
            else:
                tip = " "
            self.status.text = (
                "No QR decoded in this image."
                + tip
                + "Use a sharp, bright photo; QR should be large in the frame."
            )
            self._last_handled_payload = None
            return
        self._last_handled_payload = data
        self._dispatch(data)

    def _go_back(self, *_):
        self.manager.transition = SlideTransition(direction="right")
        self.manager.current = "home"


class HistoryScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._mode = "history"
        lay = BoxLayout(orientation="vertical", padding=dp(14), spacing=dp(10))

        hdr = BoxLayout(size_hint_y=None, height=dp(48), spacing=dp(10))
        back = RoundedButton(text="← Back", size_hint_x=None, width=dp(76),
                             bg_color=C["dark"], radius=20)
        back.bind(on_release=self._go_back)
        self.title_lbl = Label(
            text="History", font_size=sp(17), bold=True,
            color=get_color_from_hex(C["dark"] + "FF"),
        )
        clear_btn = RoundedButton(text="🗑 Clear", size_hint_x=None,
                                  width=dp(72), bg_color=C["danger"],
                                  radius=20, font_size=sp(11))
        clear_btn.bind(on_release=self._clear)
        hdr.add_widget(back)
        hdr.add_widget(self.title_lbl)
        hdr.add_widget(clear_btn)
        lay.add_widget(hdr)

        sv = ScrollView()
        self.box = BoxLayout(orientation="vertical",
                             spacing=dp(8), size_hint_y=None)
        self.box.bind(minimum_height=self.box.setter("height"))
        sv.add_widget(self.box)
        lay.add_widget(sv)
        self.add_widget(lay)

    def load(self, mode: str = "history"):
        self._mode = mode
        self.title_lbl.text = ("⭐ Favourites" if mode == "favourites"
                               else "🕘 Scan History")
        self.box.clear_widgets()
        entries = (favourites_get() if mode == "favourites"
                   else history_get())
        if not entries:
            self.box.add_widget(Label(
                text=("Nothing here yet."
                      if mode == "favourites"
                      else "No scans yet. Scan a product to get started."),
                font_size=sp(13), color=get_color_from_hex(C["sub"] + "FF"),
                size_hint_y=None, height=dp(60), halign="center",
            ))
            return
        for entry in entries:
            self.box.add_widget(self._make_row(entry))

    def _make_row(self, entry: dict):
        card = Card(size_hint_y=None, height=dp(76),
                    orientation="horizontal", spacing=dp(10))
        img_url = entry.get("image_url", "")
        if img_url:
            card.add_widget(AsyncImage(source=img_url, size_hint_x=None,
                                       width=dp(60), allow_stretch=True,
                                       keep_ratio=True))
        else:
            card.add_widget(Label(text="🍽", font_size=sp(24),
                                  size_hint_x=None, width=dp(50)))
        info = BoxLayout(orientation="vertical", spacing=dp(2))
        info.add_widget(Label(
            text=(entry.get("name") or "Unknown")[:36],
            font_size=sp(12), bold=True,
            color=get_color_from_hex(C["dark"] + "FF"),
            halign="left", size_hint_y=None, height=dp(20),
        ))
        meta_parts = []
        if entry.get("brand"):
            meta_parts.append(entry["brand"][:20])
        if entry.get("calories"):
            meta_parts.append(f"{entry['calories']:.0f} kcal")
        if entry.get("scanned_at"):
            meta_parts.append(entry["scanned_at"])
        info.add_widget(Label(
            text="  ·  ".join(meta_parts),
            font_size=sp(10), color=get_color_from_hex(C["sub"] + "FF"),
            halign="left", size_hint_y=None, height=dp(16),
        ))
        view_btn = RoundedButton(text="View →", size_hint_y=None,
                                 height=dp(26), bg_color=C["blue"], radius=8,
                                 font_size=sp(11))
        view_btn.barcode = entry.get("barcode", "")
        view_btn.bind(on_release=self._open_entry)
        info.add_widget(view_btn)
        card.add_widget(info)
        return card

    def _open_entry(self, btn):
        if not btn.barcode:
            return
        detail = self.manager.get_screen("detail")
        detail._prev = "history"
        detail.load_barcode(btn.barcode)
        self.manager.transition = SlideTransition(direction="left")
        self.manager.current = "detail"

    def _clear(self, *_):
        data = _load_data()
        if self._mode == "favourites":
            data["favourites"] = []
        else:
            data["history"] = []
        _save_data(data)
        self.load(self._mode)

    def _go_back(self, *_):
        self.manager.transition = SlideTransition(direction="right")
        self.manager.current = "home"


# ══════════════════════════════════════════════════════════════════════════════
# App
# ══════════════════════════════════════════════════════════════════════════════

class FoodScanApp(App):
    title = "FoodScan"

    def build(self):
        sm = ScreenManager()
        sm.add_widget(HomeScreen(name="home"))
        sm.add_widget(ResultsScreen(name="results"))
        sm.add_widget(DetailScreen(name="detail"))
        sm.add_widget(QrScanScreen(name="qrscan"))
        sm.add_widget(HistoryScreen(name="history"))
        return sm


if __name__ == "__main__":
    FoodScanApp().run()
