"""
analyzer.py
===========
All data-processing and classification logic.

Provides:
  - rate_nutrient()          → "safe" | "moderate" | "avoid"
  - build_nutrient_report()  → pandas DataFrame of nutrient ratings
  - detect_additives()       → list of harmful additive dicts
  - detect_allergens()       → list of allergen dicts
  - check_diet_compatibility()→ list of diet compatibility dicts
  - classify_product()       → full AnalysisReport dataclass

Uses pandas for tabular nutrient processing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

import pandas as pd

from foodscan.api_handler import ProductData

logger = logging.getLogger(__name__)

# ── types ─────────────────────────────────────────────────────────────────────

Rating = Literal["safe", "moderate", "avoid", "neutral"]
Verdict = Literal["safe", "moderate", "avoid", "unknown"]


# ══════════════════════════════════════════════════════════════════════════════
# Nutrient rating
# ══════════════════════════════════════════════════════════════════════════════

# (low_threshold, high_threshold) per 100 g
# Values below low  → safe
# Values above high → avoid
# In between        → moderate
# Keys marked with * are "higher is better" (inverted logic)
_THRESHOLDS: dict[str, tuple[float, float]] = {
    "sugars":        (5.0,  22.5),
    "saturated-fat": (1.5,   5.0),
    "salt":          (0.3,   1.5),
    "sodium":        (0.1,   0.6),
    "fat":           (3.0,  17.5),
    "energy-kcal":   (150,  400),
    "fiber":         (3.0,   6.0),   # inverted
    "proteins":      (5.0,  10.0),   # inverted
}
_INVERT = {"fiber", "proteins"}

# Human-readable labels for each nutrient key
_NUTRIENT_LABELS: dict[str, str] = {
    "sugars":        "Sugars",
    "saturated-fat": "Saturated Fat",
    "salt":          "Salt",
    "sodium":        "Sodium",
    "fat":           "Total Fat",
    "energy-kcal":   "Calories (kcal)",
    "fiber":         "Dietary Fiber",
    "proteins":      "Protein",
    "carbohydrates": "Carbohydrates",
}

# Nutriment keys to pull from ProductData.nutriments (key → display label, rating_key)
_NUTRIMENT_MAP: list[tuple[str, str, str]] = [
    ("fat_100g",           "Total Fat",        "fat"),
    ("saturated-fat_100g", "Saturated Fat",    "saturated-fat"),
    ("carbohydrates_100g", "Carbohydrates",    "carbohydrates"),
    ("sugars_100g",        "Sugars",           "sugars"),
    ("fiber_100g",         "Dietary Fiber",    "fiber"),
    ("proteins_100g",      "Protein",          "proteins"),
    ("salt_100g",          "Salt",             "salt"),
    ("sodium_100g",        "Sodium",           "sodium"),
    ("energy-kcal_100g",   "Calories (kcal)",  "energy-kcal"),
]


def rate_nutrient(key: str, value_per_100g: float) -> Rating:
    """
    Classify a single nutrient value.

    Args:
        key:            Nutrient key (e.g. "sugars", "fiber").
        value_per_100g: Amount per 100 g / 100 ml.

    Returns:
        "safe" | "moderate" | "avoid" | "neutral"
    """
    k = key.lower().replace(" ", "-")
    if k not in _THRESHOLDS:
        return "neutral"
    lo, hi = _THRESHOLDS[k]
    if k in _INVERT:
        return "safe" if value_per_100g >= hi else (
               "moderate" if value_per_100g >= lo else "avoid")
    return "safe" if value_per_100g <= lo else (
           "avoid" if value_per_100g >= hi else "moderate")


def build_nutrient_report(product: ProductData) -> pd.DataFrame:
    """
    Build a pandas DataFrame with one row per nutrient.

    Columns: nutrient | value_per_100g | unit | rating | rating_icon
    """
    rows = []
    for api_key, label, rating_key in _NUTRIMENT_MAP:
        raw = product.nutriments.get(api_key)
        if raw is None:
            continue
        try:
            val = float(raw)
        except (ValueError, TypeError):
            continue
        rating = rate_nutrient(rating_key, val)
        icon   = {"safe": "✅", "moderate": "⚠️", "avoid": "🚫",
                  "neutral": "ℹ️"}.get(rating, "ℹ️")
        unit   = "kcal" if "kcal" in api_key else "g"
        rows.append({
            "nutrient":       label,
            "value_per_100g": round(val, 2),
            "unit":           unit,
            "rating":         rating,
            "icon":           icon,
        })

    df = pd.DataFrame(rows, columns=["nutrient", "value_per_100g",
                                     "unit", "rating", "icon"])
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Harmful additive detection
# ══════════════════════════════════════════════════════════════════════════════

_HARMFUL_ADDITIVES: dict[str, tuple[str, Rating]] = {
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


def detect_additives(product: ProductData) -> list[dict]:
    """
    Scan a product for harmful additives.

    Returns:
        List of dicts: {code, name, severity, icon}
    """
    found: dict[str, dict] = {}

    # Primary: additives_tags from API  (e.g. "en:e211")
    for tag in product.additives_tags:
        code = tag.split(":")[-1].lower().replace("-", "")
        if code in _HARMFUL_ADDITIVES and code not in found:
            name, sev = _HARMFUL_ADDITIVES[code]
            found[code] = {
                "code":     code.upper(),
                "name":     name,
                "severity": sev,
                "icon":     "🚫" if sev == "avoid" else "⚠️",
            }

    # Fallback: scan raw ingredients text
    raw = product.ingredients_text.upper()
    for code, (name, sev) in _HARMFUL_ADDITIVES.items():
        e_num = code.upper()
        if e_num in raw and code not in found:
            found[code] = {
                "code":     e_num,
                "name":     name,
                "severity": sev,
                "icon":     "🚫" if sev == "avoid" else "⚠️",
            }

    result = list(found.values())
    logger.debug("Additives found: %d", len(result))
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Allergen detection
# ══════════════════════════════════════════════════════════════════════════════

_ALLERGEN_MAP: dict[str, tuple[str, str]] = {
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


def detect_allergens(product: ProductData) -> list[dict]:
    """
    Detect common allergens in a product.

    Returns:
        List of dicts: {key, icon, label}
    """
    tags_text = " ".join(product.allergens_tags).lower()
    ingr_text = product.ingredients_text.lower()
    combined  = tags_text + " " + ingr_text

    found = []
    seen  = set()
    for key, (icon, label) in _ALLERGEN_MAP.items():
        if key in combined and label not in seen:
            found.append({"key": key, "icon": icon, "label": label})
            seen.add(label)

    logger.debug("Allergens found: %d", len(found))
    return found


# ══════════════════════════════════════════════════════════════════════════════
# Diet compatibility
# ══════════════════════════════════════════════════════════════════════════════

_NON_VEGAN    = ["milk", "egg", "honey", "gelatin", "meat", "fish",
                 "chicken", "beef", "pork", "lard", "whey", "casein",
                 "lactose", "butter", "cream", "anchovy"]
_NON_HALAL    = ["pork", "lard", "gelatin", "alcohol", "wine", "beer",
                 "rum", "whisky", "vodka", "bacon", "ham"]
_GLUTEN_WORDS = ["wheat", "gluten", "barley", "rye", "spelt", "kamut",
                 "triticale", "malt"]


def check_diet_compatibility(product: ProductData) -> list[dict]:
    """
    Evaluate diet compatibility for common dietary patterns.

    Returns:
        List of dicts: {diet, icon, compatible (True/False/None), note}
        compatible=None means "unknown / insufficient data"
    """
    labels   = " ".join(product.labels_tags).lower()
    cats     = " ".join(product.categories_tags).lower()
    ingr     = product.ingredients_text.lower()
    combined = labels + " " + cats + " " + ingr

    results = []

    # ── Vegan ─────────────────────────────────────────────────────────────────
    if "vegan" in labels:
        vegan_ok, vegan_note = True, "Labelled vegan"
    elif any(w in combined for w in _NON_VEGAN):
        vegan_ok, vegan_note = False, "Contains animal-derived ingredients"
    else:
        vegan_ok, vegan_note = None, "No animal ingredients detected (unverified)"
    results.append({"diet": "Vegan", "icon": "🌱",
                    "compatible": vegan_ok, "note": vegan_note})

    # ── Vegetarian ────────────────────────────────────────────────────────────
    non_veg = ["meat", "chicken", "beef", "pork", "fish", "seafood",
               "gelatin", "lard", "anchovy", "bacon", "ham"]
    if "vegetarian" in labels:
        veg_ok, veg_note = True, "Labelled vegetarian"
    elif any(w in combined for w in non_veg):
        veg_ok, veg_note = False, "Contains meat/fish"
    else:
        veg_ok, veg_note = None, "No meat detected (unverified)"
    results.append({"diet": "Vegetarian", "icon": "🥦",
                    "compatible": veg_ok, "note": veg_note})

    # ── Keto (net carbs ≤ 5 g / 100 g) ───────────────────────────────────────
    carbs = product.carbs_100g
    fiber = product.fiber_100g
    net   = carbs - fiber
    if carbs > 0:
        keto_ok   = net <= 5
        keto_note = f"Net carbs: {net:.1f} g / 100 g"
    else:
        keto_ok, keto_note = None, "Carbohydrate data unavailable"
    results.append({"diet": "Keto", "icon": "🥑",
                    "compatible": keto_ok, "note": keto_note})

    # ── Diabetic-Friendly (sugar ≤ 5 g / 100 g) ──────────────────────────────
    sugar = product.sugars_100g
    if sugar > 0:
        diab_ok   = sugar <= 5
        diab_note = f"Sugar: {sugar:.1f} g / 100 g"
    else:
        diab_ok, diab_note = None, "Sugar data unavailable"
    results.append({"diet": "Diabetic-Friendly", "icon": "💉",
                    "compatible": diab_ok, "note": diab_note})

    # ── Halal ─────────────────────────────────────────────────────────────────
    if "halal" in labels:
        halal_ok, halal_note = True, "Labelled halal"
    elif any(w in combined for w in _NON_HALAL):
        halal_ok, halal_note = False, "Contains non-halal ingredients"
    else:
        halal_ok, halal_note = None, "No non-halal ingredients detected (unverified)"
    results.append({"diet": "Halal", "icon": "☪️",
                    "compatible": halal_ok, "note": halal_note})

    # ── Gluten-Free ───────────────────────────────────────────────────────────
    if "gluten-free" in labels or "sans-gluten" in labels:
        gf_ok, gf_note = True, "Labelled gluten-free"
    elif any(w in combined for w in _GLUTEN_WORDS):
        gf_ok, gf_note = False, "Contains gluten sources"
    else:
        gf_ok, gf_note = None, "No gluten detected (unverified)"
    results.append({"diet": "Gluten-Free", "icon": "🌾",
                    "compatible": gf_ok, "note": gf_note})

    # ── Low-Sodium (≤ 120 mg sodium / 100 g) ─────────────────────────────────
    sodium = product.sodium_100g
    if sodium > 0:
        ls_ok   = sodium <= 0.12
        ls_note = f"Sodium: {sodium * 1000:.0f} mg / 100 g"
    else:
        ls_ok, ls_note = None, "Sodium data unavailable"
    results.append({"diet": "Low-Sodium", "icon": "🧂",
                    "compatible": ls_ok, "note": ls_note})

    return results


# ══════════════════════════════════════════════════════════════════════════════
# Full analysis report
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class AnalysisReport:
    """Complete analysis result for one product."""
    product:       ProductData
    nutrient_df:   pd.DataFrame
    overall:       Verdict
    additives:     list[dict]      = field(default_factory=list)
    allergens:     list[dict]      = field(default_factory=list)
    diets:         list[dict]      = field(default_factory=list)
    warnings:      list[str]       = field(default_factory=list)

    # ── summary helpers ───────────────────────────────────────────────────────

    @property
    def has_harmful_additives(self) -> bool:
        return any(a["severity"] == "avoid" for a in self.additives)

    @property
    def allergen_labels(self) -> list[str]:
        return [a["label"] for a in self.allergens]

    @property
    def verdict_text(self) -> str:
        return {
            "safe":     "✅ Generally Safe to Consume",
            "moderate": "⚠️  Consume in Moderation",
            "avoid":    "🚫 Avoid – High Risk Ingredients",
            "unknown":  "ℹ️  Insufficient Data",
        }.get(self.overall, "ℹ️  Unknown")

    @property
    def verdict_color(self) -> str:
        """Hex colour for the verdict (used by Streamlit / HTML)."""
        return {
            "safe":     "#27AE60",
            "moderate": "#F39C12",
            "avoid":    "#E74C3C",
            "unknown":  "#7F8C8D",
        }.get(self.overall, "#7F8C8D")

    def to_dict(self) -> dict:
        return {
            "product":   self.product.to_dict(),
            "overall":   self.overall,
            "verdict":   self.verdict_text,
            "nutrients": self.nutrient_df.to_dict(orient="records"),
            "additives": self.additives,
            "allergens": self.allergens,
            "diets":     self.diets,
            "warnings":  self.warnings,
        }


def _compute_overall(df: pd.DataFrame,
                     additives: list[dict]) -> Verdict:
    """Derive an overall verdict from nutrient ratings + additive severity."""
    if df.empty:
        return "unknown"

    counts = df["rating"].value_counts().to_dict()
    avoid_count    = counts.get("avoid",    0)
    moderate_count = counts.get("moderate", 0)
    has_bad_add    = any(a["severity"] == "avoid" for a in additives)

    if avoid_count >= 2 or has_bad_add:
        return "avoid"
    if avoid_count >= 1 or moderate_count >= 3:
        return "moderate"
    if moderate_count >= 1:
        return "moderate"
    return "safe"


def classify_product(product: ProductData) -> AnalysisReport:
    """
    Run the full analysis pipeline on a ProductData object.

    Args:
        product: A ProductData instance from api_handler.

    Returns:
        AnalysisReport with all analysis results populated.
    """
    nutrient_df = build_nutrient_report(product)
    additives   = detect_additives(product)
    allergens   = detect_allergens(product)
    diets       = check_diet_compatibility(product)
    overall     = _compute_overall(nutrient_df, additives)

    # Build human-readable warnings
    warnings: list[str] = []
    avoid_nutrients = nutrient_df[nutrient_df["rating"] == "avoid"]["nutrient"].tolist()
    for n in avoid_nutrients:
        warnings.append(f"High {n} content")
    for a in additives:
        if a["severity"] == "avoid":
            warnings.append(f"Contains {a['name']}")
    if allergens:
        labels = ", ".join(a["label"] for a in allergens)
        warnings.append(f"Allergens present: {labels}")

    report = AnalysisReport(
        product=product,
        nutrient_df=nutrient_df,
        overall=overall,
        additives=additives,
        allergens=allergens,
        diets=diets,
        warnings=warnings,
    )
    logger.info("Analysis complete for '%s' → %s", product.name, overall)
    return report
