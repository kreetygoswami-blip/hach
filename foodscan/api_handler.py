"""
api_handler.py
==============
All outbound HTTP calls to the Open Food Facts API.

Provides:
  - fetch_product_by_barcode(barcode)  → dict | None
  - search_products(query, page_size)  → list[dict]
  - ProductData  dataclass (clean, typed product record)

Uses a requests.Session with retry logic and a shared timeout.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

# ── constants ─────────────────────────────────────────────────────────────────

_BASE          = "https://world.openfoodfacts.org"
_BARCODE_URL   = _BASE + "/api/v0/product/{barcode}.json"
_SEARCH_URL    = _BASE + "/cgi/search.pl"
_TIMEOUT       = 12          # seconds
_FIELDS        = ",".join([
    "product_name", "brands", "image_url", "code",
    "nutriments", "ingredients_text",
    "allergens_tags", "labels_tags",
    "categories_tags", "additives_tags",
    "quantity", "serving_size",
])

# ── shared session with retry ─────────────────────────────────────────────────

def _build_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://",  adapter)
    session.headers.update({"User-Agent": "FoodScan/2.0 (educational project)"})
    return session


_SESSION = _build_session()


# ── typed product record ──────────────────────────────────────────────────────

@dataclass
class ProductData:
    """
    Clean, typed representation of a single food product.
    All fields default to safe empty values so callers never get KeyErrors.
    """
    barcode:          str            = ""
    name:             str            = "Unknown Product"
    brand:            str            = ""
    quantity:         str            = ""
    serving_size:     str            = ""
    image_url:        str            = ""
    ingredients_text: str            = ""
    nutriments:       dict           = field(default_factory=dict)
    allergens_tags:   list[str]      = field(default_factory=list)
    labels_tags:      list[str]      = field(default_factory=list)
    categories_tags:  list[str]      = field(default_factory=list)
    additives_tags:   list[str]      = field(default_factory=list)

    # ── convenience nutrient accessors ────────────────────────────────────────

    def _n(self, *keys, default=0.0) -> float:
        """Try multiple nutriment keys, return first float found."""
        for k in keys:
            v = self.nutriments.get(k)
            if v is not None:
                try:
                    return float(v)
                except (ValueError, TypeError):
                    pass
        return default

    @property
    def calories_100g(self) -> float:
        kcal = self._n("energy-kcal_100g", "energy-kcal")
        if kcal:
            return kcal
        kj = self._n("energy_100g", "energy")
        return (kj / 4.184) if kj else 0.0

    @property
    def fat_100g(self) -> float:
        return self._n("fat_100g", "fat")

    @property
    def saturated_fat_100g(self) -> float:
        return self._n("saturated-fat_100g", "saturated-fat")

    @property
    def carbs_100g(self) -> float:
        return self._n("carbohydrates_100g", "carbohydrates")

    @property
    def sugars_100g(self) -> float:
        return self._n("sugars_100g", "sugars")

    @property
    def fiber_100g(self) -> float:
        return self._n("fiber_100g", "fiber")

    @property
    def protein_100g(self) -> float:
        return self._n("proteins_100g", "proteins")

    @property
    def salt_100g(self) -> float:
        return self._n("salt_100g", "salt")

    @property
    def sodium_100g(self) -> float:
        return self._n("sodium_100g", "sodium")

    def to_dict(self) -> dict:
        return {
            "barcode":          self.barcode,
            "name":             self.name,
            "brand":            self.brand,
            "quantity":         self.quantity,
            "serving_size":     self.serving_size,
            "image_url":        self.image_url,
            "ingredients_text": self.ingredients_text,
            "nutriments":       self.nutriments,
            "allergens_tags":   self.allergens_tags,
            "labels_tags":      self.labels_tags,
            "categories_tags":  self.categories_tags,
            "additives_tags":   self.additives_tags,
        }


# ── internal parser ───────────────────────────────────────────────────────────

def _parse_product(raw: dict) -> ProductData:
    """Convert a raw OpenFoodFacts product dict into a ProductData."""
    return ProductData(
        barcode          = str(raw.get("code", "")),
        name             = raw.get("product_name") or "Unknown Product",
        brand            = raw.get("brands", ""),
        quantity         = raw.get("quantity", ""),
        serving_size     = raw.get("serving_size", ""),
        image_url        = raw.get("image_url", ""),
        ingredients_text = raw.get("ingredients_text", "") or "",
        nutriments       = raw.get("nutriments", {}),
        allergens_tags   = raw.get("allergens_tags", []),
        labels_tags      = raw.get("labels_tags", []),
        categories_tags  = raw.get("categories_tags", []),
        additives_tags   = raw.get("additives_tags", []),
    )


# ── public API ────────────────────────────────────────────────────────────────

def fetch_product_by_barcode(barcode: str) -> Optional[ProductData]:
    """
    Fetch a single product by its barcode from Open Food Facts.

    Args:
        barcode: EAN-13, UPC-A, QR payload, etc.

    Returns:
        ProductData if found, None otherwise.
    """
    url = _BARCODE_URL.format(barcode=barcode.strip())
    try:
        resp = _SESSION.get(url, timeout=_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        if data.get("status") == 1 and "product" in data:
            product = _parse_product(data["product"])
            logger.info("Fetched product: %s (%s)", product.name, barcode)
            return product
        logger.warning("Product not found for barcode: %s", barcode)
    except requests.RequestException as exc:
        logger.error("API error for barcode %s: %s", barcode, exc)
    return None


def search_products(query: str,
                    page_size: int = 10,
                    page: int = 1) -> list[ProductData]:
    """
    Search Open Food Facts by product name.

    Args:
        query:     Search string (e.g. "Nutella", "oat milk").
        page_size: Number of results per page (max 50).
        page:      Page number (1-indexed).

    Returns:
        List of ProductData (may be empty).
    """
    params = {
        "search_terms":  query,
        "search_simple": 1,
        "action":        "process",
        "json":          1,
        "page_size":     min(page_size, 50),
        "page":          page,
        "fields":        _FIELDS,
    }
    try:
        resp = _SESSION.get(_SEARCH_URL, params=params, timeout=_TIMEOUT)
        resp.raise_for_status()
        raw_list = resp.json().get("products", [])
        products = [_parse_product(p) for p in raw_list if p.get("product_name")]
        logger.info("Search '%s' → %d results", query, len(products))
        return products
    except requests.RequestException as exc:
        logger.error("Search API error for '%s': %s", query, exc)
        return []
