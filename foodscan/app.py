"""
app.py
======
FastAPI backend for FoodScan.

Endpoints:
  GET  /                          → health check
  GET  /product/{barcode}         → fetch + analyse by barcode
  GET  /search?q=...&page_size=10 → search by name
  POST /scan/image                → upload image, decode barcode, analyse

Run:
    uvicorn foodscan.app:app --reload --port 8000
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Optional

from fastapi import FastAPI, HTTPException, Query, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Allow running this file directly: `python foodscan/app.py`
# by ensuring the repo root is on sys.path.
if __package__ is None:
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from foodscan.api_handler import fetch_product_by_barcode, search_products
from foodscan.analyzer import classify_product
from foodscan.scanner import decode_barcode_bytes, SCAN_READY

# ── logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)

# ── app ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="FoodScan API",
    description=(
        "Scan packaged food barcodes and get instant nutritional insights, "
        "additive warnings, allergen alerts, and diet compatibility."
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ══════════════════════════════════════════════════════════════════════════════
# Routes
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/", tags=["Health"])
def health_check():
    """Simple liveness probe."""
    return {
        "status":       "ok",
        "service":      "FoodScan API",
        "version":      "2.0.0",
        "scan_ready":   SCAN_READY,
    }


@app.get("/product/{barcode}", tags=["Products"])
def get_product(barcode: str):
    """
    Fetch a product by barcode and return the full analysis report.

    - **barcode**: EAN-13, UPC-A, or any barcode string
    """
    product = fetch_product_by_barcode(barcode)
    if not product:
        raise HTTPException(
            status_code=404,
            detail=f"Product with barcode '{barcode}' not found on Open Food Facts.",
        )
    report = classify_product(product)
    return JSONResponse(content=report.to_dict())


@app.get("/search", tags=["Products"])
def search(
    q: str = Query(..., min_length=2, description="Product name to search"),
    page_size: int = Query(10, ge=1, le=50, description="Results per page"),
    page: int = Query(1, ge=1, description="Page number"),
):
    """
    Search products by name and return a list with basic analysis for each.
    """
    products = search_products(q, page_size=page_size, page=page)
    if not products:
        return JSONResponse(content={"query": q, "count": 0, "results": []})

    results = []
    for p in products:
        report = classify_product(p)
        results.append({
            "barcode":    p.barcode,
            "name":       p.name,
            "brand":      p.brand,
            "image_url":  p.image_url,
            "calories":   p.calories_100g,
            "overall":    report.overall,
            "verdict":    report.verdict_text,
            "warnings":   report.warnings[:3],   # top 3 warnings in list view
        })

    return JSONResponse(content={
        "query":   q,
        "count":   len(results),
        "results": results,
    })


@app.post("/scan/image", tags=["Scanner"])
async def scan_image(file: UploadFile = File(...)):
    """
    Upload an image containing a barcode or QR code.
    The API will decode it and return the full product analysis.

    Accepts: JPEG, PNG, BMP, WEBP
    """
    if not SCAN_READY:
        raise HTTPException(
            status_code=503,
            detail="Image scanning unavailable. Install opencv-contrib-python.",
        )

    allowed = {"image/jpeg", "image/png", "image/bmp", "image/webp"}
    if file.content_type not in allowed:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{file.content_type}'. Use JPEG or PNG.",
        )

    data = await file.read()
    scan = decode_barcode_bytes(data)

    if not scan:
        raise HTTPException(
            status_code=422,
            detail="No barcode or QR code detected in the uploaded image.",
        )

    product = fetch_product_by_barcode(scan.barcode)
    if not product:
        return JSONResponse(content={
            "barcode":    scan.barcode,
            "symbology":  scan.symbology,
            "product":    None,
            "message":    "Barcode decoded but product not found on Open Food Facts.",
        })

    report = classify_product(product)
    return JSONResponse(content={
        "barcode":   scan.barcode,
        "symbology": scan.symbology,
        **report.to_dict(),
    })
