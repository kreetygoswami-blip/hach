"""
streamlit_app.py
================
Streamlit UI for FoodScan.

Run:
    streamlit run foodscan/streamlit_app.py
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
import pandas as pd
from PIL import Image
import io

from foodscan.api_handler import fetch_product_by_barcode, search_products
from foodscan.analyzer    import classify_product, AnalysisReport
from foodscan.scanner     import decode_barcode_bytes, SCAN_READY

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🥦 FoodScan",
    page_icon="🥦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #F5F7FA; }
    .verdict-box {
        padding: 14px 20px;
        border-radius: 12px;
        font-size: 1.1rem;
        font-weight: 700;
        text-align: center;
        color: white;
        margin-bottom: 12px;
    }
    .chip {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
        color: white;
        margin: 3px;
    }
    .chip-red    { background-color: #E74C3C; }
    .chip-orange { background-color: #F39C12; }
    .chip-green  { background-color: #27AE60; }
    .chip-grey   { background-color: #7F8C8D; }
    .chip-blue   { background-color: #3498DB; }
    .section-title {
        font-size: 1rem;
        font-weight: 700;
        color: #2C3E50;
        margin-top: 16px;
        margin-bottom: 6px;
        border-left: 4px solid #2ECC71;
        padding-left: 8px;
    }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _verdict_html(report: AnalysisReport) -> str:
    return (
        f'<div class="verdict-box" style="background-color:{report.verdict_color}">'
        f'{report.verdict_text}</div>'
    )


def _rating_color(rating: str) -> str:
    return {"safe": "#27AE60", "moderate": "#F39C12",
            "avoid": "#E74C3C", "neutral": "#7F8C8D"}.get(rating, "#7F8C8D")


def _chip(text: str, color_class: str) -> str:
    return f'<span class="chip chip-{color_class}">{text}</span>'


def _diet_chip(d: dict) -> str:
    c = d["compatible"]
    if c is True:   cls, tick = "green",  "✔"
    elif c is False: cls, tick = "red",   "✘"
    else:            cls, tick = "grey",  "?"
    return _chip(f"{d['icon']} {tick} {d['diet']}", cls)


def render_report(report: AnalysisReport):
    """Render the full analysis report in Streamlit."""
    p = report.product

    # ── hero row ──────────────────────────────────────────────────────────────
    col_img, col_info = st.columns([1, 2])
    with col_img:
        if p.image_url:
            st.image(p.image_url, use_container_width=True)
        else:
            st.markdown("### 🍽")

    with col_info:
        st.markdown(f"## {p.name}")
        if p.brand:
            st.markdown(f"**Brand:** {p.brand}")
        if p.quantity:
            st.markdown(f"**Quantity:** {p.quantity}")
        if p.barcode:
            st.markdown(f"**Barcode:** `{p.barcode}`")
        st.markdown(_verdict_html(report), unsafe_allow_html=True)

    st.divider()

    # ── warnings ──────────────────────────────────────────────────────────────
    if report.warnings:
        st.markdown('<div class="section-title">⚠️ Warnings</div>',
                    unsafe_allow_html=True)
        for w in report.warnings:
            st.warning(w)

    # ── calories ──────────────────────────────────────────────────────────────
    kcal = p.calories_100g
    if kcal:
        st.metric("🔥 Calories (per 100 g)", f"{kcal:.0f} kcal")

    st.divider()

    # ── nutrient table ────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">🧪 Nutrition Facts (per 100 g)</div>',
                unsafe_allow_html=True)
    if not report.nutrient_df.empty:
        df = report.nutrient_df.copy()
        df["value"] = df.apply(
            lambda r: f"{r['value_per_100g']} {r['unit']}", axis=1)
        df["status"] = df["icon"] + "  " + df["rating"].str.capitalize()

        st.dataframe(
            df[["nutrient", "value", "status"]].rename(columns={
                "nutrient": "Nutrient",
                "value":    "Amount",
                "status":   "Rating",
            }),
            use_container_width=True,
            hide_index=True,
        )

        # Mini bar chart for macros
        macro_keys = ["Total Fat", "Carbohydrates", "Protein", "Sugars",
                      "Dietary Fiber"]
        macro_df = df[df["nutrient"].isin(macro_keys)][
            ["nutrient", "value_per_100g"]
        ].set_index("nutrient")
        if not macro_df.empty:
            st.bar_chart(macro_df, color="#2ECC71")
    else:
        st.info("No nutritional data available for this product.")

    st.divider()

    # ── harmful additives ─────────────────────────────────────────────────────
    st.markdown('<div class="section-title">🚫 Harmful Additives</div>',
                unsafe_allow_html=True)
    if report.additives:
        chips_html = "".join(
            _chip(f"{a['icon']} {a['name']}",
                  "red" if a["severity"] == "avoid" else "orange")
            for a in report.additives
        )
        st.markdown(chips_html, unsafe_allow_html=True)
    else:
        st.success("✅ No harmful additives detected.")

    st.divider()

    # ── allergens ─────────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">⚠️ Allergy Alerts</div>',
                unsafe_allow_html=True)
    if report.allergens:
        chips_html = "".join(
            _chip(f"{a['icon']} {a['label']}", "red")
            for a in report.allergens
        )
        st.markdown(chips_html, unsafe_allow_html=True)
    else:
        st.success("✅ No common allergens detected.")

    st.divider()

    # ── diet compatibility ────────────────────────────────────────────────────
    st.markdown('<div class="section-title">🥗 Diet Compatibility</div>',
                unsafe_allow_html=True)
    chips_html = "".join(_diet_chip(d) for d in report.diets)
    st.markdown(chips_html, unsafe_allow_html=True)

    # Diet detail table
    diet_rows = [
        {"Diet": d["diet"], "Compatible": (
            "✔ Yes" if d["compatible"] is True else
            "✘ No"  if d["compatible"] is False else "? Unknown"
        ), "Note": d["note"]}
        for d in report.diets
    ]
    st.dataframe(pd.DataFrame(diet_rows), use_container_width=True,
                 hide_index=True)

    st.divider()

    # ── ingredients ───────────────────────────────────────────────────────────
    if p.ingredients_text:
        st.markdown('<div class="section-title">📋 Ingredients</div>',
                    unsafe_allow_html=True)
        with st.expander("Show full ingredient list"):
            st.write(p.ingredients_text)


# ══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.image("https://world.openfoodfacts.org/images/logos/off-logo-horizontal-light.svg",
             use_container_width=True)
    st.markdown("## 🥦 FoodScan")
    st.markdown("Scan · Understand · Decide")
    st.divider()

    mode = st.radio(
        "Choose input method:",
        ["🔢 Barcode Lookup", "🔍 Search by Name", "📷 Camera Capture", "🖼️ Upload Image"],
        index=0,
    )
    st.divider()
    st.markdown("**Data source:** [Open Food Facts](https://world.openfoodfacts.org)")
    st.markdown("**Scanner:** " + ("✅ Ready" if SCAN_READY
                                   else "⚠️ Install opencv-contrib-python"))


# ══════════════════════════════════════════════════════════════════════════════
# Main content
# ══════════════════════════════════════════════════════════════════════════════

st.title("🥦 FoodScan – Nutritional Intelligence")
st.caption("Instant food label analysis powered by Open Food Facts")

# ── Demo quick-launch ─────────────────────────────────────────────────────────
st.markdown("**Quick demo:**")
demo_cols = st.columns(4)
demos = [
    ("Nutella",  "3017620422003"),
    ("Pepsi",    "012000001765"),
    ("Oreo",     "7622210951137"),
    ("Pringles", "5053990101536"),
]
for col, (name, code) in zip(demo_cols, demos):
    if col.button(f"🍫 {name}", use_container_width=True):
        st.session_state["demo_barcode"] = code

st.divider()

# ── Mode: Barcode Lookup ──────────────────────────────────────────────────────
if mode == "🔢 Barcode Lookup":
    default_bc = st.session_state.pop("demo_barcode", "")
    barcode = st.text_input(
        "Enter barcode number",
        value=default_bc,
        placeholder="e.g. 3017620422003",
    )
    if st.button("🔍 Analyse", type="primary", use_container_width=True):
        if not barcode.strip():
            st.warning("Please enter a barcode.")
        else:
            with st.spinner("Fetching product data…"):
                product = fetch_product_by_barcode(barcode.strip())
            if not product:
                st.error(f"Product not found for barcode: `{barcode}`")
            else:
                with st.spinner("Analysing…"):
                    report = classify_product(product)
                render_report(report)

# ── Mode: Search by Name ──────────────────────────────────────────────────────
elif mode == "🔍 Search by Name":
    query = st.text_input("Search for a food product",
                          placeholder="e.g. Nutella, oat milk, protein bar")
    page_size = st.slider("Results to show", 3, 20, 8)

    if st.button("🔍 Search", type="primary", use_container_width=True):
        if not query.strip():
            st.warning("Please enter a search term.")
        else:
            with st.spinner(f'Searching for "{query}"…'):
                products = search_products(query, page_size=page_size)

            if not products:
                st.error("No products found. Try a different search term.")
            else:
                st.success(f"Found {len(products)} products")
                for p in products:
                    with st.expander(
                        f"{'🖼 ' if p.image_url else '🍽 '}{p.name}"
                        f"{'  —  ' + p.brand if p.brand else ''}",
                        expanded=False,
                    ):
                        with st.spinner("Analysing…"):
                            report = classify_product(p)
                        render_report(report)

# ── Mode: Camera Capture ──────────────────────────────────────────────────────
elif mode == "📷 Camera Capture":
    st.markdown("Use your camera to capture a barcode or QR code image.")

    if not SCAN_READY:
        st.error(
            "Camera scanning requires **opencv-contrib-python**.\n\n"
            "```\npip install opencv-contrib-python\n```"
        )
    else:
        img = st.camera_input("Capture barcode / QR")
        auto = st.toggle("Auto-scan (re-try decode on this photo)", value=True)
        if img:
            img_bytes = img.getvalue()
            st.image(img_bytes, caption="Captured image", width=320)

            with st.spinner("Decoding…"):
                scan = decode_barcode_bytes(img_bytes)
                if auto and not scan:
                    # Re-try a couple times (Streamlit sometimes provides a
                    # slightly different jpeg encode across reruns).
                    scan = decode_barcode_bytes(img_bytes) or decode_barcode_bytes(img_bytes)

            if not scan:
                st.error("No barcode/QR detected. Tips: fill the frame, reduce glare, hold steady.")
            else:
                st.success(f"✔ Detected: `{scan.barcode}` ({scan.symbology})")
                with st.spinner("Fetching product data…"):
                    product = fetch_product_by_barcode(scan.barcode)
                if not product:
                    st.warning(f"Decoded `{scan.barcode}` but not found on Open Food Facts.")
                else:
                    with st.spinner("Analysing…"):
                        report = classify_product(product)
                    render_report(report)

# ── Mode: Upload Image ────────────────────────────────────────────────────────
elif mode == "🖼️ Upload Image":
    st.markdown("Upload a photo of a barcode or QR code.")

    if not SCAN_READY:
        st.error(
            "Image scanning requires **opencv-contrib-python**.\n\n"
            "```\npip install opencv-contrib-python\n```"
        )
    else:
        uploaded = st.file_uploader(
            "Choose an image",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
        )
        if uploaded:
            img_bytes = uploaded.read()
            st.image(img_bytes, caption="Uploaded image", width=320)

            with st.spinner("Decoding…"):
                scan = decode_barcode_bytes(img_bytes)

            if not scan:
                st.error("No barcode/QR detected in the image. Try a clearer photo with good lighting.")
            else:
                st.success(f"✔ Detected: `{scan.barcode}` ({scan.symbology})")
                with st.spinner("Fetching product data…"):
                    product = fetch_product_by_barcode(scan.barcode)
                if not product:
                    st.warning(f"Decoded `{scan.barcode}` but not found on Open Food Facts.")
                else:
                    with st.spinner("Analysing…"):
                        report = classify_product(product)
                    render_report(report)
