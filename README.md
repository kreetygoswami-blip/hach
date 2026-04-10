# Food Ingredient & Nutrition Tracking Scanner (Python)

This repo already includes a working prototype in `agriculture.py` (app name **FoodScan**):
- Scan/enter a **barcode/QR**
- Fetch product data from **OpenFoodFacts**
- Show **ingredients, calories, nutrition facts**
- Provide **rule-based guidance** (safe / moderate / avoid), plus allergy & diet compatibility
- Stores **history** and **favourites** locally in `foodscan_data.json`

## Setup

```bash
python3 -m pip install -r requirements.txt
```

## Run

```bash
python3 agriculture.py
```

## Notes

- Camera scanning uses OpenCV (`opencv-python`). If you don’t need camera scan, you can still search by name / enter barcode manually.
- Data source: OpenFoodFacts public API.

