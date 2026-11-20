"""Nettoyage des données et construction des variables (partagé entre l'entraînement et l'application)."""
import numpy as np
import pandas as pd

NUM_COLS = ["Prod. year", "Levy", "Engine volume", "Turbo", "Mileage", "Cylinders", "Airbags"]
CAT_COLS = ["Manufacturer", "Category", "Leather interior", "Fuel type",
            "Gear box type", "Drive wheels", "Doors", "Wheel", "Color"]
FEATURES = NUM_COLS + CAT_COLS
DOORS_MAP = {"04-May": "4-5", "02-Mar": "2-3", ">5": ">5"}


def clean_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Transforme le fichier brut (format Kaggle) en variables exploitables."""
    out = df.copy()
    out["Levy"] = pd.to_numeric(out["Levy"].replace("-", np.nan), errors="coerce")
    ev = out["Engine volume"].astype(str)
    out["Turbo"] = ev.str.contains("Turbo", case=False).astype(int)
    out["Engine volume"] = pd.to_numeric(ev.str.extract(r"([\d.]+)")[0], errors="coerce")
    out["Mileage"] = pd.to_numeric(out["Mileage"].astype(str).str.extract(r"(\d+)")[0], errors="coerce")
    out["Doors"] = out["Doors"].map(DOORS_MAP).fillna(out["Doors"])
    return out


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Sélectionne les variables d'entrée du modèle, dans l'ordre attendu."""
    X = df[FEATURES].copy()
    for c in CAT_COLS:
        X[c] = X[c].astype(str)
    for c in NUM_COLS:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    return X


def price_class(price: float) -> str:
    """Gamme de prix utilisée dans le projet initial."""
    if price < 15000:
        return "Economy (< 15 000 $)"
    if price < 50000:
        return "Mid-Range (15 000 – 50 000 $)"
    return "Luxury (≥ 50 000 $)"
