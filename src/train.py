"""Entraînement du modèle d'estimation de la valeur vénale.

Usage : python src/train.py data/car_price_prediction.csv
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.base import clone
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).parent))
from preprocessing import CAT_COLS, NUM_COLS, clean_raw, prepare_features  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PRICE_MIN, PRICE_MAX = 500, 250_000   # annonces aberrantes exclues (prix symboliques ou erronés)
SEED = 42


def build_preprocessor():
    num = Pipeline([("imp", SimpleImputer(strategy="median"))])
    cat = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                    ("ohe", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20,
                                          sparse_output=False))])
    return ColumnTransformer([("num", num, NUM_COLS), ("cat", cat, CAT_COLS)])


def metrics(y_true, y_pred):
    ape = np.abs(y_pred - y_true) / y_true
    return {"r2_log": float(r2_score(np.log(y_true), np.log(y_pred))),
            "r2": float(r2_score(y_true, y_pred)),
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "median_ape": float(np.median(ape))}


def main(csv_path):
    raw = pd.read_csv(csv_path)
    df = clean_raw(raw).drop_duplicates()
    df = df[(df["Price"] >= PRICE_MIN) & (df["Price"] <= PRICE_MAX)]
    df = df[(df["Mileage"] < 1_000_000) & (df["Engine volume"] > 0)]
    X, y = prepare_features(df), df["Price"].astype(float)

    # 60 % entraînement, 20 % calibration de la fourchette, 20 % test
    X_tmp, X_test, y_tmp, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)
    X_train, X_cal, y_train, y_cal = train_test_split(X_tmp, y_tmp, test_size=0.25, random_state=SEED)

    candidates = {
        "Random Forest": RandomForestRegressor(n_estimators=200, min_samples_leaf=3, max_features=0.5,
                                               n_jobs=-1, random_state=SEED),
        "XGBoost": XGBRegressor(n_estimators=800, learning_rate=0.05, max_depth=7,
                                subsample=0.8, colsample_bytree=0.8, random_state=SEED, n_jobs=-1),
    }
    # Choix du modèle par validation croisée à 5 plis (sur entraînement + calibration),
    # avec l'erreur absolue moyenne en dollars comme critère métier
    results = {}
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    for name, reg in candidates.items():
        fold = []
        for tr, va in kf.split(X_tmp):
            pipe = Pipeline([("prep", build_preprocessor()), ("reg", clone(reg))])
            pipe.fit(X_tmp.iloc[tr], np.log(y_tmp.iloc[tr]))   # cible : logarithme du prix
            fold.append(metrics(y_tmp.iloc[va].values, np.exp(pipe.predict(X_tmp.iloc[va]))))
        results[name] = {k: float(np.mean([f[k] for f in fold])) for k in fold[0]}
        print(name, results[name])

    best = min(results, key=lambda k: results[k]["mae"])
    model = Pipeline([("prep", build_preprocessor()), ("reg", clone(candidates[best]))])
    model.fit(X_train, np.log(y_train))

    # Fourchette à 80 % par calibration conforme (résidus en log sur l'échantillon de calibration)
    res_cal = np.abs(np.log(y_cal.values) - model.predict(X_cal))
    n = len(res_cal)
    q = float(np.quantile(res_cal, min(1.0, np.ceil((n + 1) * 0.80) / n)))

    # Évaluation finale sur l'échantillon de test, jamais vu
    pred_test_log = model.predict(X_test)
    pred_test = np.exp(pred_test_log)
    test = metrics(y_test.values, pred_test)
    lo, hi = np.exp(pred_test_log - q), np.exp(pred_test_log + q)
    test["coverage_80"] = float(np.mean((y_test.values >= lo) & (y_test.values <= hi)))

    # Importance des variables (regroupées par variable d'origine)
    names = model.named_steps["prep"].get_feature_names_out()
    imp = model.named_steps["reg"].feature_importances_
    grouped = {}
    for n_, v in zip(names, imp):
        base = n_.split("__", 1)[1]
        key = next((c for c in CAT_COLS if base.startswith(c + "_")), base)
        grouped[key] = grouped.get(key, 0) + float(v)
    grouped = dict(sorted(grouped.items(), key=lambda kv: -kv[1]))

    # Valeurs proposées dans l'application
    options = {c: sorted(df[c].astype(str).unique().tolist()) for c in CAT_COLS}
    options["Manufacturer"] = df["Manufacturer"].value_counts().index[:40].sort_values().tolist()
    ranges = {c: [float(df[c].quantile(0.01)), float(df[c].quantile(0.99)), float(df[c].median())]
              for c in NUM_COLS}

    meta = {"best_model": best, "cross_validation_5_folds": results, "test": test, "interval_log_halfwidth": q,
            "n_rows_used": int(len(df)), "n_rows_raw": int(len(raw)),
            "price_filter": [PRICE_MIN, PRICE_MAX], "feature_importance": grouped,
            "options": options, "ranges": ranges}
    (ROOT / "model").mkdir(exist_ok=True)
    joblib.dump(model, ROOT / "model" / "modele_prix.joblib", compress=3)
    (ROOT / "model" / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print("Meilleur modèle :", best)
    print("Test :", json.dumps(test, indent=2))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ROOT / "data" / "car_price_prediction.csv")
