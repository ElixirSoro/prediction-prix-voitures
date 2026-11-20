"""Estimateur de valeur vénale automobile – application Streamlit.

Lancer en local : streamlit run app.py
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from preprocessing import FEATURES, clean_raw, prepare_features, price_class  # noqa: E402

st.set_page_config(page_title="Estimateur de valeur vénale auto", page_icon="🚗", layout="wide")


@st.cache_resource
def load_model():
    model = joblib.load(ROOT / "model" / "modele_prix.joblib")
    meta = json.loads((ROOT / "model" / "metadata.json").read_text())
    return model, meta


model, meta = load_model()
Q = meta["interval_log_halfwidth"]
OPT, RNG = meta["options"], meta["ranges"]
TOL = 0.0  # tolérance au-delà de la fourchette à 80 %


def estimate(X: pd.DataFrame) -> pd.DataFrame:
    """Prix estimé et fourchette à 80 % pour chaque ligne."""
    log_pred = model.predict(prepare_features(X))
    return pd.DataFrame({"Estimation": np.exp(log_pred),
                         "Borne basse (80 %)": np.exp(log_pred - Q),
                         "Borne haute (80 %)": np.exp(log_pred + Q)}, index=X.index)


def verdict(declared, low, high):
    if pd.isna(declared) or declared <= 0:
        return "—"
    if declared > high * (1 + TOL):
        return "⚠️ Sur-évaluation possible"
    if declared < low * (1 - TOL):
        return "⚠️ Sous-assurance possible"
    return "✅ Cohérente"


def usd(x):
    return f"{x:,.0f} $".replace(",", " ")


st.title("🚗 Estimateur de valeur vénale automobile")
st.caption("Démonstration d'un modèle de machine learning appliqué à l'assurance automobile : "
           "souscription, indemnisation en perte totale et détection d'anomalies.")

tab1, tab2, tab3 = st.tabs(["Estimer un véhicule", "Contrôler un portefeuille", "Le modèle"])

# ------------------------------------------------------------------ Onglet 1
with tab1:
    with st.form("vehicule"):
        c1, c2, c3 = st.columns(3)
        with c1:
            manu = st.selectbox("Constructeur", OPT["Manufacturer"],
                                index=OPT["Manufacturer"].index("TOYOTA") if "TOYOTA" in OPT["Manufacturer"] else 0)
            cat = st.selectbox("Catégorie", OPT["Category"], index=OPT["Category"].index("Sedan"))
            year = st.slider("Année de fabrication", int(RNG["Prod. year"][0]), int(RNG["Prod. year"][1]),
                             int(RNG["Prod. year"][2]))
            mileage = st.number_input("Kilométrage (km)", 0, 1_000_000, int(RNG["Mileage"][2]), step=5000)
        with c2:
            fuel = st.selectbox("Carburant", OPT["Fuel type"], index=OPT["Fuel type"].index("Petrol"))
            engine = st.number_input("Cylindrée (litres)", 0.5, 7.0, float(RNG["Engine volume"][2]), step=0.1)
            turbo = st.checkbox("Moteur turbo")
            cyl = st.selectbox("Cylindres", [3, 4, 5, 6, 8, 10, 12], index=1)
            gear = st.selectbox("Boîte de vitesses", OPT["Gear box type"], index=OPT["Gear box type"].index("Automatic"))
        with c3:
            drive = st.selectbox("Roues motrices", OPT["Drive wheels"], index=OPT["Drive wheels"].index("Front"))
            leather = st.selectbox("Intérieur cuir", OPT["Leather interior"], index=OPT["Leather interior"].index("Yes"))
            airbags = st.slider("Nombre d'airbags", 0, 16, int(RNG["Airbags"][2]))
            color = st.selectbox("Couleur", OPT["Color"], index=OPT["Color"].index("Black") if "Black" in OPT["Color"] else 0)
            doors = st.selectbox("Portes", OPT["Doors"], index=OPT["Doors"].index("4-5") if "4-5" in OPT["Doors"] else 0)
            wheel = st.selectbox("Volant", OPT["Wheel"], index=OPT["Wheel"].index("Left wheel") if "Left wheel" in OPT["Wheel"] else 0)
        levy = st.number_input("Taxe d'importation « Levy » ($), laisser 0 si inconnue", 0, 20_000, 0, step=50)
        declared = st.number_input("Valeur déclarée par l'assuré ($), facultatif", 0, 1_000_000, 0, step=500)
        go = st.form_submit_button("Estimer", type="primary")

    if go:
        row = pd.DataFrame([{
            "Manufacturer": manu, "Category": cat, "Prod. year": year, "Mileage": mileage,
            "Fuel type": fuel, "Engine volume": engine, "Turbo": int(turbo), "Cylinders": cyl,
            "Gear box type": gear, "Drive wheels": drive, "Leather interior": leather,
            "Airbags": airbags, "Color": color, "Doors": doors, "Wheel": wheel,
            "Levy": np.nan if levy == 0 else levy}])
        res = estimate(row).iloc[0]
        m1, m2, m3 = st.columns(3)
        m1.metric("Valeur vénale estimée", usd(res["Estimation"]))
        m2.metric("Fourchette à 80 %", f"{usd(res['Borne basse (80 %)'])} – {usd(res['Borne haute (80 %)'])}")
        m3.metric("Gamme", price_class(res["Estimation"]).split(" (")[0])
        if declared > 0:
            v = verdict(declared, res["Borne basse (80 %)"], res["Borne haute (80 %)"])
            ecart = declared / res["Estimation"] - 1
            (st.success if v.startswith("✅") else st.warning)(
                f"Valeur déclarée : {usd(declared)} (écart de {ecart:+.0%} par rapport à l'estimation). {v}"
                + ("" if v.startswith("✅") else " : la valeur déclarée sort de la fourchette à 80 %."))
        st.caption("La fourchette contient la vraie valeur dans environ 80 % des cas (vérifié sur des "
                   "véhicules jamais vus par le modèle). Une estimation reste indicative : elle complète, "
                   "sans la remplacer, l'expertise d'un professionnel.")

# ------------------------------------------------------------------ Onglet 2
with tab2:
    st.markdown("Chargez un fichier CSV de véhicules au **même format que le jeu de données d'origine**, "
                "avec une colonne facultative `Valeur_declaree`. L'application estime chaque véhicule et "
                "signale les valeurs déclarées hors de la fourchette.")
    example = ROOT / "data" / "exemple_portefeuille.csv"
    if example.exists():
        st.download_button("Télécharger un fichier d'exemple", example.read_bytes(),
                           "exemple_portefeuille.csv", "text/csv")
    up = st.file_uploader("Fichier CSV du portefeuille", type="csv")
    if up is not None:
        raw = pd.read_csv(up)
        missing = [c for c in FEATURES if c not in clean_raw(raw.head(1)).columns]
        if missing:
            st.error("Colonnes manquantes : " + ", ".join(missing))
        else:
            df = clean_raw(raw)
            out = pd.concat([raw.reset_index(drop=True), estimate(df).reset_index(drop=True)], axis=1)
            out["Gamme"] = out["Estimation"].apply(lambda p: price_class(p).split(" (")[0])
            if "Valeur_declaree" in out.columns:
                out["Écart"] = out["Valeur_declaree"] / out["Estimation"] - 1
                out["Statut"] = [verdict(d, lo, hi) for d, lo, hi in
                                 zip(out["Valeur_declaree"], out["Borne basse (80 %)"], out["Borne haute (80 %)"])]
                n = len(out)
                over = out["Statut"].str.contains("Sur-évaluation").sum()
                under = out["Statut"].str.contains("Sous-assurance").sum()
                k1, k2, k3 = st.columns(3)
                k1.metric("Véhicules analysés", n)
                k2.metric("Sur-évaluations possibles", f"{over} ({over / n:.0%})")
                k3.metric("Sous-assurances possibles", f"{under} ({under / n:.0%})")
            cols = [c for c in ["Manufacturer", "Model", "Prod. year", "Mileage", "Valeur_declaree",
                                "Estimation", "Borne basse (80 %)", "Borne haute (80 %)", "Écart",
                                "Gamme", "Statut"] if c in out.columns]
            st.dataframe(out[cols].style.format({"Estimation": "{:,.0f}", "Borne basse (80 %)": "{:,.0f}",
                                                  "Borne haute (80 %)": "{:,.0f}", "Écart": "{:+.0%}",
                                                  "Valeur_declaree": "{:,.0f}"}, na_rep="—"),
                         width="stretch")
            st.download_button("Télécharger les résultats (CSV)", out.to_csv(index=False).encode("utf-8"),
                               "resultats_estimation.csv", "text/csv")

# ------------------------------------------------------------------ Onglet 3
with tab3:
    t = meta["test"]
    st.subheader("Performances sur un échantillon de test (20 % des données, jamais vu à l'entraînement)")
    a, b, c, d = st.columns(4)
    a.metric("R² (log du prix)", f"{t['r2_log']:.2f}")
    b.metric("Erreur absolue moyenne", usd(t["mae"]))
    c.metric("Erreur relative médiane", f"{t['median_ape']:.0%}")
    d.metric("Couverture de la fourchette 80 %", f"{t['coverage_80']:.0%}")
    def n(x):
        return f"{x:,}".replace(",", " ")
    st.markdown(f"""
**Modèle retenu : {meta['best_model']}**, choisi par validation croisée à 5 plis parmi une forêt aléatoire et
XGBoost, sur l'erreur absolue moyenne en dollars. La cible est le **logarithme du prix**, ce qui limite l'influence
des véhicules très chers. La fourchette à 80 % est obtenue par **calibration conforme** sur un échantillon séparé.

Données : {n(meta['n_rows_used'])} annonces retenues sur {n(meta['n_rows_raw'])} (doublons et prix aberrants
hors de {n(meta['price_filter'][0])} – {n(meta['price_filter'][1])} $ exclus).
""")
    st.subheader("Importance des variables")
    labels = {"Airbags": "Airbags", "Prod. year": "Année", "Mileage": "Kilométrage",
              "Gear box type": "Boîte de vitesses", "Levy": "Taxe d'importation", "Manufacturer": "Constructeur",
              "Engine volume": "Cylindrée", "Fuel type": "Carburant", "Leather interior": "Intérieur cuir",
              "Color": "Couleur", "Category": "Catégorie", "Turbo": "Turbo", "Drive wheels": "Roues motrices",
              "Cylinders": "Cylindres", "Wheel": "Volant", "Doors": "Portes"}
    imp = pd.DataFrame({"Variable": [labels.get(k, k) for k in meta["feature_importance"]],
                        "Importance": list(meta["feature_importance"].values())})
    st.bar_chart(imp, x="Variable", y="Importance", horizontal=True, sort="-Importance")
    st.subheader("Limites")
    st.markdown("""
- Les prix proviennent d'annonces d'un **marché étranger** (en dollars) : le modèle illustre la méthode et devrait
  être réentraîné sur des données locales (Côte d'Ivoire, zone CIMA) pour un usage réel.
- Un prix d'annonce n'est pas une valeur de transaction ni une valeur à dire d'expert.
- L'état du véhicule (accidents, entretien) n'est pas observé : environ une estimation sur cinq sort de la fourchette.
""")
