# Prédiction du prix des voitures d'occasion

Projet de data science réalisé en équipe (juin 2025), suivant la méthodologie **CRISP-DM** : compréhension des données, préparation, modélisation, évaluation et déploiement.

**Objectifs**
- **Régression** : prédire le prix d'un véhicule d'occasion à partir de ses caractéristiques (constructeur, année, kilométrage, moteur, carburant, boîte de vitesses, etc.).
- **Classification** : classer chaque véhicule dans une gamme de prix : *Economy* (< 15 000 $), *Mid-Range* (15 000–50 000 $) ou *Luxury* (≥ 50 000 $).

## Démarche

1. **Nettoyage** : conversion des variables mal typées (`Levy`, `Mileage`, `Engine volume`, `Doors`), suppression des doublons, repérage des valeurs aberrantes par la méthode IQR (signalées plutôt que supprimées, pour conserver les véhicules de luxe).
2. **Traitement des valeurs manquantes de `Levy`** : comparaison de plusieurs stratégies (médiane, moyenne, zéro), puis imputation KNN dans la version finale, en excluant le prix pour éviter toute fuite de la cible.
3. **Ingénierie de variables** : âge du véhicule, kilométrage par année, ratio volume moteur / cylindres, regroupement des constructeurs peu fréquents.
4. **Modélisation** : régression linéaire, Lasso et Ridge comme références, puis Random Forest et XGBoost, dans des pipelines scikit-learn (`ColumnTransformer`, `OneHotEncoder`, `StandardScaler`), avec validation croisée et optimisation des hyperparamètres (`GridSearchCV`).

## Résultats (jeu de test)

| Tâche | Modèle | Résultat |
|---|---|---|
| Régression linéaire de référence | Régression linéaire | R² ≈ 0,21 en validation croisée |
| Régression du prix | Random Forest | **R² = 0,69**, RMSE ≈ 9 963 $ |
| Classification par gamme de prix | Random Forest | **Exactitude ≈ 81 %**, F1 pondéré ≈ 0,81 |

Les modèles linéaires expliquent mal le prix : les relations sont fortement non linéaires et la distribution des prix est très asymétrique. Les méthodes d'ensemble (forêts aléatoires) améliorent nettement les performances.

## Contenu du dépôt

```
notebooks/prediction_prix_voitures.ipynb   # analyse complète et modèles (avec les sorties)
data/                                      # emplacement du jeu de données (non inclus)
requirements.txt
```

## Reproduire l'analyse

1. Télécharger le jeu de données **Car Price Prediction Challenge** (`car_price_prediction.csv`, disponible sur Kaggle) et le placer dans `data/`.
2. Installer les dépendances : `pip install -r requirements.txt`
3. Ouvrir le notebook depuis le dossier `notebooks/` : `jupyter notebook notebooks/prediction_prix_voitures.ipynb`

## Équipe

- Soro Onansongui Elixir
- Ouedraogo Makemissa Mignantcho
- Badolo A.
- Angoran A.

## Outils

Python · pandas · NumPy · scikit-learn · XGBoost · Matplotlib · Seaborn · Jupyter
