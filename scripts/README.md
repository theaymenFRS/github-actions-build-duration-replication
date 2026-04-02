# scripts/

Ce répertoire réorganise le code source archivé selon les étapes du pipeline expérimental.

## Remarque importante

Le code d’origine est en grande partie **monolithique** : le prétraitement, l’ingénierie des caractéristiques, l’entraînement et l’évaluation sont souvent intégrés dans les scripts de modélisation eux-mêmes. Le package conserve cette organisation aussi fidèlement que possible, tout en documentant le rôle de chaque script.

## Organisation

- `data_collection/` — documentation pour l’étape de collecte externe avec GHAminer.
- `preprocessing/` — `preprocess_common.py` (prétraitement partagé : filtres, lags, `file_types`, etc., importé par RQ1 et RQ2) et `README.md`.
- `feature_engineering/` — scripts d’importance des variables et notes sur les caractéristiques temporelles du mémoire.
- `modeling/rq1/` — les cinq scripts d’entraînement de la RQ1 (DT, RF, GBR, XGBoost, LightGBM).
- `modeling/rq2/` — script d’évaluation OLD vs RECENT pour la RQ2.
- `evaluation/` — baseline lag-1 et extraction des prédictions itération 5.
- `plotting/` — scripts auxiliaires pour générer figures et tableaux du mémoire.

## Journaux RQ1 (noms de fichiers)

Chaque script RQ1 écrit sous `results/rq1/<modèle>/` un fichier par projet, avec un nom qui reprend explicitement le modèle :

- `results/rq1/decision_tree/decision_tree_<projet>_RQ1.log`
- `results/rq1/random_forest/random_forest_<projet>_RQ1.log`
- `results/rq1/gradient_boosting/gradient_boosting_<projet>_RQ1.log`
- `results/rq1/xgboost/xgboost_<projet>_RQ1.log`
- `results/rq1/lightgbm/lightgbm_<projet>_RQ1.log`

## Baseline lag-1 (RQ1)

`evaluation/baseline_lag1.py` écrit pour chaque projet :

- `results/rq1/baseline_lag1/<projet>/baseline_lag1.log`

## Sorties RQ2 (noms de fichiers)

Le script `modeling/rq2/rq2_old_vs_recent.py` écrit sous `results/rq2/concept_drift_old_vs_recent_DEFAULTPARAMS/` :

- `rq2_ALL_PROJECTS_RQ2.log` — journal agrégé sur tous les projets
- Par projet (`<projet>/`) : `rq2_<projet>_RQ2.log`, `<projet>_processed.csv`, `rq2_<projet>_RQ2_GRID.png`
- Figure globale : `rq2_ALL_PROJECTS_RQ2_GRID.png`
