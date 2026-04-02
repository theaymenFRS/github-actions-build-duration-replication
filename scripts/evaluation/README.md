# scripts/evaluation/

Ce dossier contient les scripts liés aux références de comparaison et à l’analyse qualitative.

## Fichiers

- `baseline_lag1.py` — reproduit la baseline lag-1 sur les projets étudiés. Sortie : `results/rq1/baseline_lag1/<projet>/baseline_lag1.log`.
- `extract_iter5_predictions_all_models.py` — reconstruit les prédictions build par build pour **tous les modèles** de la RQ1 à l’itération 5, ainsi que les deux baselines.

## Utilisation recommandée

Depuis la racine du dépôt :

```bash
py scripts/evaluation/baseline_lag1.py
```

```bash
py scripts/evaluation/extract_iter5_predictions_all_models.py
```

Options possibles :

```bash
py scripts/evaluation/extract_iter5_predictions_all_models.py --model random_forest
py scripts/evaluation/extract_iter5_predictions_all_models.py --project filterlists
py scripts/evaluation/extract_iter5_predictions_all_models.py --project bmad --model xgboost
```
