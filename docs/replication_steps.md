# Étapes de réplication

Ce document présente l’ordre recommandé pour reproduire les expériences.

## Prérequis

1. Créer l’environnement Python à partir de `environment.yml` ou de `requirements.txt`.
2. Lire `docs/experimental_setup.md`.
3. Les scripts Python du pipeline utilisent des chemins relatifs au dépôt : placer les CSV attendus sous `data/raw/` (voir `data/project_links.csv` et la structure de `data/`).

## Étape 1 — Examiner la liste des projets

- Fichier : `data/project_links.csv`
- But : lister les 10 dépôts publics sélectionnés ainsi que les fichiers de données associés.

## Étape 2 — (Optionnel) Reconstruire les données brutes avec GHAminer

- Outil externe : GHAminer
- Référence : `scripts/data_collection/README.md`
- Sortie attendue : fichiers CSV similaires à ceux présents dans `data/raw/`

## Étape 3 — Fichiers d’entrée pour les expériences

- Répertoire utilisé par les scripts : `data/raw/` (CSV des workflows retenus pour les modèles).
- Le dossier `data/processed/` peut contenir des jeux dérivés ou prétraités selon l’organisation du dépôt ; les scripts listés ci‑dessous lisent les entrées depuis `data/raw/`.

## Étape 4 — Exécuter la baseline lag-1

```bash
py scripts/evaluation/baseline_lag1.py
```

Sortie attendue : un journal par projet sous `results/rq1/baseline_lag1/<projet>/baseline_lag1.log`.

## Étape 5 — Exécuter les scripts de la RQ1

Exécuter les cinq scripts de la RQ1 un par un :

```bash
py scripts/modeling/rq1/decision_tree_rq1.py
py scripts/modeling/rq1/random_forest_rq1.py
py scripts/modeling/rq1/gradient_boosting_rq1.py
py scripts/modeling/rq1/xgboost_rq1.py
py scripts/modeling/rq1/lightgbm_rq1.py
```

Ces scripts réalisent en interne le prétraitement, l’ingénierie des caractéristiques, la sélection de variables, l’optimisation par algorithme génétique et l’évaluation chronologique.

## Étape 6 — Exécuter la RQ2 (OLD vs RECENT)

```bash
py scripts/modeling/rq2/rq2_old_vs_recent.py
```

Sortie attendue : journaux et figures par projet sous `results/rq2/concept_drift_old_vs_recent_DEFAULTPARAMS/` (voir `results/README.md` et `scripts/README.md`).

## Étape 7 — Extraire les prédictions build par build pour l’itération 5

Après l’exécution de la RQ1, exécuter :

```bash
py scripts/evaluation/extract_iter5_predictions_all_models.py
```

Ce script reconstruit, pour tous les modèles de la RQ1, les prédictions build par build sur le **fold 10** de l’**itération 5**. Les fichiers générés dans `results/qualitative_cases/` contiennent notamment :
- la durée réelle ;
- la prédiction du modèle ;
- la baseline lag-1 ;
- la baseline moyenne glissante sur les 7 dernières durées ;
- l’erreur absolue ;
- l’URL GitHub Actions du build.

Cette étape permet de reproduire l’analyse qualitative discutée dans le mémoire.

## Étape 8 — Recréer les figures et artefacts agrégés

Des scripts auxiliaires optionnels sont disponibles dans `scripts/plotting/`.
Ils ont été utilisés pour générer les figures du mémoire, comme le boxplot de la RQ1 et les comparaisons détaillées par modèle.

## Ordre d’exécution conseillé pour reproduire les artefacts du mémoire

1. baseline lag-1
2. modèles appris de la RQ1
3. analyse OLD vs RECENT de la RQ2
4. extraction build par build des prédictions de l’itération 5
5. génération des figures
6. agrégation de l’importance des variables
