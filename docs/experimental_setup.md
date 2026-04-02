# Configuration expérimentale

Ce fichier résume la configuration expérimentale reconstruite à partir des fichiers archivés du projet.

## Environnement de développement

- IDE utilisé par l’auteur : **PyCharm Professional 2024.3.1.1** (licence ETS)
- Outil de dessin de la figure du pipeline : **Lucidchart**

Ni PyCharm ni Lucidchart ne sont nécessaires pour réexécuter les expériences.

## Version Python recommandée

Le projet archivé contient des fichiers `__pycache__` pour **CPython 3.12** ; Python 3.12 est donc recommandé.

## Bibliothèques Python principales importées dans les scripts archivés

- pandas
- numpy
- scikit-learn
- matplotlib
- scipy
- xgboost
- lightgbm

## Choix méthodologiques principaux reconstruits à partir des scripts

### RQ1

- 10 folds chronologiques (`N_FOLDS = 10`)
- 5 itérations en fenêtre croissante (`N_ITERS = 5`)
- baseline lag-1 (`baseline_lag1.py`)
- une autre baseline (moyenne des 7 dernières durées) intégrée aux journaux de la RQ1
- seed GA fixée à `42` dans les scripts de la RQ1
- borne supérieure sur la durée : `300 * 24 * 60 * 60` secondes

### RQ2

- 10 folds chronologiques
- 4 comparaisons OLD vs RECENT (`N_ITERS = 4`)
- mêmes fichiers de workflows traités en entrée
- comparaison statistique fondée sur Mann–Whitney U dans le script archivé

## Correspondance avec le pipeline de données

- `data/raw/` — CSV d’entrée des scripts de modélisation (workflows retenus)
- `data/processed/` — jeux dérivés ou prétraités selon le dépôt
- `results/` — journaux, figures et sorties générées par les scripts
- `figures/` — figures du mémoire
