# results/

Ce répertoire contient les sorties expérimentales archivées.

## Organisation

- `rq1/<modèle>/` — journaux RQ1 regroupés par modèle, au format `<modèle>_<projet>_RQ1.log` (voir `scripts/README.md`).
- `rq1/baseline_lag1/<projet>/` — baseline lag-1 : `baseline_lag1.log` par projet (voir `scripts/evaluation/baseline_lag1.py`).
- `rq2/concept_drift_old_vs_recent_DEFAULTPARAMS/<projet>/` — sorties RQ2 (OLD vs RECENT) : `rq2_<projet>_RQ2.log`, `<projet>_processed.csv`, `rq2_<projet>_RQ2_GRID.png` ; journal agrégé `rq2_ALL_PROJECTS_RQ2.log` et figure globale `rq2_ALL_PROJECTS_RQ2_GRID.png` à la racine de ce dossier.
- Ancienne arborescence `by_project/` : **retirée du dépôt** après réorganisation ; le contenu équivalent est sous `rq1/`, `rq2/`, etc., comme ci-dessus.
- `aggregated/` — journaux agrégés et tableaux LaTeX utilisés dans le mémoire.
- `feature_importance/` — sorties CSV et figures liées à l’importance des variables.
- `qualitative_cases/` — fichiers CSV générés pour l’analyse qualitative build par build de l’itération 5.

Le dépôt conserve autant que possible les fichiers de résultats d’origine, tout en les réorganisant pour faciliter la navigation.
