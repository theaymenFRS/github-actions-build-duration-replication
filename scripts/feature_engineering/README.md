# feature_engineering/

Ce répertoire contient les scripts d’importance des variables archivés avec le projet.

L’ingénierie des caractéristiques principale utilisée dans le mémoire est implémentée dans les scripts de modélisation et comprend notamment :
- des variables de retard (`duration_lag_k`) ;
- des statistiques glissantes ;
- l’encodage des variables catégorielles ;
- l’expansion des types de fichiers ;
- une sélection de variables visant à réduire la redondance.

Les scripts d’importance des variables ont été utilisés pour agréger et visualiser les scores d’importance après l’entraînement des modèles.

## Données d’entrée attendues

Les scripts `compute_feature_importance_iter5.py` et apparentés lisent les jeux prétraités sous `results/rq2/concept_drift_old_vs_recent_DEFAULTPARAMS/<projet>/` et les journaux RQ1 sous `results/rq1/<modèle>/` (voir `scripts/preprocessing/README.md` et `results/README.md`).
