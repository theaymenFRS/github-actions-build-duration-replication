# preprocessing/

Il n’y a pas d’exécutable “prétraitement seul” : le flux complet reste dans les scripts de modélisation, mais la **logique de prétraitement** est centralisée dans **`scripts/preprocessing/preprocess_common.py`** (`preprocess_data`, `verbose=True` pour la RQ1 et `verbose=False` pour la RQ2). Les scripts RQ1/RQ2 importent ce module puis enchaînent entraînement / évaluation.

À haut niveau, l’étape de prétraitement comprend :
- le filtrage vers les builds réussis pour la tâche de régression ;
- le tri chronologique ;
- les vérifications de cohérence sur les durées et le filtrage par borne supérieure ;
- le filtrage des événements de workflow ;
- les vérifications d’historique minimal.

## Où est le code de prétraitement ?

- **Partagé** : `scripts/preprocessing/preprocess_common.py` — `preprocess_data(file_path, output_path=..., verbose=...)`.
- **RQ1** : `scripts/modeling/rq1/*.py` importent ce module (`verbose=True` par défaut).
- **RQ2** : `scripts/modeling/rq2/rq2_old_vs_recent.py` (`verbose=False`).

## Avant le prétraitement (collecte)

Si vous recollectez les données avec **GHAminer**, vous pouvez **filtrer les exécutions par `workflow_id`** côté outil pour ne garder que le(s) workflow(s) ciblé(s). Ce n’est pas une étape du code Python de ce dépôt ; voir `scripts/data_collection/README.md`.

## Entrées / sorties

- **Entrées CSV** : `data/raw/` (les scripts lisent directement les CSV depuis ce dossier).
- **Sortie CSV “processed”** : chaque script RQ1 sauvegarde une copie prétraitée sous `data/processed/` (utile pour inspection et traçabilité).
- **Autres sorties** : journaux/figures sous `results/`.

Voir aussi `docs/replication_steps.md` et `docs/experimental_setup.md`.
