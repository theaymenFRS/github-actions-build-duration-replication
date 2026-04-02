# results/qualitative_cases/

Ce dossier contient les prédictions build par build reconstruites pour l’itération 5 de la RQ1.

Génération : depuis la racine du dépôt, après les scripts RQ1, exécuter `py scripts/evaluation/extract_iter5_predictions_all_models.py` (voir `docs/replication_steps.md`).

Les fichiers générés permettent de comparer, pour chaque build du fold 10 :
- la durée réelle ;
- la prédiction du modèle ;
- la baseline lag-1 ;
- la baseline moyenne glissante sur les 7 dernières durées ;
- les erreurs absolues ;
- l’URL GitHub Actions correspondante.

Ces fichiers servent à documenter l’analyse qualitative présentée dans le chapitre 4 du mémoire.
