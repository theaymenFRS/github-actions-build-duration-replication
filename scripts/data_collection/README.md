# data_collection/

L’étape de collecte des données repose sur l’outil externe **GHAminer** :

- Dépôt : https://github.com/stilab-ets/GHAminer

Ce package de réplication **ne redistribue pas** GHAminer lui-même. Les fichiers bruts des projets sélectionnés sont déjà fournis sous `data/raw/`, donc la collecte est optionnelle sauf si vous souhaitez reconstruire complètement le jeu de données.

Si vous recollectez les données, veiller à :
1. utiliser les dépôts GitHub publics listés dans `data/project_links.csv` ;
2. appliquer la même logique de sélection des workflows que dans le mémoire ;
3. lorsque c’est pertinent, **filtrer les exécutions par `workflow_id`** directement côté GHAminer (afin de ne conserver que le(s) workflow(s) ciblé(s) pour l’analyse) ;
4. reproduire l’étape de correction et de filtrage, puis aligner les sorties sur la structure attendue par `data/` (voir `data/README.md`).

Après collecte, le prétraitement côté Python des expériences est décrit dans `scripts/preprocessing/README.md`.
