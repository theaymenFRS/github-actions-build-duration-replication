# Dépannage

## 1) Un script ne trouve pas les fichiers de données

Les scripts principaux résolvent la racine du dépôt à partir de leur emplacement et attendent les CSV d’entrée dans `data/raw/`. Vérifiez que ces fichiers sont présents et que vous lancez Python depuis n’importe quel répertoire (le chemin ne dépend pas du dossier courant).

## 2) Les journaux de sortie sont écrits au mauvais endroit

Les scripts de baseline, RQ1 et RQ2 concernés écrivent sous `results/` (sous-dossiers dédiés). Si vous ne voyez rien à la racine du dépôt, consultez `results/`.

## 3) GHAminer n’est pas inclus

Ce package documente l’étape de collecte mais ne redistribue pas GHAminer. Utiliser le dépôt public indiqué dans `scripts/data_collection/README.md`.

## 4) Les versions exactes des bibliothèques sont inconnues

Les fichiers archivés révèlent les bibliothèques importées, mais pas un `pip freeze` complet ni un export conda exact. Les fichiers d’environnement fournis sont donc des recommandations reconstruites. Si l’environnement de la machine d’origine est encore disponible, remplacer ces fichiers par les versions exactes.

## 5) Le dépôt GitHub devient trop volumineux

Si vous prévoyez de publier ce package en dépôt public, envisager l’usage de Git LFS, Zenodo ou des assets de release pour les gros fichiers de données et de résultats.
