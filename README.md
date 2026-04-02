Prédiction de la durée de build dans GitHub Actions

**prédiction de la durée de build dans GitHub Actions** à partir de données historiques de workflows, de caractéristiques temporelles, de plusieurs modèles de régression et d’une analyse OLD vs RECENT liée à la dérive de concept.

## Objectif

L’objectif est de reproduire les analyses principales :

1. **Collecte des données** avec l’outil externe GHAminer
2. **Prétraitement** et filtrage des workflows pertinents (build,CI,CD...ect.)
3. **Ingénierie des caractéristiques** et sélection/réduction des variables
4. **Entraînement des modèles pour la RQ1**
5. **Génération des baselines**
6. **Évaluation OLD vs RECENT pour la RQ2**
7. **Génération des figures** et agrégation des résultats

## Structure du dépôt

- `data/` — données brutes, données traitées et liens publics vers les projets étudiés.
- `scripts/` — scripts d’origine réorganisés selon les étapes du pipeline (détail : `scripts/README.md`).
- `results/` — sorties expérimentales (RQ1 par modèle, baseline lag-1, RQ2, importance des variables, cas qualitatifs ; détail dans `results/README.md`).
- `docs/` — documentation détaillée : étapes de réplication, configuration expérimentale, description des projets et dépannage.
- `figures/` — figure du pipeline et figures/résultats utilisés dans le mémoire.
- `requirements.txt` / `environment.yml` — environnement logiciel recommandé.
- `LICENSE` — MIT

## Remarques importantes

- L’outil de collecte d’origine est **GHAminer** et **n’est pas inclus** dans ce dépôt. Voir `scripts/data_collection/README.md`.
- Le code original a été développé avec **PyCharm Professional 2024.3.1.1**. PyCharm n’est **pas nécessaire** pour reproduire les expériences.
- **Lucidchart** a été utilisé pour dessiner la figure du pipeline. Lucidchart n’est **pas nécessaire** pour la reproduction.
- Les scripts principaux du pipeline utilisent des **chemins relatifs au dépôt** : la racine est détectée automatiquement, les CSV d’entrée sont lus sous `data/raw/`, et les journaux et sorties sont écrits sous `results/`. Aucune configuration manuelle de répertoire absolu n’est nécessaire.
- Le package conserve les **scripts originaux archivés**, avec une documentation indiquant l’ordre d’exécution recommandé.
- Les versions exactes des bibliothèques n’étaient pas archivées dans le projet d’origine. Les fichiers d’environnement fournis sont donc une **reconstruction raisonnable** fondée sur les bibliothèques importées et sur la présence de bytecode **CPython 3.12** dans les fichiers archivés.

## Démarrage rapide

```bash
# 1) créer l’environnement
conda env create -f environment.yml
conda activate gha-build-duration

# ou
pip install -r requirements.txt

# 2) consulter l’ordre d’exécution
#    docs/replication_steps.md

# 3) exécuter les scripts : entrées CSV sous data/raw/, sorties sous results/
```

## Ordre principal de reproduction

- **Étape 1** : lire `docs/projects_description.md` et `data/project_links.csv`
- **Étape 2** : consulter `docs/experimental_setup.md`
- **Étape 3** : si nécessaire, recollecter les données avec GHAminer
- **Étape 4** : utiliser les fichiers CSV d’entrée dans `data/raw/`
- **Étape 5** : exécuter le script de baseline lag-1
- **Étape 6** : exécuter les cinq scripts de la RQ1
- **Étape 7** : exécuter le script RQ2 OLD vs RECENT
- **Étape 8** : générer les figures et agréger les tableaux si nécessaire

## Recommandation de citation

ETS ESPACE
