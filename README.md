# Classification d'exercices d'algorithmique

Prédire les tags d'un exercice Codeforces parmi `math`, `graphs`, `strings`, `number theory`, `trees`, `geometry`,
`games` et `probabilities`, à partir de l'énoncé, de la solution Python et de la difficulté. Le sujet complet est
dans [sujet.md](sujet.md).

Le modèle retenu est un stacking : une régression logistique par source (embeddings de l'énoncé, de la spécification
d'entrée, des notes et du code, difficulté, TF-IDF de l'énoncé), puis une régression logistique par tag qui combine
leurs sorties. Un seuil de décision est réglé pour chaque tag.

Sur le jeu de test (726 exercices) :

| Modèle | Macro-F1 | Micro-F1 |
|---|---|---|
| Regex à mots-clés (plancher) | 0,518 | 0,454 |
| Régression logistique sur l'énoncé seul | 0,612 | 0,554 |
| **Stacking (modèle retenu)** | **0,647** | **0,627** |
| LLM Gemini zero-shot (référence haute) | 0,753 | 0,674 |

Une prédiction prend environ 65 ms par exercice sur un portable.

## Organisation

```
src/               code de production : nettoyage, features, modèle, entraînement, CLI
configs/           hyperparamètres du modèle (stacking.yaml)
notebooks_prod/    exploration et expériences, du nettoyage aux résultats finaux
tests/             tests pytest
data/              dataset, caches d'embeddings, prédictions LLM (non versionné)
models/            modèle entraîné (non versionné)
.venv/             environnement virtuel (non versionné, voir Installation)
```

Les notebooks racontent la démarche et contiennent tout le code des expériences, y compris celles qui n'ont pas été
retenues (regex, MLP, concaténation). Ils ne dépendent pas de `src/`. À lire dans l'ordre :

1. `01_eda_and_cleaning` : exploration, filtre de langue, suppression des quasi-doublons, split train / val / test.
   Écrit `data/processed/clean.parquet`, utilisé par les suivants.
2. `02_feature_engineering` : pour chaque source (énoncé, spécification d'entrée, notes, code, difficulté,
   métadonnées), les représentations comparées et celle retenue, puis une ablation pour choisir les sources.
3. `03_models_benchmark` : choix du modèle. D'abord regex, régression logistique et MLP sur l'énoncé seul, puis
   concaténation, stacking et MLP sur toutes les sources retenues.
4. `04_results` : évaluation du modèle retenu sur le test, comparaison aux alternatives et à un LLM, analyse
   d'erreurs et latence.

`src/` ne contient que le modèle retenu :

| Fichier | Rôle |
|---|---|
| `preprocessing.py` | normalisation des textes, filtre de langue, quasi-doublons |
| `data.py` | lecture des fichiers JSON, labels, split stratifié |
| `features.py` | encodeurs Sentence-BERT et UniXcoder, cache des embeddings, encodage de la difficulté |
| `model.py` | stacking, seuils par tag, sauvegarde |
| `train.py`, `predict.py`, `evaluate.py` | entraînement, prédiction, métriques |
| `cli.py` | ligne de commande |
| `config.py` | chemins, tags, lecture de `configs/stacking.yaml` |
| `llm_tagger.py` | prédictions du LLM Gemini (facultatif) |

## Installation

L'environnement virtuel n'est pas versionné (`.venv/` est dans `.gitignore`) : il se reconstruit à partir de
`requirements.txt`. Il faut Python 3.10 ou plus récent (le projet a été testé avec 3.13).

```bash
python3 -m venv .venv                  # crée l'environnement dans .venv/
source .venv/bin/activate              # sous Windows : .venv\Scripts\activate
pip install -r requirements.txt
python -m nltk.downloader wordnet      # dictionnaire utilisé pour la lemmatisation
```

`requirements.txt` regroupe les dépendances du code de production, celles des notebooks et les outils de test.
L'installation télécharge PyTorch et prend quelques minutes.

### Données

Le dataset (4 982 exercices) n'est pas versionné. Pour le télécharger et l'extraire dans `data/` :

```bash
mkdir -p data
gdown 1FyNPiEKkZBfhz4ua0eM3PQjbRWVwO95J -O data/dataset.zip
unzip data/dataset.zip -d data && rm data/dataset.zip
```

On doit obtenir `data/code_classification_dataset/sample_0.json`, `sample_1.json`, etc. Le notebook 01 fait la même
chose s'il ne trouve pas les données.

## Utilisation

Toutes les commandes se lancent depuis la racine du projet, avec le venv activé.

### Entraîner

```bash
python -m src.cli train
```

La commande nettoie le dataset, le découpe en train / val / test (70 / 15 / 15 %), entraîne le stacking sur train,
règle les seuils sur val, affiche les métriques sur test et sauvegarde le modèle dans `models/stacking.joblib`.

Le premier lancement télécharge les deux encodeurs depuis Hugging Face (environ 1 Go) et calcule les embeddings des
4 834 exercices, ce qui prend plusieurs minutes. Les embeddings sont ensuite gardés en cache dans `data/embeddings/`
et les entraînements suivants prennent une quinzaine de secondes.

Les hyperparamètres sont dans [configs/stacking.yaml](configs/stacking.yaml), avec en commentaire le notebook d'où
vient chaque valeur. Pour entraîner avec un autre fichier :

```bash
python -m src.cli train --config configs/mon_essai.yaml
```

### Prédire

```bash
python -m src.cli predict data/code_classification_dataset/sample_40.json
python -m src.cli predict chemin/vers/un/dossier/ --output predictions.jsonl
```

L'entrée est un fichier JSON au format du dataset, ou un dossier de fichiers `sample_*.json`. Le nom de fichier doit
se terminer par `_<numéro>.json` : ce numéro sert d'identifiant dans la sortie. Seuls `prob_desc_description` et
`source_code` sont obligatoires ; `prob_desc_input_spec`, `prob_desc_notes` et `difficulty` peuvent manquer.

La sortie contient une ligne JSON par exercice :

```json
{"sample_id": 40, "tags": ["strings"]}
```

### Évaluer

```bash
python -m src.cli evaluate                        # sur le split test du dataset
python -m src.cli evaluate chemin/vers/un/dossier/  # sur des exercices annotés (champ "tags")
```

La commande affiche le Micro-F1, le Macro-F1, la Hamming loss et le Jaccard, puis la précision, le rappel et le F1
de chaque tag, ainsi que la latence médiane et maximale par exercice.

### Lancer les tests

```bash
pytest
```

Les tests couvrent le nettoyage, le split, la lecture de la config, le modèle (sur des données synthétiques) et le
format de sortie. Le test de la CLI est ignoré tant que le modèle n'a pas été entraîné.

Pour vérifier le style du code :

```bash
ruff check src tests
ruff format --check src tests
```
