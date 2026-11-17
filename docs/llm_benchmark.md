# Plafond LLM : classification des tags par Gemini (zero-shot)

## Objectif

Estimer une performance de référence « haute » avec un LLM généraliste, sans entraînement, et la comparer au meilleur
modèle entraîné du projet : le stacking embeddings + difficulté + TF-IDF de la description (notebook 04).

## Mise en place

| Élément | Choix |
|---|---|
| Plateforme | Vertex AI, projet GCP **dédié** `illuin-codeforces-tags` (séparé des projets existants) |
| Authentification | identifiants par défaut de `gcloud` (ADC) : aucune clé stockée. Le projet de quota est forcé sur le projet dédié pour que l'usage n'y soit pas mélangé à un autre projet. |
| Modèle | `gemini-3.1-flash-lite` : le moins cher accessible depuis le projet (les modèles 2.5 renvoient 404 pour un projet récent) |
| Coût estimé | environ 730 000 tokens d'entrée pour le test, soit environ 0,25 $ (tarifs d'agrégateurs, non vérifiés sur la page officielle) |
| Code | [`src/llm_tagger.py`](../src/llm_tagger.py) ; nettoyage et split lus dans [`configs/stacking.yaml`](../configs/stacking.yaml), comme pour le modèle entraîné ; résultats lus dans [`notebooks_prod/04_multimodal_and_results.ipynb`](../notebooks_prod/04_multimodal_and_results.ipynb) |

Mise en place du projet GCP (une seule fois) :

```bash
gcloud projects create illuin-codeforces-tags
gcloud billing projects link illuin-codeforces-tags --billing-account=<ID>
gcloud services enable aiplatform.googleapis.com --project=illuin-codeforces-tags
gcloud auth application-default login
```

Prédictions sur le split test :

```bash
export GOOGLE_CLOUD_PROJECT=illuin-codeforces-tags
python -m src.llm_tagger --limit 5                          # essai
python -m src.llm_tagger --model gemini-3.1-flash-lite      # cache data/llm/*.jsonl, reprise auto
```

**Prompt** (en anglais, comme les énoncés) :
- définition courte des 8 tags cibles, et consigne de n'attribuer un tag que s'il fait clairement partie de la solution ;
- **les mêmes colonnes que le meilleur modèle** : énoncé, spécification d'entrée, notes, difficulté, et code source tronqué à
  6 000 caractères.

**Sortie structurée** : JSON contraint par un schéma `{"tags": [...], "justification": "..."}`, où les valeurs de `tags` sont
limitées aux 8 tags cibles (une liste vide est autorisée). `temperature=0`.

## Protocole

- **Données** : les **726 exercices du test** uniquement, avec le même nettoyage et le même découpage que tous les autres modèles.
- **Zero-shot** : aucun exemple annoté dans le prompt, aucun réglage. Contrairement aux modèles entraînés, le LLM ne bénéficie
  d'**aucun seuil réglé sur la validation**.
- **Références ré-entraînées à l'identique** (entraînement sur train, réglages sur val) : regex TF-IDF, régression logistique
  sur la description, stacking (meilleur modèle).
- **Couverture** : 726 réponses valides sur 726. Quelques erreurs temporaires de l'API (quota) ont été rattrapées par les
  nouvelles tentatives.

## Résultats

### Métriques globales (test, 726 exercices)

| Modèle | Micro-F1 | Macro-F1 | Hamming loss ↓ | Jaccard |
|---|---|---|---|---|
| Regex TF-IDF (sans apprentissage) | 0.454 | 0.518 | 0.100 | 0.494 |
| Régression logistique (description seule) | 0.554 | 0.612 | 0.083 | 0.532 |
| **Stacking (meilleur modèle entraîné)** | 0.627 | 0.647 | 0.072 | **0.600** |
| **LLM `gemini-3.1-flash-lite` (zero-shot)** | **0.674** | **0.753** | **0.067** | 0.581 |

**Écart de Macro-F1** (bootstrap apparié, 1 000 tirages) :

| Comparaison | Écart | IC 95 % |
|---|---|---|
| LLM − stacking | **+0,108** | [+0,070 ; +0,151] |
| LLM − régression logistique | +0,143 | [+0,099 ; +0,188] |
| LLM − regex TF-IDF | +0,236 | [+0,195 ; +0,278] |

### Par tag

| Tag (support test) | LLM P / R / **F1** | Stacking P / R / **F1** |
|---|---|---|
| math (207) | 0.48 / 0.82 / **0.60** | 0.57 / 0.71 / **0.63** |
| graphs (80) | 0.61 / 0.79 / **0.68** | 0.52 / 0.60 / **0.55** |
| strings (61) | 0.57 / 0.92 / **0.70** | 0.56 / 0.80 / **0.66** |
| number theory (51) | 0.81 / 0.67 / **0.73** | 0.55 / 0.65 / **0.59** |
| trees (46) | 0.80 / 0.89 / **0.85** | 0.67 / 0.70 / **0.68** |
| geometry (24) | 0.70 / 0.58 / **0.64** | 0.46 / 0.88 / **0.60** |
| games (15) | 0.88 / 0.93 / **0.90** | 0.72 / 0.87 / **0.79** |
| probabilities (14) | 1.00 / 0.86 / **0.92** | 0.62 / 0.71 / **0.67** |

## Analyse

- **Le LLM est le meilleur modèle sur 7 tags sur 8**, avec +10,8 points de Macro-F1 sur le stacking, un écart significatif.
  Les plus gros gains portent sur les tags **conceptuels ou rares** :
  - `probabilities` : F1 de 0.67 à 0.92, avec une précision de 1.00 ;
  - `trees` : de 0.68 à 0.85 ;
  - `number theory` : de 0.59 à 0.73 ;
  - `games` : de 0.79 à 0.90.

  Le LLM *comprend* la technique attendue, là où les modèles entraînés n'en voient que des indices lexicaux ou statistiques.
- **`math` est la seule exception** : F1 de 0.60 contre 0.63 pour le stacking. Le LLM surévalue ce tag, avec **187 faux positifs**
  pour 207 positifs réels. Il assigne `math` dès qu'il y a un raisonnement mathématique (inégalité triangulaire, comptage par
  coefficients binomiaux, opérations bit à bit), alors que l'annotation Codeforces de `math` est plus restrictive et peu cohérente.
  Sur les désaccords concernant `math`, le stacking a raison 110 fois contre 58 : il a **appris la convention d'annotation**,
  ce que le zero-shot ne peut pas faire.
- **Le LLM prédit trop de tags par exercice** : seulement 19 % de réponses vides, contre 47 % dans la vérité, et 0,97 tag par
  exercice en moyenne contre 0,69. C'est pourquoi son **Jaccard est inférieur** à celui du stacking (0.581 contre 0.600) malgré
  un meilleur F1 : le Jaccard récompense le fait de répondre « aucun tag » sur les nombreux exercices qui n'en ont pas.
- **Combinaison naïve** : l'union des prédictions du LLM et du stacking fait moins bien que le LLM seul (Macro-F1 0.688), car elle
  cumule les faux positifs des deux. Une combinaison utile demanderait un arbitrage par tag (par exemple, le stacking pour `math`).

## Limites

- **Contamination probable** : Codeforces est public et ancien. Gemini a très probablement vu ces énoncés, et parfois leurs tags,
  à l'entraînement. Une partie du score peut venir de la **mémorisation**, pas seulement du raisonnement : c'est un plafond
  **optimiste**, à présenter comme tel.
- **Pas de probabilités** : la sortie est un ensemble de tags, sans score de confiance. Il n'y a donc ni réglage de seuil par tag
  (qui corrigerait l'excès de `math`) ni métrique de rang (Macro-AP).
- **Coût et dépendance** : chaque prédiction est un appel réseau payant, soumis au quota et à la disponibilité du modèle. Les
  modèles entraînés tournent en local, sans coût marginal (61 ms par exemple). La latence par appel du LLM n'a pas été mesurée.
- **Modèle le plus économique** : un modèle plus puissant (flash, pro ou Gemini 4) ferait probablement encore mieux. Un prompt
  few-shot, avec des exemples annotés du train, pourrait aussi aligner le LLM sur la convention d'annotation (`math`).

## Conclusion

Un LLM généraliste en zero-shot dépasse le meilleur modèle entraîné : Macro-F1 de **0.753 contre 0.647**. Il excelle sur les tags
qui demandent de comprendre la technique de résolution (`probabilities`, `trees`, `number theory`, `games`). Il reste en
revanche moins bon sur `math`, dont la convention d'annotation s'apprend des données, et il répond trop rarement « aucun tag ».
Ce résultat est un **plafond optimiste** (contamination probable) : il situe la marge de progression du modèle entraîné, sans
remettre en cause son intérêt (coût nul, local, rapide, déterministe, sortie probabiliste réglable par tag).
