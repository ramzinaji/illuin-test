# Métriques d'évaluation

Ce document décrit toutes les métriques utilisées pour évaluer et comparer les modèles du projet : ce qu'elles mesurent,
leur formule, leur fonctionnement et leurs pièges. Le code de production est dans [`src/evaluate.py`](../src/evaluate.py) (métriques) et
[`src/model.py`](../src/model.py) (seuils par tag) ; Macro-AP, validation croisée et bootstrap sont dans
[`notebooks_prod/03_models_benchmark.ipynb`](../notebooks_prod/03_models_benchmark.ipynb).

## 1. Cadre : une classification multi-label

Chaque exercice peut porter **0, 1 ou plusieurs** des 8 tags cibles (`math`, `graphs`, `strings`, `number theory`, `trees`,
`geometry`, `games`, `probabilities`). Le problème se ramène à **8 décisions binaires indépendantes** par exercice.

Notations :
- $N$ : nombre d'exercices évalués (726 sur le test) ; $L = 8$ : nombre de tags.
- $y_{i\ell} \in \{0, 1\}$ : vérité (l'exercice $i$ porte-t-il le tag $\ell$ ?) ; $\hat{y}_{i\ell} \in \{0, 1\}$ : prédiction.
- $Y_i$ et $\hat{Y}_i$ : ensembles des tags vrais et prédits de l'exercice $i$.

Pour chaque tag $\ell$, on compte sur tous les exercices :

| | Prédit $\hat{y}=1$ | Prédit $\hat{y}=0$ |
|---|---|---|
| **Vrai $y=1$** | $TP_\ell$ (vrai positif) | $FN_\ell$ (faux négatif, oubli) |
| **Vrai $y=0$** | $FP_\ell$ (faux positif, fausse alarme) | $TN_\ell$ (vrai négatif) |

**Particularité des données** : la matrice est très creuse. En moyenne, un exercice porte 0,69 tag cible sur 8, et 46 % des
exercices n'en ont aucun. Les tags sont très déséquilibrés : `math` concerne 28 % des exercices, `probabilities` 1,9 %.
Ces deux points déterminent le choix et la lecture des métriques.

## 2. Métriques par tag

### Précision

$$\text{Précision}_\ell = \frac{TP_\ell}{TP_\ell + FP_\ell}$$

**Ce qu'elle mesure** : parmi les exercices auxquels le modèle attribue le tag, la part qui le porte vraiment.
Une précision faible signifie beaucoup de fausses alarmes.

### Rappel

$$\text{Rappel}_\ell = \frac{TP_\ell}{TP_\ell + FN_\ell}$$

**Ce qu'il mesure** : parmi les exercices qui portent vraiment le tag, la part que le modèle retrouve.
Un rappel faible signifie beaucoup d'oublis.

### F1

$$F1_\ell = 2 \cdot \frac{\text{Précision}_\ell \cdot \text{Rappel}_\ell}{\text{Précision}_\ell + \text{Rappel}_\ell} = \frac{2\,TP_\ell}{2\,TP_\ell + FP_\ell + FN_\ell}$$

**Ce qu'il mesure** : la moyenne **harmonique** de la précision et du rappel. Contrairement à une moyenne arithmétique, elle
reste basse si l'un des deux est bas : un modèle ne peut pas compenser un rappel nul par une précision parfaite.
Les vrais négatifs n'interviennent pas, ce qui est voulu : sur un tag rare, ils sont nombreux et triviaux à obtenir.

**Support** : $TP_\ell + FN_\ell$, le nombre d'exercices qui portent réellement le tag. Un F1 calculé sur 14 exemples
(`probabilities` en test) varie d'environ ±0,1 à un seul exemple près.

## 3. Métriques globales

### Micro-F1

On additionne d'abord les comptes de tous les tags, puis on calcule un seul F1 :

$$TP = \sum_\ell TP_\ell,\quad FP = \sum_\ell FP_\ell,\quad FN = \sum_\ell FN_\ell \qquad\Rightarrow\qquad \text{Micro-F1} = \frac{2\,TP}{2\,TP + FP + FN}$$

**Ce qu'il mesure** : la qualité globale des décisions (exercice, tag). Chaque décision pèse autant, donc **les tags fréquents
dominent** : `math` (207 positifs en test) pèse environ 15 fois plus que `probabilities` (14).

### Macro-F1 (métrique principale du projet)

$$\text{Macro-F1} = \frac{1}{L} \sum_{\ell=1}^{L} F1_\ell$$

**Ce qu'il mesure** : la qualité **moyenne par tag**, chaque tag comptant autant quelle que soit sa fréquence. Un modèle qui
néglige les tags rares est fortement pénalisé.

**Pourquoi c'est la métrique principale** : le sujet demande quels tags sont les mieux prédits. Le Macro-F1 est la moyenne
exacte des F1 par tag, il résume donc directement cette question. Un écart Macro > Micro indique que les tags rares sont mieux
prédits que les tags fréquents. C'est le cas ici, parce que `math` est à la fois le plus fréquent et le plus difficile.

### Hamming loss

$$\text{Hamming} = \frac{1}{N \cdot L} \sum_{i=1}^{N} \sum_{\ell=1}^{L} \mathbb{1}\left[\hat{y}_{i\ell} \neq y_{i\ell}\right] = \frac{FP + FN}{N \cdot L}$$

**Ce qu'elle mesure** : la proportion de cases (exercice, tag) mal prédites. Plus elle est basse, mieux c'est.

**Piège** : sur une matrice aussi creuse, un modèle qui **ne prédit jamais rien** obtient déjà environ
$0{,}69 / 8 \approx 0{,}085$, un score proche de celui de nos bons modèles. La Hamming loss ne se lit donc jamais seule. Elle
sert surtout à repérer un modèle qui prédit **trop** de tags, car chaque faux positif l'augmente.

### Indice de Jaccard (moyenne par exercice)

$$J_i = \frac{|\hat{Y}_i \cap Y_i|}{|\hat{Y}_i \cup Y_i|}, \qquad \text{Jaccard} = \frac{1}{N} \sum_{i=1}^{N} J_i$$

**Ce qu'il mesure** : pour chaque exercice, le recouvrement entre l'ensemble des tags prédits et l'ensemble des tags vrais,
puis la moyenne sur les exercices. C'est le point de vue d'un utilisateur : « la liste proposée pour cet exercice est-elle
la bonne ? »

**Exemple** : vrai = {math, number theory}, prédit = {math, geometry}. Intersection = 1, union = 3, d'où $J_i = 0{,}33$.

**Convention** : si un exercice n'a aucun tag vrai et qu'aucun n'est prédit, l'union est vide. On compte alors $J_i = 1$
(prédiction parfaite), avec `zero_division=1`. Ce choix pèse beaucoup ici, puisque 46 % des exercices n'ont aucun tag cible.
Le Jaccard **récompense un modèle qui sait répondre « aucun tag »**. C'est pourquoi le LLM, qui répond trop rarement
« aucun tag », a un meilleur F1 mais un moins bon Jaccard que le stacking.

## 4. Métrique de sélection : Macro-AP (sans seuil)

Les métriques précédentes s'appliquent à des décisions 0/1, donc **après** le choix d'un seuil. Pour **choisir entre des
modèles ou des hyperparamètres**, on utilise une métrique qui ne dépend d'aucun seuil, calculée sur les probabilités.

**Average Precision (AP) d'un tag** : on trie les exercices par probabilité décroissante, puis on parcourt la liste en
abaissant progressivement le seuil. À chaque nouveau vrai positif rencontré, on relève la précision obtenue :

$$AP_\ell = \sum_{k} \left(R_k - R_{k-1}\right) P_k$$

où $P_k$ et $R_k$ sont la précision et le rappel au $k$-ième seuil. C'est l'**aire sous la courbe précision-rappel**.

$$\text{Macro-AP} = \frac{1}{L} \sum_{\ell=1}^{L} AP_\ell$$

**Ce qu'elle mesure** : la capacité du modèle à **classer** les exercices positifs avant les négatifs, tous seuils confondus.
Une AP de 1 signifie que tous les positifs ont une probabilité supérieure à celle de tous les négatifs. Un classement aléatoire
donne une AP environ égale à la prévalence du tag.

**Pourquoi l'utiliser pour la sélection** : elle évalue le modèle indépendamment du réglage des seuils, qui est bruité sur
les tags rares. Comparer deux modèles en Macro-F1 mélangerait leur qualité et la qualité de leurs seuils.

## 5. Des probabilités aux décisions : les seuils par tag

Les modèles produisent une probabilité $p_{i\ell}$. La décision est $\hat{y}_{i\ell} = \mathbb{1}[p_{i\ell} \geq s_\ell]$, avec un
**seuil propre à chaque tag** :

$$s_\ell = \arg\max_{s \in \{0{,}05;\ 0{,}06;\ \dots;\ 0{,}95\}} F1_\ell(s) \quad \text{calculé sur la validation}$$

**Pourquoi un seuil par tag** : avec la pondération `class_weight="balanced"`, les probabilités ne sont pas calibrées de la
même façon d'un tag à l'autre, et un seuil unique de 0,5 serait sous-optimal pour les tags rares.
**Pourquoi sur la validation** : régler les seuils sur le test reviendrait à évaluer le modèle sur des données qui ont servi
à le régler, ce qui gonflerait les scores.

La grille (`start`, `stop`, `step`) est définie dans la section `thresholds` de
[`configs/stacking.yaml`](../configs/stacking.yaml). Les seuils appris sont sauvegardés avec le modèle
(`models/stacking.joblib`) et recalculés à chaque `python -m src.cli train`.

## 6. Plancher de référence : la stratégie « toujours oui »

Prédire un tag pour **tous** les exercices donne un rappel de 1 et une précision égale à la prévalence $p_\ell$ du tag :

$$F1_\ell^{\text{trivial}} = \frac{2\,p_\ell}{1 + p_\ell}$$

**Ce qu'il mesure** : le F1 qu'on obtient sans aucune intelligence. Un F1 se juge **par rapport à cette valeur** : un F1 de 0,44
sur `math` (prévalence 28 %) ne vaut rien, alors qu'un F1 de 0,57 sur `graphs` (prévalence 11 %, plancher 0,20) est déjà
presque 3 fois meilleur que le hasard. Sur les 8 tags, cette stratégie obtient un Macro-F1 de 0,149 et une Hamming loss de 0,914.

## 7. Fiabilité des comparaisons

### Validation croisée stratifiée multi-label

La sélection des modèles se fait par **validation croisée à 5 folds** sur train + val. Le jeu est découpé en 5 parts qui
conservent chacune la prévalence de chaque tag (`MultilabelStratifiedKFold`, stratification itérative). Chaque part sert
une fois d'évaluation, et le modèle est entraîné sur les 4 autres. On rapporte la **moyenne ± l'écart-type** de la Macro-AP
sur les 5 folds.

**Lecture** : un écart entre deux modèles nettement plus petit que l'écart-type entre folds (environ 0,02 à 0,03 ici)
n'est pas concluant.

### Bootstrap apparié sur le test

Le test ne compte que 726 exercices, donc un écart de quelques points peut être dû au hasard du découpage. Pour deux modèles
A et B, la fonction `bootstrap_diff` (notebooks 03 et 04) procède ainsi :
1. tirer avec remise 726 exercices parmi les 726 du test ;
2. calculer $\Delta = \text{Macro-F1}(A) - \text{Macro-F1}(B)$ **sur ces mêmes exercices** (d'où « apparié ») ;
3. répéter 1 000 fois.

On rapporte l'écart moyen, l'**intervalle de confiance à 95 %** (percentiles 2,5 et 97,5 des 1 000 écarts) et la part des tirages
où A bat B.

**Lecture** :
- **L'intervalle exclut 0** : l'écart est significatif. Par exemple, LLM contre stacking : +0,108, IC [+0,070 ; +0,151].
- **L'intervalle contient 0** : on ne peut pas conclure. Par exemple, stacking avec `tfidf_desc` contre stacking sans :
  +0,001, IC [-0,023 ; +0,028].

**Limite** : le bootstrap mesure le hasard **du test**, pas la variance de l'entraînement (graine, seuils, découpage
train/val). C'est pourquoi il complète la validation croisée sans la remplacer.

## 8. Récapitulatif

| Métrique | Unité de calcul | Sens | Favorise | Usage dans le projet |
|---|---|---|---|---|
| Précision / Rappel / F1 par tag | tag | ↑ | — | quels tags sont les mieux prédits |
| Micro-F1 | décision (exercice, tag) | ↑ | les tags fréquents | qualité globale |
| **Macro-F1** | tag | ↑ | l'équité entre tags | **métrique principale** sur le test |
| Hamming loss | case de la matrice | ↓ | les modèles qui prédisent peu | détecter le sur-étiquetage |
| Jaccard | exercice | ↑ | savoir répondre « aucun tag » | qualité perçue par exercice |
| Macro-AP | tag, sans seuil | ↑ | le bon classement des probabilités | sélection en validation croisée |
| F1 « toujours oui » | tag | — | — | plancher de référence |
| IC bootstrap | différence de Macro-F1 | — | — | significativité d'un écart sur le test |
