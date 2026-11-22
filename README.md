# Classification de Questions par Séparateurs à Vaste Marge (SVM) et Noyaux d'Arbres

Ce projet de Master 2 en Traitement Automatique du Langage Naturel (NLP) reproduit et évalue les contributions de l'article de référence :
> **Dell Zhang & Wee Sun Lee (SIGIR 2003)**  
> *Question Classification using Support Vector Machines*

L'objectif est d'identifier la classe sémantique attendue en réponse à une question formulée en langage naturel, à deux niveaux de granularité (grossier et fin), en comparant les représentations lexicales de surface (mots, n-grammes) aux représentations syntaxiques hiérarchiques par **noyaux d'arbres syntaxiques (Tree Kernels)**.

---

## 1. Jeu de données & Taxonomie

Le corpus utilisé est le jeu standard **TREC-10** (Li & Roth, 2002) :
- **Entraînement (`train_5500.label`)** : 5 452 questions réparties en sous-jeux emboîtés de taille $N \in \{1000, 2000, 3000, 4000, 5500\}$.
- **Test (`TREC_10.label`)** : 500 questions hors-échantillon.

### Deux niveaux de classes
1. **6 classes grossières (coarse)** :
   - `ABBR` : Abréviation (ex: *What does NASA stand for?*)
   - `DESC` : Description ou définition (ex: *What is an atom?*)
   - `ENTY` : Entité (animal, aliment, substance...) (ex: *What is the fastest animal?*)
   - `HUM` : Personne ou groupe (ex: *Who was the first American in space?*)
   - `LOC` : Lieu géographique (ex: *Where is the Eiffel Tower?*)
   - `NUM` : Valeur numérique, date, monnaie... (ex: *How many people live in Tokyo?*)
2. **50 classes fines (fine)** : sous-catégories précises (ex: `LOC:city`, `HUM:ind`, `NUM:count`, `DESC:def`...).

---

## 2. Architecture du Projet

```text
projet_question_classification/
├── data/                       # Données brutes TREC (.label) et CSV
│   ├── train_5500.label
│   ├── TREC_10.label
│   ├── train.csv
│   └── test.csv
├── data_prepared/              # Données prétraitées et matrices
│   ├── train_1000.csv ... train_5500.csv
│   ├── test_500.csv
│   ├── X_bow_*_{train,test}.npz       # Matrices creuses Sac de Mots
│   ├── X_ngram_*_{train,test}.npz     # Matrices creuses N-grammes
│   ├── arbres_train_5500.jsonl        # Arbres syntaxiques en constituants
│   ├── arbres_test_500.jsonl
│   └── gram_0.2_0.9_5452.npy          # Matrice de Gram précalculée (237 Mo)
├── results/                    # Tableaux de résultats générés (CSV)
│   ├── tableau_2.csv ... tableau_5.csv  # Baselines ML (NN, NB, DT, SNoW, SVM)
│   ├── tableau_7.csv & tableau_8.csv    # Réglage de lambda et mu
│   ├── tableau_9.csv & tableau_9_fin.csv# Comparaison finale Mots vs N-grams vs Arbre
│   ├── tableau_10_matrice_confusion.csv # Matrice de confusion
│   └── tableau_11.csv                   # Test de signe (significativité statistique)
├── src/                        # Code source (scripts et notebooks)
│   ├── prep_donnees.py / .ipynb         # Tâche 1 : Préparation & vectorisation
│   ├── comparaison_algos.py / .ipynb    # Tâche 2 : 5 algorithmes de classification
│   ├── tree_kernel.py / .ipynb          # Tâche 3 : Calcul & parsing du Tree Kernel
│   ├── eval_tree_kernel.py / .ipynb     # Tâche 3 : Évaluation & matrices de Gram
│   └── serveur.py                       # Tâche 4 : Serveur web Flask de démonstration
├── web/                        # Interface utilisateur web
│   └── index.html                       # Frontend interactif
├── requirements.txt            # Dépendances Python
└── README.md                   # Documentation du projet
```

---

## 3. Méthodes & Algorithmes Implémentés

### A. Représentations vectorielles de surface (Tâche 1)
- **Mots (BoW)** : Caractéristiques binaires indicatrices d'occurrence de mots, sans suppression des mots vides (*stop-words*) car les pronoms interrogatifs (*Who, Where, How, What*) sont cruciaux en question answering.
- **N-grammes (Ngram)** : Toutes les séquences continues de $1$ à $L_{max}$ mots (~210 000 dimensions pour $N=5500$).

### B. Algorithmes de base (Tâche 2 - Reproduction Tableaux 2 à 5)
1. **Plus Proche Voisin (NN)** : $k$-NN ($k=5$) avec fonction de similarité par produit scalaire sur vecteurs binaires.
2. **Naïve Bayes (NB)** : Multinomial Naive Bayes.
3. **Arbre de Décision (DT)** : Critère d'entropie (équivalent C4.5).
4. **Winnow / SNoW** : Réseau de classifieurs linéaires avec mises à jour multiplicatives ($\alpha=1.1, \beta=0.9$).
5. **SVM Linéaire** : SVM à marge rigide/douce avec stratégie One-against-One.

### C. Noyau d'Arbre Syntaxique (Tree Kernel - Tâches 3 & 4)
Le noyau d'arbre de Collins & Duffy (2001) adapté par Zhang & Lee (2003) calcule le produit scalaire implicite dans l'espace de tous les sous-arbres partagés par programmation dynamique :
$$K(T_1, T_2) = \sum_{n_1 \in N_1} \sum_{n_2 \in N_2} C(n_1, n_2)$$

Deux hyperparamètres de régularisation et d'atténuation sont optimisés :
- $\lambda \in [0, 1]$ : Facteur d'atténuation pénalisant les fragments selon leur taille (nombre de règles de production).
- $\mu \in [0, 1]$ : Facteur d'atténuation pénalisant les fragments selon leur profondeur dans l'arbre syntaxique.
- *Propriété limite* : Lorsque $\lambda = 0$ et $\mu = 1$, le noyau d'arbre équivaut exactement au produit scalaire des fréquences de mots terminaux (démontré et validé par les tests unitaires).

---

## 4. Résultats Expérimentaux & Comparaisons

### A. Tableau 9 : Précision sur les classes grossières (Test 500 questions, en %)

| Méthode | N=1000 | N=2000 | N=3000 | N=4000 | N=5500 (Projet) | N=5500 (Article) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Mots (SVM)** | 78.0 | 80.8 | 83.6 | 85.4 | **86.4** | 85.8 |
| **N-grammes (SVM)** | 72.0 | 78.0 | 80.6 | 83.8 | **84.4** | 87.4 |
| **Noyau d'arbre (SVM)** | **83.2** | **85.8** | **88.6** | **90.0** | **91.0** | **90.0** |

> **Conclusion** : Le SVM à noyau d'arbre syntaxique surpasse systématiquement les représentations de surface (+4.6% par rapport aux mots et +6.6% par rapport aux n-grammes pour $N=5500$), atteignant **91.0%** d'exactitude.

### B. Précision sur les classes fines (Test 500 questions, en %)

| Méthode | N=1000 | N=2000 | N=3000 | N=4000 | N=5500 (Projet) | N=5500 (Article) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Mots (SVM)** | 68.0 | 72.0 | 76.0 | 80.4 | **80.8** | 80.2 |
| **N-grammes (SVM)** | 58.4 | 64.8 | 69.6 | 72.4 | **74.4** | 79.2 |
| **Noyau d'arbre (SVM)** | **70.0** | **75.0** | **75.6** | **81.2** | **82.8** | **80.0** |

### C. Test de Signe & Significativité Statistique (Tableau 11)

| Comparaison (A vs B) | $n$ (divergences) | $k$ (succès A) | $Z$ | $P\text{-value}$ |
| :--- | :---: | :---: | :---: | :---: |
| **Tree Kernel vs Mots** | 35 | 29 | 3.89 | **$0.0001$** |
| **Tree Kernel vs N-grammes** | 47 | 40 | 4.81 | **$< 10^{-4}$** |

Les deux tests confirment la supériorité statistiquement significative ($p < 0.001$) de l'approche structurelle syntaxique.

---

## 5. Réglage des Hyperparamètres (Tuning)

1. **Pénalisation de taille $\lambda$ (Tableau 7, $\mu=1.0$)** :
   - Évalué par validation croisée à 4 plis sur 1000 questions :
   - $\lambda=0.0 \to 69.9\%$ | $\lambda=0.1 \to 74.9\%$ | **$\lambda=0.2 \to 75.5\%$** | $\lambda=0.3 \to 75.1\%$ | $\lambda=0.4 \to 74.2\%$
   - *Optimum retenu* : $\mathbf{\lambda = 0.2}$.

2. **Atténuation de profondeur $\mu$ (Tableau 8, $\lambda=0.2$)** :
   - $\mu=1.0 \to 75.5\%$ | **$\mu=0.9 \to 76.7\%$** | $\mu=0.8 \to 76.2\%$ | $\mu=0.7 \to 73.9\%$ | $\mu=0.6 \to 70.1\%$
   - *Optimum retenu* : $\mathbf{\mu = 0.9}$.

3. **Paramètre de régularisation $C$ du SVM** :
   - Les valeurs $C \in [1.0, 10.0]$ fournissent une performance robuste et stable (91.0% en grossier, 82.8% en fin).

4. **Noyau Hybride & Normalisation** :
   - *Normalisation de Collins & Duffy* : $K_{norm}(T_1, T_2) = \frac{K(T_1, T_2)}{\sqrt{K(T_1, T_1) K(T_2, T_2)}}$ atteint $90.4\%$ avec $C=5.0$.
   - *Combinaison linéaire Arbre + BoW* : $K = K_{tree} + \gamma K_{bow}$ permet d'injecter des indices lexicaux directs tout en conservant la structure hiérarchique.

---

## 6. Utilisation & Exécution

### Pré-requis
Environnement Python 3.10+ avec les bibliothèques suivantes :
```bash
pip install -r requirements.txt
pip install flask stanza
```

### Commandes directes

#### 1. Lancer les tests unitaires du noyau d'arbre
Vérifie la programmation dynamique contre l'énumération par force brute et la propriété de limite $\lambda=0, \mu=1$ :
```bash
python src/tree_kernel.py test
```

#### 2. Préparer les données et caractéristiques (Tâche 1)
```bash
python src/prep_donnees.py
```

#### 3. Comparer les algorithmes de base (Tâche 2)
```bash
python src/comparaison_algos.py --rapide   # Test rapide sur N=1000 (~8s)
python src/comparaison_algos.py            # Reproduction complète des Tableaux 2 à 5
```

#### 4. Évaluer le SVM à Noyau d'Arbre (Tâche 3)
```bash
python src/eval_tree_kernel.py --rapide --params 0.2 0.9   # Test rapide (~10s)
python src/eval_tree_kernel.py                             # Évaluation complète
```

#### 5. Lancer l'interface web de démonstration (Tâche 4)
```bash
python src/serveur.py
```
Ouvrez ensuite votre navigateur sur **http://localhost:5000**.  
*(Pour exécuter sans dépendance Stanza : `python src/serveur.py --sans-arbre`)*
