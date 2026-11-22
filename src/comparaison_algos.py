"""Tâche 2 : comparaison des algorithmes (Section 3 de l'article).
Reproduit les Tableaux 2 à 5 : NN, NB, DT, SNoW (substitut), SVM.

Utilisation (depuis le dossier racine du projet) :
    python src/comparaison_algos.py             # tout (peut prendre du temps)
    python src/comparaison_algos.py --rapide    # seulement n=1000, pour tester

Pré-requis : avoir lancé prep_donnees.py (dossier data_prepared/ présent).
Tous les algorithmes gardent leurs paramètres par défaut, comme dans l'article.
"""
import os
import sys
import time
import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Assurer l'import des modules frères dans src/
RACINE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(RACINE, "src"))

from prep_donnees import charger_pour_modele, TAILLES

RESULTS_DIR = os.path.join(RACINE, "results")
DOSSIER = os.path.join(RACINE, "data_prepared")


# ------------------------------------------------------------------ NN (kNN simplifié)
class NNProduitScalaire:
    """Version simplifiée de kNN (article, 3.4) : similarité = nombre de
    caractéristiques communes (= produit scalaire sur vecteurs binaires).
    Classe prédite = classe dominante parmi les k voisins les plus similaires.
    L'article ne donne pas k : on prend k=5 (valeur par défaut de scikit-learn)."""

    def __init__(self, k=5):
        self.k = k

    def fit(self, X, y):
        self.X_ = X.tocsr()
        self.classes_, self.y_idx_ = np.unique(y, return_inverse=True)
        return self

    def predict(self, X):
        S = (X.tocsr() @ self.X_.T).toarray()          # similarités test x train
        k = min(self.k, S.shape[1])
        voisins = np.argpartition(-S, k - 1, axis=1)[:, :k]
        preds = []
        for i, idx in enumerate(voisins):
            votes = np.zeros(len(self.classes_))
            for j in idx:
                votes[self.y_idx_[j]] += 1 + 1e-6 * S[i, j]   # égalité -> plus similaire
            preds.append(self.classes_[votes.argmax()])
        return np.array(preds)


# ------------------------------------------------------------------ Winnow (substitut de SNoW)
class WinnowMulticlasse:
    """
    SUBSTITUT de SNoW (le logiciel d'origine n'est pas utilisé ici).
    Réseau de fonctions linéaires (une par classe) mis à jour par Winnow.
    NOTE : Pour les grandes dimensions (n-grams), on utilise des paramètres
    plus stables (alpha=1.1, beta=0.9) pour éviter l'effondrement des poids.
    """

    def __init__(self, alpha=1.1, beta=0.9, theta=1.0, w0=0.1, epochs=5, seed=0):
        self.alpha, self.beta, self.theta, self.w0 = alpha, beta, theta, w0
        self.epochs, self.seed = epochs, seed

    def fit(self, X, y):
        X = X.tocsr()
        self.classes_, y_idx = np.unique(y, return_inverse=True)
        self.W_ = np.full((len(self.classes_), X.shape[1]), self.w0)
        rng = np.random.RandomState(self.seed)

        for _ in range(self.epochs):
            for i in rng.permutation(X.shape[0]):
                idx = X.indices[X.indptr[i]:X.indptr[i + 1]]
                scores = self.W_[:, idx].sum(axis=1)
                actifs = scores > self.theta

                for c in range(len(self.classes_)):
                    if c == y_idx[i] and not actifs[c]:
                        self.W_[c, idx] *= self.alpha
                    elif c != y_idx[i] and actifs[c]:
                        self.W_[c, idx] *= self.beta
        return self

    def predict(self, X):
        X = X.tocsr()
        scores = X @ self.W_.T
        return self.classes_[np.asarray(scores).argmax(axis=1)]


# ------------------------------------------------------------------ Les 5 algorithmes
ALGOS = {
    "NN":   lambda: NNProduitScalaire(k=5),
    "NB":   lambda: MultinomialNB(),
    "DT":   lambda: DecisionTreeClassifier(criterion="entropy", random_state=0),
    "SNoW": lambda: WinnowMulticlasse(),
    "SVM":  lambda: SVC(kernel="linear"),
}

# Tableaux de l'article (précision en %), pour comparer avec les résultats
ARTICLE = {
    ("bow", "coarse"): {  # Tableau 2
        "NN": [70.0, 73.6, 74.8, 74.8, 75.6], "NB": [53.8, 60.4, 74.2, 76.0, 77.4],
        "DT": [78.8, 79.8, 82.0, 83.4, 84.2], "SNoW": [71.8, 73.4, 74.2, 78.2, 66.8],
        "SVM": [76.8, 83.4, 87.2, 87.4, 85.8]},
    ("ngram", "coarse"): {  # Tableau 3
        "NN": [72.0, 81.0, 79.8, 80.8, 79.8], "NB": [73.0, 79.2, 80.0, 81.8, 83.2],
        "DT": [73.8, 82.6, 83.0, 84.6, 84.2], "SNoW": [59.8, 85.2, 80.6, 87.0, 86.6],
        "SVM": [77.6, 82.6, 84.0, 84.8, 87.4]},
    ("bow", "fine"): {  # Tableau 4
        "NN": [57.4, 62.8, 65.2, 67.2, 68.4], "NB": [48.8, 52.8, 56.6, 56.2, 58.4],
        "DT": [67.0, 70.0, 73.6, 75.4, 77.0], "SNoW": [42.2, 66.2, 69.0, 66.6, 74.0],
        "SVM": [68.0, 75.0, 77.2, 77.4, 80.2]},
    ("ngram", "fine"): {  # Tableau 5
        "NN": [59.4, 64.6, 67.2, 67.4, 68.6], "NB": [54.4, 58.4, 63.0, 65.0, 67.8],
        "DT": [62.8, 72.2, 72.6, 73.0, 77.0], "SNoW": [44.0, 67.0, 75.0, 55.8, 75.8],
        "SVM": [65.0, 74.0, 74.8, 77.4, 79.2]},
}
NUM_TABLEAU = {("bow", "coarse"): 2, ("ngram", "coarse"): 3,
               ("bow", "fine"): 4, ("ngram", "fine"): 5}


def main(rapide=False):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    tailles = [TAILLES[0]] if rapide else TAILLES
    lignes = []
    for (feat, niveau), num in NUM_TABLEAU.items():
        for n in tailles:
            X_tr, y_tr, X_te, y_te = charger_pour_modele(n, feat, niveau, dossier=DOSSIER)
            for nom, fabrique in ALGOS.items():
                t0 = time.time()
                modele = fabrique().fit(X_tr, y_tr)
                acc = accuracy_score(y_te, modele.predict(X_te)) * 100
                idx = TAILLES.index(n)
                lignes.append({"tableau": num, "features": feat, "niveau": niveau,
                               "algo": nom, "n": n, "accuracy": round(acc, 1),
                               "article": ARTICLE[(feat, niveau)][nom][idx]})
                print(f"Tableau {num} | {feat:5s} {niveau:6s} | n={n:4d} | {nom:5s} "
                      f"| {acc:5.1f}% (article {ARTICLE[(feat, niveau)][nom][idx]}%) "
                      f"[{time.time() - t0:.0f}s]", flush=True)

    df = pd.DataFrame(lignes)
    df.to_csv(os.path.join(RESULTS_DIR, "tableaux_2_a_5.csv"), index=False)

    for num, g in df.groupby("tableau"):
        t = g.pivot(index="algo", columns="n", values="accuracy").reindex(list(ALGOS))
        t.to_csv(os.path.join(RESULTS_DIR, f"tableau_{num}.csv"))
        print(f"\n=== Tableau {num} (résultats, %) ===")
        print(t.to_string())
    print(f"\nFichiers écrits dans {RESULTS_DIR}/")


if __name__ == "__main__":
    rapide = "--rapide" in sys.argv
    main(rapide=rapide)
