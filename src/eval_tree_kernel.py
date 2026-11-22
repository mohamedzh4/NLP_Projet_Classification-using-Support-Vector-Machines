"""Tâche 3 (partie 2) : évaluation du SVM à noyau d'arbre (Section 4.4 de l'article).
Produit : Tableaux 7, 8 (réglage de lambda et mu), 9, 10 (matrice de confusion), 11 (test de signe).

Utilisation (depuis le dossier racine du projet) :
    python src/tree_kernel.py parser        # 1 fois : crée data_prepared/arbres_*.jsonl
    python src/eval_tree_kernel.py          # tout
    python src/eval_tree_kernel.py --rapide # petit test (n_max = 1000)
    python src/eval_tree_kernel.py --params 0.1 0.9   # imposer lambda et mu (valeurs de l'article)

Rappel : lambda = 0 et mu = 1 donne le noyau linéaire sur les mots.
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import KFold
from sklearn.svm import SVC

RACINE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(RACINE, "src"))

from prep_donnees import charger_pour_modele, TAILLES
from tree_kernel import charger_arbres, preparer_tous, matrice_gram

DOSSIER = os.path.join(RACINE, "data_prepared")
RESULTATS = os.path.join(RACINE, "results")
CLASSES = ["ABBR", "DESC", "ENTY", "HUM", "LOC", "NUM"]
LAMBDAS = [0.0, 0.1, 0.2, 0.3, 0.4]        # Tableau 7 (mu = 1)
MUS = [1.0, 0.9, 0.8, 0.7, 0.6]            # Tableau 8 (lambda fixe)
SEED_CV = 0


def svm_precalcule(K_tr, y_tr, K_te, C=1.0):
    """SVM avec noyau précalculé (paramètres par défaut, comme dans l'article)."""
    return SVC(kernel="precomputed", C=C).fit(K_tr, y_tr).predict(K_te)


def cv_4_plis(K, y):
    """Validation croisée à 4 plis sur une matrice de Gram déjà calculée."""
    scores = []
    for tr, te in KFold(4, shuffle=True, random_state=SEED_CV).split(K):
        pred = svm_precalcule(K[np.ix_(tr, tr)], y[tr], K[np.ix_(te, tr)])
        scores.append(accuracy_score(y[te], pred))
    return float(np.mean(scores)) * 100


def regler_parametres(arbres_1000, y_1000):
    """Tableaux 7 et 8 : CV à 4 plis sur 1000 exemples (catégories grossières)."""
    print("Réglage de lambda (mu = 1) ...", flush=True)
    t7 = {}
    for lam in LAMBDAS:
        K = matrice_gram(arbres_1000, arbres_1000, lam, 1.0, sym=True)
        t7[lam] = cv_4_plis(K, y_1000)
        print(f"   lambda={lam:.1f} : {t7[lam]:.1f} %", flush=True)
    lam_best = max(t7, key=t7.get)
    print(f"Réglage de mu (lambda = {lam_best}) ...", flush=True)
    t8 = {}
    for mu in MUS:
        K = matrice_gram(arbres_1000, arbres_1000, lam_best, mu, sym=True)
        t8[mu] = cv_4_plis(K, y_1000)
        print(f"   mu={mu:.1f} : {t8[mu]:.1f} %", flush=True)
    mu_best = max(t8, key=t8.get)
    pd.Series(t7, name="accuracy").rename_axis("lambda").round(1).to_csv(f"{RESULTATS}/tableau_7.csv")
    pd.Series(t8, name="accuracy").rename_axis("mu").round(1).to_csv(f"{RESULTATS}/tableau_8.csv")
    return lam_best, mu_best


def test_de_signe(ok_a, ok_b):
    """Test de signe (article, Tableau 11) : n = questions où les deux méthodes
    diffèrent (l'une juste, l'autre fausse), k = parmi elles, celles où A est juste.
    Z = (k - n/2) / (racine(n)/2), P-value unilatérale."""
    ok_a, ok_b = np.asarray(ok_a), np.asarray(ok_b)
    n = int(np.sum(ok_a != ok_b))
    k = int(np.sum(ok_a & ~ok_b))
    z = (k - n / 2) / (np.sqrt(n) / 2) if n > 0 else float("nan")
    p = float(norm.sf(z)) if n > 0 else float("nan")
    return n, k, z, p


def main(rapide=False, params=None):
    os.makedirs(RESULTATS, exist_ok=True)
    tailles = [1000] if rapide else TAILLES
    n_max = max(tailles)

    chemin_tr = os.path.join(DOSSIER, "arbres_train_5500.jsonl")
    chemin_te = os.path.join(DOSSIER, "arbres_test_500.jsonl")
    if not (os.path.exists(chemin_tr) and os.path.exists(chemin_te)):
        sys.exit("Arbres introuvables : lancez d'abord python src/tree_kernel.py parser")

    df_tr = pd.read_csv(os.path.join(DOSSIER, "train_5500.csv"))
    df_te = pd.read_csv(os.path.join(DOSSIER, "test_500.csv"))
    arbres_tr, arbres_te = charger_arbres(chemin_tr), charger_arbres(chemin_te)
    assert len(arbres_tr) == len(df_tr) and len(arbres_te) == len(df_te), "arbres et questions ne correspondent pas"

    tr, vocab = preparer_tous(arbres_tr[:n_max])
    te, _ = preparer_tous(arbres_te, vocab)

    # ---- Tableaux 7 et 8
    if params is None:
        lam, mu = regler_parametres(tr[:1000], df_tr["coarse"].values[:1000])
    else:
        lam, mu = params
    print(f"\nParamètres retenus : lambda = {lam}, mu = {mu}\n", flush=True)

    # ---- Matrices de Gram (calculées une seule fois, ou chargées depuis le cache)
    cache_tr = os.path.join(DOSSIER, f"gram_{lam}_{mu}_{n_max}.npy")
    if os.path.exists(cache_tr):
        print(f"Chargement de la matrice de Gram d'entraînement en cache ({cache_tr}) ...", flush=True)
        K_tr = np.load(cache_tr)
    else:
        print(f"Calcul de la matrice de Gram d'entraînement ({n_max} x {n_max}) ...", flush=True)
        K_tr = matrice_gram(tr, tr, lam, mu, sym=True)
        if not rapide:
            np.save(cache_tr, K_tr)

    print(f"Calcul de la matrice de Gram test ({len(te)} x {n_max}) ...", flush=True)
    K_te = matrice_gram(te, tr, lam, mu)

    # ---- Tableau 9 (grossier) et équivalent fin
    sorties = {}
    for niveau, num in [("coarse", 9), ("fine", "9_fin")]:
        y_tr_all, y_te = df_tr[niveau].values, df_te[niveau].values
        lignes = {"word": {}, "ngram": {}, "tree": {}}
        preds_5500 = {}
        for n in tailles:
            for feat in ["bow", "ngram"]:
                X_tr, y_tr, X_te, y_te_ = charger_pour_modele(n, feat, niveau, dossier=DOSSIER)
                p = SVC(kernel="linear").fit(X_tr, y_tr).predict(X_te)
                lignes["word" if feat == "bow" else "ngram"][n] = round(accuracy_score(y_te_, p) * 100, 1)
                preds_5500[(feat, n)] = p
            p = svm_precalcule(K_tr[:n, :n], y_tr_all[:n], K_te[:, :n])
            lignes["tree"][n] = round(accuracy_score(y_te, p) * 100, 1)
            preds_5500[("tree", n)] = p
            print(f"[{niveau:6s}] n={n:4d} | mots {lignes['word'][n]:5.1f} | ngrams {lignes['ngram'][n]:5.1f} | tree {lignes['tree'][n]:5.1f}", flush=True)
        t9 = pd.DataFrame(lignes).T
        t9.to_csv(f"{RESULTATS}/tableau_{num}.csv")
        sorties[niveau] = (preds_5500, y_te)
        print(f"\n=== Tableau {num} ({niveau}, %) ===\n{t9.to_string()}\n", flush=True)

    # ---- Tableau 10 : matrice de confusion (grossier, n_max)
    preds, y_te = sorties["coarse"]
    cm = confusion_matrix(y_te, preds[("tree", n_max)], labels=CLASSES)
    cm = pd.DataFrame(cm, index=CLASSES, columns=CLASSES)
    cm.to_csv(f"{RESULTATS}/tableau_10_matrice_confusion.csv")
    print(f"=== Tableau 10 : matrice de confusion (lignes = vraie classe) ===\n{cm.to_string()}\n")

    # ---- Tableau 11 : test de signe (grossier, n_max)
    lignes = []
    for nom, feat in [("word", "bow"), ("ngram", "ngram")]:
        ok_a = preds[("tree", n_max)] == y_te
        ok_b = preds[(feat, n_max)] == y_te
        n, k, z, p = test_de_signe(ok_a, ok_b)
        lignes.append({"A": "tree", "B": nom, "train": n_max, "test": len(y_te),
                       "n": n, "k": k, "Z": round(z, 2), "P-value": round(p, 4)})
    t11 = pd.DataFrame(lignes)
    t11.to_csv(f"{RESULTATS}/tableau_11.csv", index=False)
    print(f"=== Tableau 11 : test de signe ===\n{t11.to_string(index=False)}\n")
    print(f"Fichiers écrits dans {RESULTATS}/")


if __name__ == "__main__":
    rapide = "--rapide" in sys.argv
    params = None
    if "--params" in sys.argv:
        idx = sys.argv.index("--params")
        params = (float(sys.argv[idx + 1]), float(sys.argv[idx + 2]))
    main(rapide=rapide, params=params)
