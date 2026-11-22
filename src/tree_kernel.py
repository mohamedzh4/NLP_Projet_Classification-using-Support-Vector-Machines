"""Tache 3 (partie 1) : arbres syntaxiques et tree kernel.
Commandes (depuis la racine du projet) :
    python src/tree_kernel.py test
    python src/tree_kernel.py parser
"""
import json
import math
import os
import sys
from joblib import Parallel, delayed
import numpy as np

# ===================================================================== Arbres
# Format d'un arbre dans les fichiers : liste imbriquee JSON
#   mot (str)                      -> noeud terminal
#   [etiquette, [enfants...]]      -> noeud interne ou pre-terminal


def depuis_stanza(t):
    """Convertit un arbre de constituants Stanza en liste imbriquee."""
    if t.is_leaf():
        return t.label
    return [t.label, [depuis_stanza(c) for c in t.children]]

def depuis_nltk(t):
    """Convertit un nltk.Tree en liste imbriquee."""
    if isinstance(t, str):
        return t
    return [t.label(), [depuis_nltk(c) for c in t]]

def parser_questions(textes, lot=64):
    """Analyse syntaxique en constituants avec Stanza (une question = une phrase)."""
    import stanza
    stanza.download("en", processors="tokenize,pos,constituency")
    nlp = stanza.Pipeline("en", processors="tokenize,pos,constituency",
                          tokenize_no_ssplit=True, verbose=False)
    arbres = []
    for debut in range(0, len(textes), lot):
        docs = nlp([stanza.Document([], text=t) for t in textes[debut:debut + lot]])
        for d in docs:
            arbres.append(depuis_stanza(d.sentences[0].constituency))
        print(f"  {len(arbres)}/{len(textes)} questions analysees", flush=True)
    return arbres

def sauver_arbres(arbres, chemin):
    with open(chemin, "w", encoding="utf-8") as f:
        for a in arbres:
            f.write(json.dumps(a, ensure_ascii=False) + "\n")

def charger_arbres(chemin):
    with open(chemin, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

# ===================================================================== Structure de calcul
class Arbre:
    """Arbre prepare pour le calcul : tableaux de noeuds (parcours en profondeur)."""
    __slots__ = ("prod", "depth", "kids", "kind", "by_prod", "w", "n")

def preparer(arbre, vocab):
    """Prepare un arbre. 'vocab' associe chaque regle de production a un entier
    (partage entre tous les arbres). Les mots sont mis en minuscules."""
    prod, depth, kids, kind = [], [], [], []

    def etiquette(c):
        return ("w", c.lower()) if isinstance(c, str) else ("l", c[0])

    def ajouter(noeud, d):
        i = len(prod)
        if isinstance(noeud, str):                       # terminal (mot) : taille 0
            cle, k, enfants = ("T", noeud.lower()), 0, []
        else:
            label, enfants = noeud
            if len(enfants) == 1 and isinstance(enfants[0], str):
                cle, k = ("P", label, enfants[0].lower()), 1   # pre-terminal : POS -> mot
            else:
                cle, k = ("N", label, tuple(etiquette(c) for c in enfants)), 2
        prod.append(vocab.setdefault(cle, len(vocab)))
        depth.append(d)
        kids.append(())
        kind.append(k)
        kids[i] = tuple(ajouter(c, d + 1) for c in enfants)
        return i

    ajouter(arbre, 0)
    a = Arbre()
    a.prod, a.depth, a.kids, a.kind = prod, depth, kids, kind
    a.n = len(prod)
    a.by_prod = {}
    for i, p in enumerate(prod):
        a.by_prod.setdefault(p, []).append(i)
    a.w = None
    return a

def preparer_tous(arbres, vocab=None):
    vocab = {} if vocab is None else vocab
    return [preparer(a, vocab) for a in arbres], vocab

def ponderer(arbres, mu):
    """Calcule pour chaque noeud le poids racine(mu)^profondeur."""
    sm = math.sqrt(mu)
    for a in arbres:
        a.w = [sm ** d for d in a.depth]

# ===================================================================== Noyau
def _C(a, b, i, j, lam, memo):
    k = a.kind[i]
    if k == 0:
        return 1.0
    if k == 1:
        return lam
    cle = i * b.n + j
    v = memo.get(cle)
    if v is not None:
        return v
    r = lam
    pa, pb = a.prod, b.prod
    for ci, cj in zip(a.kids[i], b.kids[j]):
        if pa[ci] == pb[cj]:
            r *= 1.0 + _C(a, b, ci, cj, lam, memo)
    memo[cle] = r
    return r

def noyau_paire(a, b, lam):
    """k(T1,T2). Les poids a.w et b.w doivent avoir ete calcules avec ponderer()."""
    pb = b.by_prod
    memo = {}
    total = 0.0
    for p, li in a.by_prod.items():
        lj = pb.get(p)
        if lj is None:
            continue
        for i in li:
            wi = a.w[i]
            for j in lj:
                total += wi * b.w[j] * _C(a, b, i, j, lam, memo)
    return total

def _bloc(lignes, A, B, lam, sym):
    out = []
    for i in lignes:
        debut = i if sym else 0
        out.append((i, debut, [noyau_paire(A[i], B[j], lam) for j in range(debut, len(B))]))
    return out

def matrice_gram(A, B, lam, mu, n_jobs=-1, sym=False):
    """Matrice de Gram K[i,j] = k(A[i], B[j]). sym=True si A et B sont la meme liste
    (on ne calcule alors que la moitie)."""
    ponderer(A, mu)
    if B is not A:
        ponderer(B, mu)
    K = np.zeros((len(A), len(B)))
    n_blocs = 1 if n_jobs == 1 else 32
    blocs = [list(range(k, len(A), n_blocs)) for k in range(n_blocs)]
    blocs = [b for b in blocs if b]
    res = Parallel(n_jobs=n_jobs)(delayed(_bloc)(b, A, B, lam, sym) for b in blocs)
    for lot in res:
        for i, debut, valeurs in lot:
            K[i, debut:] = valeurs
            if sym:
                K[debut:, i] = valeurs
    return K

# ===================================================================== Tests
def _fragments_force_brute(noeud):
    """Enumere tous les fragments enracines en 'noeud' : liste de (description, taille)."""
    if isinstance(noeud, str):
        return [(noeud.lower(), 0)]
    label, enfants = noeud
    if len(enfants) == 1 and isinstance(enfants[0], str):
        return [(f"({label} {enfants[0].lower()})", 1)]
    options = []
    for c in enfants:
        opts = [(f"({c[0]})", 0)]                       # enfant laisse non developpe
        opts += _fragments_force_brute(c)
        options.append(opts)
    resultats = [("", 1)]
    for opts in options:
        resultats = [(d + " " + d2, s + s2) for d, s in resultats for d2, s2 in opts]
    return [(f"({label}{d})", s) for d, s in resultats]

def _tous_noeuds(noeud, d=0):
    yield noeud, d
    if not isinstance(noeud, str):
        for c in noeud[1]:
            yield from _tous_noeuds(c, d + 1)

def _vecteur_force_brute(arbre, lam, mu):
    v = {}
    for noeud, d in _tous_noeuds(arbre):
        for desc, s in _fragments_force_brute(noeud):
            v[desc] = v.get(desc, 0.0) + math.sqrt(lam) ** s * math.sqrt(mu) ** d
    return v

def _noyau_force_brute(a1, a2, lam, mu):
    v1, v2 = _vecteur_force_brute(a1, lam, mu), _vecteur_force_brute(a2, lam, mu)
    return sum(x * v2.get(k, 0.0) for k, x in v1.items())

# Arbre de la Figure 1(a) de l'article : "What is an atom ?"
ARBRE_ATOME = ["S1", [["SBARQ", [
    ["WHNP", [["WP", ["What"]]]],
    ["SQ", [["VP", [["AUX", ["is"]], ["NP", [["DT", ["an"]], ["NN", ["atom"]]]]]]]],
    [".", ["?"]]]]]]
ARBRE_B = ["S1", [["SBARQ", [
    ["WHNP", [["WP", ["Who"]]]],
    ["SQ", [["VP", [["AUX", ["is"]], ["NP", [["DT", ["an"]], ["NN", ["atom"]]]]]]]],
    [".", ["?"]]]]]]
ARBRE_C = ["S1", [["SBARQ", [
    ["WHNP", [["WDT", ["Which"], ], ]],
    ["SQ", [["VP", [["AUX", ["is"]], ["NP", [["DT", ["a"]], ["JJ", ["big"]], ["NN", ["city"]]]]]]]],
    [".", ["?"]]]]]]

def tester():
    ok = True
    arbres = [ARBRE_ATOME, ARBRE_B, ARBRE_C]
    vocab = {}
    prep, vocab = preparer_tous(arbres, vocab)

    print("1) Programmation dynamique contre enumeration de tous les fragments")
    for lam, mu in [(0.1, 0.9), (0.3, 1.0), (0.5, 0.7), (1.0, 1.0)]:
        ponderer(prep, mu)
        pire = 0.0
        for x in range(3):
            for y in range(3):
                dp = noyau_paire(prep[x], prep[y], lam)
                bf = _noyau_force_brute(arbres[x], arbres[y], lam, mu)
                pire = max(pire, abs(dp - bf))
        etat = "OK" if pire < 1e-9 else "ERREUR"
        ok &= pire < 1e-9
        print(f"   lambda={lam}, mu={mu} : ecart max = {pire:.2e}  [{etat}]")

    print("2) lambda=0, mu=1 : le noyau est le produit scalaire des comptes de mots")
    ponderer(prep, 1.0)
    def mots(t):
        return [w.lower() for n, _ in _tous_noeuds(t) if isinstance(n, str) for w in [n]]
    for x in range(3):
        for y in range(3):
            attendu = sum(mots(arbres[x]).count(w) * mots(arbres[y]).count(w)
                          for w in set(mots(arbres[x])))
            obtenu = noyau_paire(prep[x], prep[y], 0.0)
            ok &= abs(attendu - obtenu) < 1e-9
    print("   [OK]" if ok else "   [ERREUR]")

    print("3) Exemple de l'article : fragments du sous-arbre NP 'an atom' (Tableau 6)")
    np_noeud = ARBRE_ATOME[1][0][1][1][1][0][1][1]            # S1>SBARQ>SQ>VP>NP
    profondeurs = {id(n): d for n, d in _tous_noeuds(ARBRE_ATOME)}
    print("   noeud   taille  profondeur  fragment")
    for nom, noeud in [("NP", np_noeud), ("DT", np_noeud[1][0]), ("NN", np_noeud[1][1]),
                       ("mot an", np_noeud[1][0][1][0]), ("mot atom", np_noeud[1][1][1][0])]:
        d = profondeurs[id(noeud)]
        for desc, s in _fragments_force_brute(noeud):
            print(f"   {nom:8s}{s:5d}{d:9d}     {desc}")
    print("   Attendu (Tableau 6) : c1 (3,4) ; c2 (2,4) ; c3 (2,4) ; c4 (2,4) ; c5, c6 (1,5) ; c7, c8 (0,6).")
    print("   Remarque : la formule de l'article donne pour le fragment NP -> DT NN seul")
    print("   une taille 1, alors que le Tableau 6 indique 2 pour c4 : probable erreur du tableau.")
    print("\nTous les tests passent." if ok else "\nATTENTION : un test a echoue.")
    return ok

# ===================================================================== Ligne de commande
def commande_parser():
    import pandas as pd
    dossier = "data_prepared"
    for nom_csv, nom_arbres in [("train_5500.csv", "arbres_train_5500.jsonl"),
                                ("test_500.csv", "arbres_test_500.jsonl")]:
        sortie = os.path.join(dossier, nom_arbres)
        if os.path.exists(sortie):
            print(f"{sortie} existe deja : on le garde (supprimez-le pour refaire).")
            continue
        textes = pd.read_csv(os.path.join(dossier, nom_csv))["texte"].tolist()
        print(f"Analyse de {len(textes)} questions ({nom_csv}) ...")
        sauver_arbres(parser_questions(textes), sortie)
        print(f"-> {sortie}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "test"
    if cmd == "test":
        tester()
    elif cmd == "parser":
        commande_parser()
    else:
        print("Commande inconnue : utilisez 'test' ou 'parser'")
