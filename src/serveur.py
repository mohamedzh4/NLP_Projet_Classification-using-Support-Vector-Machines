"""
Interface de test : classer de NOUVELLES questions avec les modeles du projet.

Utilisation (depuis le dossier racine du projet) :
    pip install flask
    python src/serveur.py                 # puis ouvrir http://localhost:5000
    python src/serveur.py --sans-arbre    # sans noyau d'arbre (Stanza non necessaire)

Au demarrage, les modeles sont entraines sur toutes les questions d'entrainement
(quelques minutes la premiere fois ; la matrice de Gram du noyau d'arbre est
ensuite gardee dans data_prepared/gram_*.npy).
Pre-requis : prep_donnees.py lance ; pour l'arbre : tree_kernel.py parser lance.
"""
import os
import sys
import time
import numpy as np
import pandas as pd

RACINE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(RACINE, "src"))
# Ajout au sys.path des packages du venv local en fallback (ex: flask)
venv_site = os.path.join(RACINE, "venv", "Lib", "site-packages")
if os.path.exists(venv_site) and venv_site not in sys.path:
    sys.path.append(venv_site)

from flask import Flask, jsonify, request, send_from_directory
from sklearn.svm import SVC

from prep_donnees import faire_vectoriseur
from tree_kernel import (charger_arbres, depuis_stanza, matrice_gram, noyau_paire,
                         ponderer, preparer, preparer_tous)

DOSSIER = os.path.join(RACINE, "data_prepared")
WEB = os.path.join(RACINE, "web")   # dossier web/ à la racine
LAM, MU = 0.2, 0.9          # parametres retenus par la validation croisee
NOMS = {"ABBR": "Abréviation", "DESC": "Description / définition", "ENTY": "Entité",
        "HUM": "Personne / groupe", "LOC": "Lieu", "NUM": "Valeur numérique"}


def entrainer_surface(df):
    """SVM lineaires (mots, n-grammes) pour les niveaux grossier et fin."""
    modeles = {}
    for feat, nom in [("bow", "Mots (SVM linéaire)"), ("ngram", "N-grammes (SVM linéaire)")]:
        t0 = time.time()
        vec = faire_vectoriseur(feat, df["texte"].tolist())
        X = vec.fit_transform(df["texte"])
        coarse = SVC(kernel="linear").fit(X, df["coarse"])
        fine = SVC(kernel="linear").fit(X, df["fine"])
        modeles[nom] = {"predire": _predire_lineaire(vec, coarse, fine)}
        print(f"  {nom} entraîné ({time.time() - t0:.0f}s)", flush=True)
    return modeles


def _predire_lineaire(vec, coarse, fine):
    def predire(q):
        x = vec.transform([q])
        return coarse.predict(x)[0], fine.predict(x)[0]
    return predire


def charger_parseur():
    import stanza
    nlp = stanza.Pipeline("en", processors="tokenize,pos,constituency",
                          tokenize_no_ssplit=True, verbose=False)
    return lambda q: depuis_stanza(nlp(q).sentences[0].constituency)


def entrainer_arbre(df, parseur):
    """SVM a noyau d'arbre precalcule (grossier et fin)."""
    chemin = os.path.join(DOSSIER, "arbres_train_5500.jsonl")
    if not os.path.exists(chemin):
        raise FileNotFoundError("arbres introuvables : lancez python src/tree_kernel.py parser")
    arbres = charger_arbres(chemin)
    assert len(arbres) == len(df), "arbres et questions ne correspondent pas"
    trees, vocab = preparer_tous(arbres)
    cache = os.path.join(DOSSIER, f"gram_{LAM}_{MU}_{len(trees)}.npy")
    if os.path.exists(cache):
        K = np.load(cache)
        ponderer(trees, MU)
    else:
        print("  Calcul de la matrice de Gram (une seule fois) ...", flush=True)
        K = matrice_gram(trees, trees, LAM, MU, sym=True)
        np.save(cache, K)
    coarse = SVC(kernel="precomputed").fit(K, df["coarse"])
    fine = SVC(kernel="precomputed").fit(K, df["fine"])

    def predire(q):
        t = preparer(parseur(q), vocab)
        ponderer([t], MU)
        k = np.array([[noyau_paire(t, a, LAM) for a in trees]])
        return coarse.predict(k)[0], fine.predict(k)[0]
    return {"Noyau d'arbre (SVM)": {"predire": predire}}


def creer_app(modeles, indisponibles=None):
    app = Flask(__name__, static_folder=None)
    indisponibles = indisponibles or {}

    @app.route("/")
    def accueil():
        return send_from_directory(os.path.abspath(WEB), "index.html")

    @app.route("/api/predire", methods=["POST"])
    def predire():
        question = ((request.get_json(silent=True) or {}).get("question") or "").strip()
        if not question:
            return jsonify(erreur="Question vide."), 400
        resultats = []
        for nom, m in modeles.items():
            try:
                c, f = m["predire"](question)
                resultats.append({"modele": nom, "grossier": str(c),
                                  "grossier_nom": NOMS.get(str(c), ""), "fin": str(f)})
            except Exception as e:                      # une erreur n'arrete pas les autres modeles
                resultats.append({"modele": nom, "erreur": str(e)})
        for nom, raison in indisponibles.items():
            resultats.append({"modele": nom, "erreur": raison})
        return jsonify(question=question, resultats=resultats)

    return app


if __name__ == "__main__":
    df = pd.read_csv(os.path.join(DOSSIER, "train_5500.csv"))
    print(f"Entraînement sur {len(df)} questions ...", flush=True)
    modeles, indispo = entrainer_surface(df), {}
    if "--sans-arbre" not in sys.argv:
        try:
            modeles.update(entrainer_arbre(df, charger_parseur()))
            print("  Noyau d'arbre prêt.", flush=True)
        except Exception as e:
            indispo["Noyau d'arbre (SVM)"] = f"indisponible : {e}"
            print(f"  Noyau d'arbre ignoré : {e}", flush=True)
    print("\nOuvrez http://localhost:5000", flush=True)
    creer_app(modeles, indispo).run(host="127.0.0.1", port=5000)