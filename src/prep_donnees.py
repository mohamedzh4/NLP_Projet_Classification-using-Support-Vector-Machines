"""Tache 1 : preparation des donnees et des caracteristiques.
Utilisation (depuis la racine du projet) : python src/prep_donnees.py
"""
import os
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer

SEED = 42                                  # graine fixe : meme tirage pour toute l'equipe
TAILLES = [1000, 2000, 3000, 4000, 5500]   # tailles des jeux d'entrainement (article)
DATA_DIR = "data"
OUT_DIR = "data_prepared"

# ---------------------------------------------------------------- 1. Chargement
def charger_fichier(chemin):
    """Lit un fichier .label : chaque ligne = 'COARSE:fine texte de la question'."""
    lignes = []
    # latin-1 : le fichier contient un octet qui n'est pas du UTF-8
    with open(chemin, encoding="latin-1") as f:
        for ligne in f:
            ligne = ligne.strip()
            if not ligne:
                continue
            etiquette, _, texte = ligne.partition(" ")
            coarse, _, fine = etiquette.partition(":")
            lignes.append({
                "texte": texte,
                "coarse": coarse,                 # 6 classes (ABBR, DESC, ENTY, HUM, LOC, NUM)
                "fine": f"{coarse}:{fine}",       # ~50 classes ; on garde COARSE:fine car
                                                  # 'other' existe dans ENTY, LOC et NUM
            })
    return pd.DataFrame(lignes)

# ---------------------------------------------------------------- 2. Verification
def verifier(train, test):
    print(f"Train : {len(train)} questions | Test : {len(test)} questions")
    print(f"Classes grossieres : {train['coarse'].nunique()} (train), {test['coarse'].nunique()} (test)")
    print(f"Classes fines      : {train['fine'].nunique()} (train), {test['fine'].nunique()} (test)")
    print("\nRepartition des classes grossieres (train) :")
    print(train["coarse"].value_counts().to_string())
    print(f"\nLongueur max d'une question : {train['texte'].str.split().str.len().max()} mots")

# ---------------------------------------------------------------- 3. Tirage des sous-jeux
def faire_sous_jeux(train):
    """Melange une fois (graine fixe) puis prend les N premieres questions.
    Les jeux sont donc emboites : 1000 inclus dans 2000, etc."""
    melange = train.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    return {n: melange.iloc[:n].copy() for n in TAILLES}

# ---------------------------------------------------------------- 4. Caracteristiques
def faire_vectoriseur(type_feat, textes_train):
    """Vecteurs BINAIRES (0/1), SANS supprimer les stop-words.
    - bow   : sac de mots
    - ngram : toutes les sequences continues de mots (de 1 a la longueur max)
    """
    if type_feat == "bow":
        ngram_range = (1, 1)
    elif type_feat == "ngram":
        max_len = max(len(t.split()) for t in textes_train)
        ngram_range = (1, max_len)
    else:
        raise ValueError("type_feat doit etre 'bow' ou 'ngram'")
    return CountVectorizer(
        binary=True,                       # 0/1 comme dans l'article
        lowercase=True,
        stop_words=None,                   # on GARDE what, is, who...
        token_pattern=r"(?u)\b\w+\b",      # garde aussi les mots d'une lettre (a, I...)
        ngram_range=ngram_range,
    )

def construire_features(train_n, test, type_feat):
    """Le vocabulaire est appris sur le train uniquement, puis applique au test."""
    vec = faire_vectoriseur(type_feat, train_n["texte"].tolist())
    X_train = vec.fit_transform(train_n["texte"]).astype(np.float32)
    X_test = vec.transform(test["texte"]).astype(np.float32)
    return X_train, X_test, vec

# ---------------------------------------------------------------- 5. Sauvegarde
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    train = charger_fichier(os.path.join(DATA_DIR, "train_5500.label"))
    test = charger_fichier(os.path.join(DATA_DIR, "TREC_10.label"))
    verifier(train, test)

    sous_jeux = faire_sous_jeux(train)
    test.to_csv(os.path.join(OUT_DIR, "test_500.csv"), index=False)   # texte + 2 niveaux de labels

    for n, df_n in sous_jeux.items():
        # Texte + labels : utile pour la personne 3 (analyse syntaxique) et pour le rapport
        df_n.to_csv(os.path.join(OUT_DIR, f"train_{n}.csv"), index=False)
        for type_feat in ["bow", "ngram"]:
            X_train, X_test, vec = construire_features(df_n, test, type_feat)
            sparse.save_npz(os.path.join(OUT_DIR, f"X_{type_feat}_{n}_train.npz"), X_train)
            sparse.save_npz(os.path.join(OUT_DIR, f"X_{type_feat}_{n}_test.npz"), X_test)
            print(f"[{type_feat:5s}] n={n:4d} -> {X_train.shape[1]} caracteristiques")
    print(f"\nTermine. Fichiers dans : {OUT_DIR}/")

def charger_pour_modele(n, type_feat, niveau="coarse", dossier=OUT_DIR):
    """Fonction pour la personne 2 : renvoie X_train, y_train, X_test, y_test.
    niveau = 'coarse' (6 classes) ou 'fine' (~50 classes)."""
    X_train = sparse.load_npz(os.path.join(dossier, f"X_{type_feat}_{n}_train.npz"))
    X_test = sparse.load_npz(os.path.join(dossier, f"X_{type_feat}_{n}_test.npz"))
    y_train = pd.read_csv(os.path.join(dossier, f"train_{n}.csv"))[niveau].values
    y_test = pd.read_csv(os.path.join(dossier, "test_500.csv"))[niveau].values
    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    main()
