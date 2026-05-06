# Projet MPI — Calculs Parallèles (M1 HPC, USTHB)

**Auteur :** Merbouche Abdelatif  
**Encadrante :** Mme Baba Ali  
**Année universitaire :** 2025/2026  

---

## Table des matières

1. [Vue d'ensemble](#vue-densemble)
2. [Architecture du projet](#architecture-du-projet)
3. [Pré-requis](#pré-requis)
4. [Installation & Compilation](#installation--compilation)
5. [Utilisation — Interface Web](#utilisation--interface-web)
6. [Utilisation — Version Terminal (CLI)](#utilisation--version-terminal-cli)
7. [Format des datasets](#format-des-datasets)
8. [Exemples de commandes](#exemples-de-commandes)
9. [Dépannage](#dépannage)
10. [Ressources](#ressources)

---

## Vue d'ensemble

Ce projet implémente deux algorithmes classiques d'apprentissage automatique avec MPI (Message Passing Interface) pour l'accélération parallèle :

| Algorithme | Type | Dataset | Métriques |
|------------|------|---------|-----------|
| **k-NN** (k plus proches voisins) | Classification | `dataset_classification.csv` | Précision (%) |
| **K-Means** | Clustering | `dataset_clustering.csv` | Itérations de convergence |

Les objectifs sont :
- Comparer le temps séquentiel (un processus) avec le temps parallèle (plusieurs processus).
- Mesurer le **speedup** (`S = Ts / Tp`) et l'**efficacité** (`E = S / p × 100`).
- Fournir une interface web interactive et une interface terminal de secours.

---

## Architecture du projet

```
.
├── kmeans_mpi.c               # Implémentation parallèle K-Means (C + MPI)
├── knn_mpi.c                  # Implémentation parallèle k-NN (C + MPI)
├── interface_mpi.html         # Interface web interactive (HTML/CSS/JS)
├── server.py                  # Serveur Flask (API REST + websocket)
├── cli_run.py                 # Version terminal de secours (Python 3)
├── dataset_classification.csv # Données de classification (Iris similaire)
├── dataset_clustering.csv     # Données de clustering (concentrique similaire)
├── rapport_MPI_final.pdf      # Rapport scientifique (PDF)
├── requirements.txt           # Dépendances Python
└── README.md                  # Ce fichier
```

---

## Pré-requis

### Système
- Linux (testé sur Ubuntu 22.04/24.04)
- Python 3.8 ou supérieur
- OpenMPI 4.x ou 5.x (`mpicc`, `mpirun`)

### Packages système (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install -y build-essential openmpi-bin libopenmpi-dev python3 python3-pip
```

### Packages Python
```bash
pip3 install -r requirements.txt
```
> Ou manuellement : `pip3 install numpy scipy matplotlib flask flask-cors`

### Vérification
```bash
mpicc --version        # doit afficher gcc + OpenMPI
mpirun --version       # doit afficher la version OpenMPI
python3 --version      # >= 3.8
```

---

## Installation & Compilation

### 1. Cloner ou extraire le projet
```bash
cd /chemin/vers/projet_MPI
```

### 2. Compiler les programmes C
```bash
# k-NN
mpicc -O2 -o knn_mpi knn_mpi.c -lm

# K-Means
mpicc -O2 -o kmeans_mpi kmeans_mpi.c -lm
```

> **Remarque :** Les programmes C utilisent déjà `__linux__` pour Linux et `__MACH__` pour macOS. Sous macOS avec Homebrew, adaptez le chemin d'inclusion si nécessaire.

### 3. Vérifier les fichiers
```bash
ls -l knn_mpi kmeans_mpi dataset_*.csv
```

---

## Utilisation — Interface Web

### Lancer le serveur
```bash
python3 server.py
```

Le terminal affichera :
```
Running on http://127.0.0.1:5000
```

### Accéder à l'interface
Ouvrez votre navigateur à l'adresse :
```
http://127.0.0.1:5000
```

### Fonctionnalités de l'interface web
1. **Sélection de l'algorithme** : boutons "k-NN" ou "K-Means"
2. **Téléversement de dataset** : glissez-déposez ou sélectionnez un fichier CSV
3. **Paramètre MPI** : nombre de processus (curseur 1–20)
4. **Lancer l'expérience** : appelle `mpirun` côté serveur
5. **Résultats** : temps séquentiel, temps parallèle, speedup, efficacité
6. **Graphique** : courbes de speedup et efficacité selon le nombre de processus
7. **Historique** : tableau des exécutions précédentes
8. **Réinitialiser** : bouton pour effacer l'historique et le graphique

---

## Utilisation — Version Terminal (CLI)

La version CLI est une **alternative de secours** si l'interface web est indisponible.

### Commandes rapides

```bash
# k-NN avec dataset de classification (4 processus)
python3 cli_run.py --algo knn --dataset dataset_classification.csv --np 4

# K-Means avec dataset de clustering (8 processus)
python3 cli_run.py --algo kmeans --dataset dataset_clustering.csv --np 8

# Mode interactif (menu terminal)
python3 cli_run.py --interactive
```

### Options de `cli_run.py`

| Option | Description | Valeur par défaut |
|--------|-------------|-------------------|
| `--algo {knn,kmeans}` | Algorithme à exécuter | `knn` |
| `--dataset FICHIER` | Chemin du dataset CSV | auto (classif. pour knn, cluster. pour kmeans) |
| `--np N` | Nombre de processus MPI | `4` |
| `--interactive, -i` | Mode menu interactif | `False` |
| `--export-json FICHIER` | Exporter l'historique en JSON | — |
| `--export-csv FICHIER` | Exporter l'historique en CSV | — |

### Exemple — Mode interactif

```bash
$ python3 cli_run.py -i

╔══════════════════════════════════════════════════════════════╗
║            Calculs Parallèles — M1 HPC · USTHB               ║
║              Terminal MPI · k-NN & K-Means                  ║
║                    Merbouche Abdelatif                       ║
╚══════════════════════════════════════════════════════════════╝

── Menu Principal ──
  1. Lancer une exécution
  2. Voir l'historique
  3. Exporter l'historique (JSON)
  4. Exporter l'historique (CSV)
  5. Vérifier les fichiers requis
  0. Quitter

Choix : 1
  Algorithme (knn / kmeans) [knn] : 
  Dataset [dataset_classification.csv] : 
  Nombre de processus [4] : 8

$ mpirun --oversubscribe --allow-run-as-root -np 8 knn_mpi dataset_classification.csv
Exécution en cours... terminée ✓

══════ Résultats de l'exécution ══════
  Algorithme : k-NN
  Dataset    : Classification (dataset_classification.csv)
  Processus  : 8

  Ts (séquentiel) : 0.0211 s
  Tp (parallèle)  : 0.0059 s
  S (speedup)     : 3.58
  E (efficacité)  : 44.8 %
  Précision       : 93.33 %
═══════════════════════════════════════
```

---

## Format des datasets

### Classification (`dataset_classification.csv`)
```csv
sepal_length,sepal_width,petal_length,petal_width,class
5.1,3.5,1.4,0.2,Iris-setosa
4.9,3.0,1.4,0.2,Iris-setosa
...
```
- La dernière colonne est la **classe** (chaîne de caractères).
- Les colonnes précédentes sont les **caractéristiques numériques**.

### Clustering (`dataset_clustering.csv`)
```csv
x,y
1.2,3.4
5.6,7.8
...
```
- Toutes les colonnes sont **numériques** (pas de colonne de classe).
- Le séparateur est la virgule.

### Dataset personnalisé
Vous pouvez téléverser n'importe quel CSV respectant le format ci-dessus via l'interface web.

---

## Exemples de commandes

### Compilation
```bash
mpicc -O2 -o knn_mpi knn_mpi.c -lm
mpicc -O2 -o kmeans_mpi kmeans_mpi.c -lm
```

### Exécution directe (sans Python)
```bash
# Séquentiel (1 processus)
mpirun --oversubscribe --allow-run-as-root -np 1 ./knn_mpi dataset_classification.csv

# Parallèle (4 processus)
mpirun --oversubscribe --allow-run-as-root -np 4 ./knn_mpi dataset_classification.csv
```

### Web
```bash
python3 server.py
# Ouvrir http://127.0.0.1:5000 dans le navigateur
```

### CLI
```bash
# Une seule exécution
python3 cli_run.py --algo knn --np 4

# Mode interactif complet
python3 cli_run.py --interactive

# Exécution + export JSON
python3 cli_run.py --algo kmeans --np 8 --export-json results.json
```

---

## Dépannage

| Problème | Cause probable | Solution |
|----------|---------------|----------|
| `mpicc: command not found` | OpenMPI non installé | `sudo apt install openmpi-bin libopenmpi-dev` |
| `mpirun not found` | OpenMPI non installé | Voir ci-dessus |
| `Permission denied` | Binaires non exécutables | `chmod +x knn_mpi kmeans_mpi` |
| `unable to find mpicc` (macOS) | Chemin non configuré | `export PATH=/usr/local/opt/open-mpi/bin:$PATH` |
| `There are not enough slots` | Pas assez de cœurs | Ajoutez `--oversubscribe` à `mpirun` |
| Flask ne démarre pas | Port 5000 occupé | Modifiez `app.run(port=5000)` dans `server.py` |
| Résultats vides | Dataset mal formaté | Vérifiez le CSV (virgules, dernière colonne) |
| Précision = 0 % | Caractéristiques mal normalisées | Utilisez le format standard Iris |

---

## Ressources

- **OpenMPI** : https://www.open-mpi.org/
- **Documentation MPI** : https://www.mpi-forum.org/docs/
- **MPI Tutorial (LLNL)** : https://computing.llnl.gov/tutorials/mpi/
- **Cours Calculs Parallèles, Mme Baba Ali, USTHB**

---

> **Licence & usage académique** — Ce projet est destiné à un usage pédagogique dans le cadre du Master 1 HPC de l'USTHB.
