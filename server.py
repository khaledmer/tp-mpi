"""
===============================================================
 server.py  —  Serveur Flask pour Interface MPI
===============================================================
 Auteur     : Merbouche Abdelatif
 Module     : Calculs Parallèles — M1 HPC, USTHB
 Encadrante : Mme Baba Ali
 Année      : 2025/2026

 Description :
   Serveur backend léger pour l'interface web du projet MPI.
   Expose une API REST sur le port 5000 pour :
   - Servir l'interface HTML statique
   - Exécuter les programmes C compilés via mpirun
   - Récupérer et parser les résultats (temps, speedup, efficacité)
   - Accepter le téléversement de datasets personnalisés

 Routes :
   GET  /              → Interface web (interface_mpi.html)
   POST /run           → Lance une expérience MPI et retourne JSON

 Pré-requis :
   - Les binaires `knn_mpi` et `kmeans_mpi` compilés
   - Les datasets `dataset_classification.csv` et `dataset_clustering.csv`
   - OpenMPI installé (mpirun disponible)

 Lancement :
   python3 server.py
   # puis ouvrir http://127.0.0.1:5000 dans le navigateur
===============================================================
"""

import os
import re
import subprocess
import tempfile
from datetime import datetime

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# ── Configuration ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__)
CORS(app)

DEFAULT_DATASETS = {
    "dataset_classification.csv": os.path.join(BASE_DIR, "dataset_classification.csv"),
    "dataset_clustering.csv": os.path.join(BASE_DIR, "dataset_clustering.csv"),
}

# ── Helpers ───────────────────────────────────────────────────
def parse_output(output: str, algorithm: str) -> dict:
    """
    Parse la sortie texte des programmes C pour en extraire
    Ts, Tp, S, E, ainsi que la métrique spécifique (précision ou itérations).
    """
    result = {
        "ts": 0.0,
        "tp": 0.0,
        "speedup": 0.0,
        "efficiency": 0.0,
    }

    # Temps séquentiel Ts
    m = re.search(r"\[S[ée]quentiel\].*?Ts\s*=\s*([\d.]+)", output, re.S)
    if m:
        result["ts"] = float(m.group(1))

    # Temps parallèle Tp, Speedup S, Efficacité E
    m = re.search(r"\[Parall[èe]le\].*?Tp\s*=\s*([\d.]+).*?S\s*=\s*([\d.]+).*?E\s*=\s*([\d.]+)", output, re.S)
    if m:
        result["tp"] = float(m.group(1))
        result["speedup"] = float(m.group(2))
        result["efficiency"] = float(m.group(3))

    # Métrique spécifique à l'algorithme
    if algorithm == "knn":
        m = re.search(r"Pr[ée]cision\s*=\s*([\d.]+)", output)
        if m:
            result["accuracy"] = float(m.group(1))
    else:
        m = re.search(r"It[ée]rations\s*=\s*(\d+)", output)
        if m:
            result["iterations"] = int(m.group(1))

    return result


def get_dataset_path(dataset_name: str, dataset_content: str = None, dataset_filename: str = None) -> str:
    """
    Retourne le chemin absolu du dataset à utiliser.
    Si dataset_content est fourni, crée un fichier temporaire.
    """
    # 1. Dataset personnalisé uploadé
    if dataset_content is not None and dataset_filename:
        # Créer un fichier temporaire unique
        tmp_name = f"uploaded_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{dataset_filename}"
        tmp_path = os.path.join(BASE_DIR, tmp_name)
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(dataset_content)
        return tmp_path

    # 2. Dataset prédéfini
    if dataset_name in DEFAULT_DATASETS:
        return DEFAULT_DATASETS[dataset_name]

    # 3. Chemin direct (si l'utilisateur passe un chemin absolu/relatif)
    if os.path.isfile(dataset_name):
        return dataset_name

    # 4. Recherche dans le répertoire courant
    local_path = os.path.join(BASE_DIR, dataset_name)
    if os.path.isfile(local_path):
        return local_path

    return None


# ── Routes ────────────────────────────────────────────────────
@app.route("/")
def index():
    """Sert l'interface web principale."""
    return send_from_directory(BASE_DIR, "interface_mpi.html")


@app.route("/run", methods=["POST"])
def run_experiment():
    """
    Exécute une expérience MPI selon les paramètres JSON reçus :
      - algorithm   : "knn" ou "kmeans"
      - dataset     : nom du fichier dataset (optionnel si upload)
      - processes   : nombre de processus MPI
      - dataset_content : contenu texte du CSV uploadé (optionnel)
      - dataset_filename : nom original du fichier uploadé (optionnel)

    Retourne un JSON avec les résultats parsés.
    """
    data = request.get_json(force=True)

    algorithm = data.get("algorithm", "knn")
    dataset_name = data.get("dataset", "dataset_classification.csv")
    processes = int(data.get("processes", 4))
    dataset_content = data.get("dataset_content")
    dataset_filename = data.get("dataset_filename")

    # Validation de l'algorithme
    if algorithm not in ("knn", "kmeans"):
        return jsonify({"error": "Invalid algorithm. Use 'knn' or 'kmeans'."}), 400

    # Validation du binaire
    exe_path = os.path.join(BASE_DIR, f"{algorithm}_mpi")
    if not os.path.isfile(exe_path):
        return jsonify({
            "error": f"Executable '{algorithm}_mpi' not found. Compile with: mpicc -O2 -o {algorithm}_mpi {algorithm}_mpi.c -lm"
        }), 500

    # Résolution du chemin du dataset
    dataset_path = get_dataset_path(dataset_name, dataset_content, dataset_filename)
    if dataset_path is None or not os.path.isfile(dataset_path):
        return jsonify({"error": f"Dataset '{dataset_name}' not found."}), 400

    # Construction de la commande mpirun
    cmd = [
        "mpirun",
        "--oversubscribe",
        "--allow-run-as-root",
        "-np", str(processes),
        exe_path,
        dataset_path,
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
            cwd=BASE_DIR,
        )
        output = result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return jsonify({"error": "MPI execution timed out (>120s)."}), 504
    except FileNotFoundError:
        return jsonify({"error": "mpirun not found. Please install OpenMPI."}), 500
    except Exception as exc:
        return jsonify({"error": f"Execution failed: {str(exc)}"}), 500

    if result.returncode != 0:
        return jsonify({
            "error": f"MPI program exited with code {result.returncode}",
            "details": output,
        }), 500

    # Parsing des résultats
    parsed = parse_output(output, algorithm)
    parsed["algorithm"] = algorithm
    parsed["dataset"] = dataset_filename or dataset_name
    parsed["processes"] = processes
    parsed["raw_output"] = output
    parsed["timestamp"] = datetime.now().isoformat()

    return jsonify(parsed)


# ── Point d'entrée ────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  Serveur MPI — Interface Web")
    print("  Auteur : Merbouche Abdelatif · M1 HPC · USTHB")
    print("=" * 60)
    print(f"  Répertoire : {BASE_DIR}")
    print("  Démarrage sur http://127.0.0.1:5000")
    print("=" * 60)
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
   
