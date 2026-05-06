#!/usr/bin/env python3
"""
===============================================================
 cli_run.py  —  Interface Terminal pour les expériences MPI
===============================================================
 Auteur     : Merbouche Abdelatif
 Module     : Calculs Parallèles — M1 HPC, USTHB
 Encadrante : Mme Baba Ali
 Année      : 2025/2026

 Description :
   Version fallback (ligne de commande) du projet MPI.
   Si l'interface web est indisponible, ce script permet de
   lancer directement les programmes C compilés (knn_mpi,
   kmeans_mpi) avec mpirun et d'afficher les résultats dans
   le terminal.

   Fonctionnalités :
   - Exécution directe des algorithmes k-NN et K-Means
   - Paramétrage du nombre de processus MPI
   - Choix du dataset (classification ou clustering)
   - Affichage formaté des résultats (temps, accélération,
     efficacité)
   - Historique des exécutions dans la session
   - Export des résultats en JSON ou CSV

 Pré-requis :
   - OpenMPI installé (mpicc, mpirun)
   - Les binaires compilés : knn_mpi et kmeans_mpi
   - Python 3.8+

 Utilisation :
   python3 cli_run.py --algo knn --dataset dataset_classification.csv --np 4
   python3 cli_run.py --algo kmeans --dataset dataset_clustering.csv --np 8
   python3 cli_run.py --interactive
===============================================================
"""

import argparse
import json
import csv
import os
import re
import subprocess
import sys
from datetime import datetime
from typing import Dict, List, Optional

# ── Constantes ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ALGO = "knn"
DEFAULT_DATASET = "dataset_classification.csv"
DEFAULT_NP = 4
HISTORY: List[Dict] = []

# ── Couleurs terminal ───────────────────────────────────────
class Color:
    BLUE    = "\033[94m"
    GREEN   = "\033[92m"
    ORANGE  = "\033[93m"
    RED     = "\033[91m"
    CYAN    = "\033[96m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    RESET   = "\033[0m"


def banner():
    """Affiche l'en-tête du programme."""
    print(f"""
{Color.CYAN}╔══════════════════════════════════════════════════════════════╗
║            Calculs Parallèles — M1 HPC · USTHB               ║
║              Terminal MPI · k-NN & K-Means                  ║
║                    Merbouche Abdelatif                       ║
╚══════════════════════════════════════════════════════════════╝{Color.RESET}
""")


def check_binaries() -> Dict[str, bool]:
    """Vérifie la présence des exécutables et datasets."""
    files = {
        "knn_mpi": os.path.join(BASE_DIR, "knn_mpi"),
        "kmeans_mpi": os.path.join(BASE_DIR, "kmeans_mpi"),
        "dataset_classification.csv": os.path.join(BASE_DIR, "dataset_classification.csv"),
        "dataset_clustering.csv": os.path.join(BASE_DIR, "dataset_clustering.csv"),
    }
    return {name: os.path.isfile(path) for name, path in files.items()}


def print_status(status: Dict[str, bool]):
    """Affiche l'état des fichiers requis."""
    print(f"{Color.BOLD}État des fichiers requis :{Color.RESET}")
    for name, ok in status.items():
        sym = f"{Color.GREEN}✓{Color.RESET}" if ok else f"{Color.RED}✗{Color.RESET}"
        print(f"  {sym} {name}")
    print()


def run_mpi(algo: str, dataset: str, np_val: int) -> Optional[Dict]:
    """
    Exécute mpirun avec le binaire choisi et parse la sortie.
    Retourne un dict avec les résultats ou None en cas d'erreur.
    """
    exe_path = os.path.join(BASE_DIR, f"{algo}_mpi")
    dataset_path = os.path.join(BASE_DIR, dataset)

    if not os.path.isfile(exe_path):
        print(f"{Color.RED}Erreur : exécutable '{algo}_mpi' introuvable.{Color.RESET}")
        print(f"Compilez d'abord : mpicc -O2 -o {algo}_mpi {algo}_mpi.c -lm")
        return None

    if not os.path.isfile(dataset_path):
        print(f"{Color.RED}Erreur : dataset '{dataset}' introuvable.{Color.RESET}")
        return None

    cmd = [
        "mpirun", "--oversubscribe", "--allow-run-as-root",
        "-np", str(np_val), exe_path, dataset_path
    ]

    print(f"{Color.DIM}$ {' '.join(cmd)}{Color.RESET}\n")
    print("Exécution en cours...", end=" ", flush=True)

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120, cwd=BASE_DIR)
        output = result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        print(f"\n{Color.RED}Timeout (>120s){Color.RESET}")
        return None
    except FileNotFoundError:
        print(f"\n{Color.RED}mpirun introuvable. Installez OpenMPI.{Color.RESET}")
        return None
    except Exception as e:
        print(f"\n{Color.RED}Erreur : {e}{Color.RESET}")
        return None

    if result.returncode != 0:
        print(f"\n{Color.RED}Le programme s'est terminé avec l'erreur {result.returncode}{Color.RESET}")
        print(output)
        return None

    print(f"{Color.GREEN}terminée ✓{Color.RESET}\n")

    # ── Parsing ───────────────────────────────────────────
    ts = None
    m = re.search(r'\[S[ée]quentiel\].*?Ts\s*=\s*([\d.]+)', output)
    if m:
        ts = float(m.group(1))

    tp = speedup = efficacite = None
    m = re.search(r'\[Parall[èe]le\].*?Tp\s*=\s*([\d.]+).*?S\s*=\s*([\d.]+).*?E\s*=\s*([\d.]+)', output)
    if m:
        tp = float(m.group(1))
        speedup = float(m.group(2))
        efficacite = float(m.group(3))

    iterations = None
    m = re.search(r'It[ée]rations\s*=\s*(\d+)', output)
    if m:
        iterations = int(m.group(1))

    precision = None
    m = re.search(r'Pr[ée]cision\s*=\s*([\d.]+)', output)
    if m:
        precision = float(m.group(1))

    if ts is None or tp is None:
        print(f"{Color.RED}Impossible de parser la sortie du programme.{Color.RESET}")
        print("Sortie brute :")
        print(output)
        return None

    return {
        "algo": algo,
        "dataset": dataset,
        "np": np_val,
        "ts": ts,
        "tp": tp,
        "speedup": speedup,
        "efficacite": efficacite,
        "iterations": iterations,
        "precision": precision,
        "output": output.strip(),
        "timestamp": datetime.now().isoformat(),
    }


def print_results(res: Dict):
    """Affiche les résultats de manière formatée."""
    algo_name = "k-NN" if res["algo"] == "knn" else "K-Means"
    ds_name = "Classification" if "class" in res["dataset"] else "Clustering"

    print(f"{Color.BOLD}══════ Résultats de l'exécution ══════{Color.RESET}")
    print(f"  Algorithme : {Color.CYAN}{algo_name}{Color.RESET}")
    print(f"  Dataset    : {Color.CYAN}{ds_name}{Color.RESET} ({res['dataset']})")
    print(f"  Processus  : {Color.CYAN}{res['np']}{Color.RESET}")
    print()
    print(f"  {Color.BLUE}Ts (séquentiel){Color.RESET} : {res['ts']:.4f} s")
    print(f"  {Color.BLUE}Tp (parallèle) {Color.RESET} : {res['tp']:.4f} s")
    print(f"  {Color.GREEN}S (speedup)    {Color.RESET} : {res['speedup']:.3f}")
    print(f"  {Color.ORANGE}E (efficacité) {Color.RESET} : {res['efficacite']:.1f} %")
    if res["iterations"] is not None:
        print(f"  Itérations   : {res['iterations']}")
    if res["precision"] is not None:
        print(f"  Précision    : {res['precision']:.2f} %")
    print(f"{Color.BOLD}═══════════════════════════════════════{Color.RESET}\n")


def add_to_history(res: Dict):
    """Ajoute un résultat à l'historique de session."""
    HISTORY.append(res)


def print_history():
    """Affiche l'historique des exécutions."""
    if not HISTORY:
        print(f"{Color.DIM}Aucune exécution dans l'historique.{Color.RESET}\n")
        return

    print(f"{Color.BOLD}═══════════ Historique des exécutions ═══════════{Color.RESET}")
    print(f"{'#':<4} {'Algo':<8} {'Dataset':<14} {'p':<4} {'Ts(s)':<10} {'Tp(s)':<10} {'S':<8} {'E%':<8}")
    print("-" * 70)
    for i, r in enumerate(HISTORY, 1):
        ds_short = "Class" if "class" in r["dataset"] else "Clust"
        algo = "k-NN" if r["algo"] == "knn" else "K-Means"
        print(
            f"{i:<4} {algo:<8} {ds_short:<14} {r['np']:<4} "
            f"{r['ts']:<10.4f} {r['tp']:<10.4f} {r['speedup']:<8.3f} {r['efficacite']:<8.1f}"
        )
    print()


def export_json(path: str):
    """Exporte l'historique au format JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(HISTORY, f, indent=2, ensure_ascii=False)
    print(f"{Color.GREEN}Historique exporté vers {path}{Color.RESET}\n")


def export_csv(path: str):
    """Exporte l'historique au format CSV."""
    if not HISTORY:
        print(f"{Color.RED}Aucune donnée à exporter.{Color.RESET}\n")
        return
    keys = ["timestamp", "algo", "dataset", "np", "ts", "tp", "speedup", "efficacite", "iterations", "precision"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in HISTORY:
            writer.writerow({k: row.get(k, "") for k in keys})
    print(f"{Color.GREEN}Historique exporté vers {path}{Color.RESET}\n")


def interactive_mode():
    """Mode interactif : menu terminal."""
    banner()
    status = check_binaries()
    print_status(status)

    if not all(status.values()):
        print(f"{Color.RED}Certains fichiers sont manquants. Vérifiez la compilation et les datasets.{Color.RESET}\n")

    while True:
        print(f"{Color.BOLD}── Menu Principal ──{Color.RESET}")
        print("  1. Lancer une exécution")
        print("  2. Voir l'historique")
        print("  3. Exporter l'historique (JSON)")
        print("  4. Exporter l'historique (CSV)")
        print("  5. Vérifier les fichiers requis")
        print("  0. Quitter")
        choice = input(f"\n{Color.BOLD}Choix : {Color.RESET}").strip()

        if choice == "1":
            print()
            algo = input("  Algorithme (knn / kmeans) [knn] : ").strip() or "knn"
            if algo not in ("knn", "kmeans"):
                print(f"{Color.RED}Algorithme invalide.{Color.RESET}\n")
                continue

            if algo == "knn":
                default_ds = "dataset_classification.csv"
            else:
                default_ds = "dataset_clustering.csv"
            ds = input(f"  Dataset [{default_ds}] : ").strip() or default_ds

            np_str = input(f"  Nombre de processus [{DEFAULT_NP}] : ").strip()
            np_val = int(np_str) if np_str.isdigit() else DEFAULT_NP

            print()
            res = run_mpi(algo, ds, np_val)
            if res:
                print_results(res)
                add_to_history(res)

        elif choice == "2":
            print()
            print_history()

        elif choice == "3":
            print()
            path = input("  Nom du fichier [results.json] : ").strip() or "results.json"
            export_json(os.path.join(BASE_DIR, path))

        elif choice == "4":
            print()
            path = input("  Nom du fichier [results.csv] : ").strip() or "results.csv"
            export_csv(os.path.join(BASE_DIR, path))

        elif choice == "5":
            print()
            status = check_binaries()
            print_status(status)

        elif choice == "0":
            print(f"\n{Color.CYAN}Au revoir !{Color.RESET}\n")
            break

        else:
            print(f"\n{Color.RED}Choix invalide.{Color.RESET}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Interface terminal pour expériences MPI (k-NN & K-Means)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples :
  python3 cli_run.py --algo knn --np 4
  python3 cli_run.py --algo kmeans --dataset dataset_clustering.csv --np 8
  python3 cli_run.py --interactive
        """
    )
    parser.add_argument("--algo", choices=["knn", "kmeans"], default=DEFAULT_ALGO,
                        help="Algorithme à exécuter (knn ou kmeans)")
    parser.add_argument("--dataset", default=None,
                        help="Chemin du dataset CSV")
    parser.add_argument("--np", type=int, default=DEFAULT_NP,
                        help="Nombre de processus MPI")
    parser.add_argument("--interactive", "-i", action="store_true",
                        help="Lancer le mode interactif (menu terminal)")
    parser.add_argument("--export-json", metavar="FILE",
                        help="Exporter l'historique en JSON à la fin")
    parser.add_argument("--export-csv", metavar="FILE",
                        help="Exporter l'historique en CSV à la fin")
    args = parser.parse_args()

    if args.interactive:
        interactive_mode()
        return

    banner()
    status = check_binaries()
    print_status(status)

    algo = args.algo
    dataset = args.dataset
    if dataset is None:
        dataset = "dataset_classification.csv" if algo == "knn" else "dataset_clustering.csv"
    np_val = args.np

    res = run_mpi(algo, dataset, np_val)
    if res:
        print_results(res)
        add_to_history(res)
        print_history()

        if args.export_json:
            export_json(args.export_json)
        if args.export_csv:
            export_csv(args.export_csv)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
