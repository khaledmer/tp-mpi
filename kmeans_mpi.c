/*
 * ============================================================
 * kmeans_mpi.c  —  K-Means Parallèle MPI avec accélération
 * ============================================================
 * Auteur     : Merbouche Abdelatif
 * Module     : Calculs Parallèles — M1 HPC, USTHB
 * Encadrante : Mme Baba Ali
 * Année      : 2025/2026
 *
 * Description :
 *   Implémentation parallèle MPI de l'algorithme K-Means (Lloyd).
 *
 *   Schéma Maître/Esclave :
 *   - Maître (rang 0) :
 *       1. Lit le dataset CSV
 *       2. Exécute K-Means SÉQUENTIEL pour obtenir Ts
 *       3. Distribue les données via MPI_Scatterv
 *       4. Participe au calcul parallèle
 *       5. Collecte via MPI_Allreduce (global à chaque itération)
 *       6. Affiche l'accélération S = Ts / Tp
 *   - Esclaves :
 *       1. Reçoivent leur portion via MPI_Scatterv
 *       2. Participent aux MPI_Allreduce à chaque itération
 *
 * Communication principale : MPI_Allreduce O(K*d) par itération,
 * indépendant de N → très bonne scalabilité théorique.
 *
 * Compilation : mpicc -O2 -o kmeans_mpi kmeans_mpi.c -lm
 * Exécution   : mpirun --oversubscribe -np <P> ./kmeans_mpi [dataset.csv]
 * ============================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <float.h>
#include <mpi.h>

/* ---- Paramètres ---- */
#define MAX_POINTS  15000
#define MAX_DIM     10
#define K            5
#define MAX_ITER   200
#define EPSILON    1e-6

static int DIM = 4;  /* mis à jour après lecture */
static int N   = 0;

/* ============================================================
 * load_csv_unlabeled : charge un CSV sans colonne label
 * (ou ignore la dernière colonne si elle existe)
 * ============================================================ */
int load_csv_unlabeled(const char *path, double *data, int max_n, int *dim_out) {
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "Erreur ouverture %s\n", path); return -1; }
    char line[4096];
    fgets(line, sizeof(line), f); /* entête */
    int ncols = 0;
    char tmp[4096]; strcpy(tmp, line);
    char *tok = strtok(tmp, ",\n");
    while (tok) { ncols++; tok = strtok(NULL, ",\n"); }
    /* Pour le clustering on utilise toutes les colonnes numériques sauf la dernière */
    *dim_out = (ncols > 1) ? ncols - 1 : ncols;
    int n = 0;
    while (n < max_n && fgets(line, sizeof(line), f)) {
        char *p = line;
        for (int d = 0; d < *dim_out; d++) {
            data[n * MAX_DIM + d] = strtod(p, &p);
            if (*p == ',') p++;
        }
        n++;
    }
    fclose(f);
    return n;
}

/* ============================================================
 * dist2 : carré de la distance euclidienne
 * ============================================================ */
double dist2(const double *a, const double *b) {
    double s = 0.0;
    for (int d = 0; d < DIM; d++) { double x = a[d]-b[d]; s += x*x; }
    return s;
}

/* ============================================================
 * nearest_centroid : retourne l'indice du centroïde le plus proche
 * ============================================================ */
int nearest_centroid(const double *point, const double *centroids) {
    double best = DBL_MAX; int bi = 0;
    for (int c = 0; c < K; c++) {
        double d = dist2(point, &centroids[c * MAX_DIM]);
        if (d < best) { best = d; bi = c; }
    }
    return bi;
}

/* ============================================================
 * kmeans_sequential : K-Means séquentiel complet
 * Retourne le temps d'exécution.
 * ============================================================ */
double kmeans_sequential(const double *data, int n, double *centroids_out,
                         int *labels_out) {
    double centroids[K * MAX_DIM];
    /* Initialisation : K premiers points comme centroïdes */
    for (int c = 0; c < K; c++)
        memcpy(&centroids[c * MAX_DIM], &data[c * MAX_DIM], DIM * sizeof(double));

    double t0 = MPI_Wtime();
    for (int iter = 0; iter < MAX_ITER; iter++) {
        double sum[K * MAX_DIM]; int cnt[K];
        memset(sum, 0, sizeof(sum)); memset(cnt, 0, sizeof(cnt));

        /* Affectation */
        for (int i = 0; i < n; i++) {
            int c = nearest_centroid(&data[i * MAX_DIM], centroids);
            if (labels_out) labels_out[i] = c;
            cnt[c]++;
            for (int d = 0; d < DIM; d++)
                sum[c * MAX_DIM + d] += data[i * MAX_DIM + d];
        }
        /* Mise à jour des centroïdes + test de convergence */
        double shift = 0.0;
        for (int c = 0; c < K; c++) {
            if (cnt[c] == 0) continue;
            for (int d = 0; d < DIM; d++) {
                double nv = sum[c * MAX_DIM + d] / cnt[c];
                double dv = fabs(nv - centroids[c * MAX_DIM + d]);
                if (dv > shift) shift = dv;
                centroids[c * MAX_DIM + d] = nv;
            }
        }
        if (shift < EPSILON) break;
    }
    double elapsed = MPI_Wtime() - t0;
    if (centroids_out)
        memcpy(centroids_out, centroids, K * MAX_DIM * sizeof(double));
    return elapsed;
}

/* ============================================================
 *                        MAIN
 * ============================================================ */
int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    static double all_data[MAX_POINTS * MAX_DIM];
    double Ts = 0.0;

    /* ================================================================
     * MAÎTRE : lecture + phase séquentielle
     * ================================================================ */
    if (rank == 0) {
        const char *path = (argc > 1) ? argv[1] : "dataset_clustering.csv";
        N = load_csv_unlabeled(path, all_data, MAX_POINTS, &DIM);
        if (N < 0) { MPI_Abort(MPI_COMM_WORLD, 1); }

        printf("=== K-Means MPI — USTHB M1 HPC | Encadrante : Mme Baba Ali ===\n");
        printf("Dataset=%s | N=%d | dim=%d | K=%d | p=%d\n\n",
               path, N, DIM, K, size);

        /* Phase séquentielle */
        double seq_centroids[K * MAX_DIM];
        Ts = kmeans_sequential(all_data, N, seq_centroids, NULL);
        printf("[Séquentiel] Ts = %.4f s\n\n", Ts);

        /* Diffusion des métadonnées */
        MPI_Bcast(&Ts,  1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(&DIM, 1, MPI_INT,    0, MPI_COMM_WORLD);
        MPI_Bcast(&N,   1, MPI_INT,    0, MPI_COMM_WORLD);
    } else {
        MPI_Bcast(&Ts,  1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(&DIM, 1, MPI_INT,    0, MPI_COMM_WORLD);
        MPI_Bcast(&N,   1, MPI_INT,    0, MPI_COMM_WORLD);
    }

    /* ---- Répartition des points ---- */
    int base = N / size, rem = N % size;
    int *scounts = (int*)malloc(size * sizeof(int));
    int *displs  = (int*)malloc(size * sizeof(int));
    int *sb      = (int*)malloc(size * sizeof(int));
    int *db      = (int*)malloc(size * sizeof(int));
    for (int i = 0; i < size; i++) scounts[i] = (i < rem) ? base+1 : base;
    displs[0] = 0;
    for (int i = 1; i < size; i++) displs[i] = displs[i-1] + scounts[i-1];
    for (int i = 0; i < size; i++) {
        sb[i] = scounts[i] * DIM * sizeof(double);
        db[i] = displs[i]  * DIM * sizeof(double);
    }

    int n_local = scounts[rank];
    double *local_data = (double*)malloc(n_local * MAX_DIM * sizeof(double));

    /* Scatter des données */
    MPI_Scatterv(all_data, sb, db, MPI_BYTE,
                 local_data, n_local * DIM * sizeof(double), MPI_BYTE,
                 0, MPI_COMM_WORLD);

    /* ---- Initialisation des centroïdes (K premiers points, diffusés) ---- */
    double centroids[K * MAX_DIM];
    if (rank == 0)
        for (int c = 0; c < K; c++)
            memcpy(&centroids[c * MAX_DIM], &all_data[c * MAX_DIM], DIM * sizeof(double));
    MPI_Bcast(centroids, K * MAX_DIM, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    /* ================================================================
     * PHASE PARALLÈLE — Boucle K-Means
     * ================================================================ */
    MPI_Barrier(MPI_COMM_WORLD);
    double tp_start = MPI_Wtime();

    int iter = 0;
    double shift = DBL_MAX;
    while (iter < MAX_ITER && shift > EPSILON) {
        /* Accumulation locale */
        double local_sum[K * MAX_DIM]; int local_cnt[K];
        memset(local_sum, 0, sizeof(local_sum)); memset(local_cnt, 0, sizeof(local_cnt));
        for (int i = 0; i < n_local; i++) {
            int c = nearest_centroid(&local_data[i * MAX_DIM], centroids);
            local_cnt[c]++;
            for (int d = 0; d < DIM; d++)
                local_sum[c * MAX_DIM + d] += local_data[i * MAX_DIM + d];
        }

        /* MPI_Allreduce : agrégation globale des sommes et compteurs */
        double global_sum[K * MAX_DIM]; int global_cnt[K];
        MPI_Allreduce(local_sum, global_sum, K * MAX_DIM, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(local_cnt, global_cnt, K,           MPI_INT,    MPI_SUM, MPI_COMM_WORLD);

        /* Mise à jour des centroïdes et calcul de la convergence */
        shift = 0.0;
        for (int c = 0; c < K; c++) {
            if (global_cnt[c] == 0) continue;
            for (int d = 0; d < DIM; d++) {
                double nv = global_sum[c * MAX_DIM + d] / global_cnt[c];
                double dv = fabs(nv - centroids[c * MAX_DIM + d]);
                if (dv > shift) shift = dv;
                centroids[c * MAX_DIM + d] = nv;
            }
        }
        iter++;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double Tp = MPI_Wtime() - tp_start;

    /* ---- Maître : affichage des résultats ---- */
    if (rank == 0) {
        double S = Ts / Tp;
        double E = S / size * 100.0;
        printf("[Parallèle]  p=%2d | Tp = %.4f s | Itérations = %d | "
               "Accélération S = %.3f | Efficacité E = %.1f%%\n",
               size, Tp, iter, S, E);
    }

    free(local_data); free(scounts); free(displs); free(sb); free(db);
    MPI_Finalize();
    return 0;
}
