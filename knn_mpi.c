/*
 * ============================================================
 * knn_mpi.c  —  k-NN Parallèle MPI avec accélération
 * ============================================================
 * Auteur     : Merbouche Abdelatif
 * Module     : Calculs Parallèles — M1 HPC, USTHB
 * Encadrante : Mme Baba Ali
 * Année      : 2025/2026
 *
 * Description :
 *   Implémentation parallèle MPI du k-NN (k plus proches voisins).
 *
 *   Schéma Maître/Esclave :
 *   - Maître (rang 0) :
 *       1. Lit le dataset CSV et le divise en apprentissage (80%) / test (20%)
 *       2. Exécute le k-NN SÉQUENTIEL pour obtenir le temps de référence Ts
 *       3. Diffuse l'ensemble d'apprentissage via MPI_Bcast
 *       4. Diffuse les points de test via MPI_Bcast
 *       5. Traite sa propre portion de points de test
 *       6. Collecte les prédictions via MPI_Gatherv
 *       7. Calcule et affiche l'accélération S = Ts / Tp
 *   - Esclaves (rangs 1..p-1) :
 *       1. Reçoivent les données via MPI_Bcast
 *       2. Calculent le k-NN sur leur portion locale
 *       3. Envoient les prédictions au Maître via MPI_Gatherv
 *
 * Compilation : mpicc -O2 -o knn_mpi knn_mpi.c -lm
 * Exécution   : mpirun --oversubscribe -np <P> ./knn_mpi [dataset.csv]
 * ============================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <mpi.h>

/* ---- Paramètres ---- */
#define MAX_POINTS  15000
#define MAX_DIM     10
#define NUM_CLASSES  5
#define K_NEIGHBORS  5
#define TRAIN_RATIO  0.8

/* ---- Structure d'un point de données ---- */
typedef struct {
    double feat[MAX_DIM];  /* Vecteur de caractéristiques */
    int    label;          /* Classe réelle               */
} Point;

/* ---- Variables globales mises à jour après lecture du CSV ---- */
static int DIM    = 6;
static int NTRAIN = 0;
static int NTEST  = 0;

/* ============================================================
 * load_csv : charge un fichier CSV (1ère ligne = entête)
 * Retourne le nombre de points lus, -1 si erreur.
 * ============================================================ */
int load_csv(const char *path, Point *data, int max_n, int *dim_out) {
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "Erreur ouverture %s\n", path); return -1; }
    char line[4096];
    /* Lire l'entête pour compter les colonnes (dernière = label) */
    fgets(line, sizeof(line), f);
    int ncols = 0;
    char tmp[4096]; strcpy(tmp, line);
    char *tok = strtok(tmp, ",\n");
    while (tok) { ncols++; tok = strtok(NULL, ",\n"); }
    *dim_out = ncols - 1;
    int n = 0;
    while (n < max_n && fgets(line, sizeof(line), f)) {
        char *p = line;
        for (int d = 0; d < *dim_out; d++) {
            data[n].feat[d] = strtod(p, &p);
            if (*p == ',') p++;
        }
        data[n].label = (int)strtod(p, NULL);
        n++;
    }
    fclose(f);
    return n;
}

/* ============================================================
 * dist2 : distance euclidienne au carré entre deux vecteurs
 * ============================================================ */
double dist2(const double *a, const double *b) {
    double s = 0.0;
    for (int i = 0; i < DIM; i++) { double d = a[i]-b[i]; s += d*d; }
    return s;
}

/* ============================================================
 * knn_predict : classifie un point par vote des k plus proches voisins
 * Complexité : O(Ntrain * d)
 * ============================================================ */
int knn_predict(const Point *train, int ntrain, const double *q) {
    double best_d[K_NEIGHBORS];
    int    best_l[K_NEIGHBORS];
    for (int i = 0; i < K_NEIGHBORS; i++) { best_d[i] = 1e300; best_l[i] = 0; }

    for (int i = 0; i < ntrain; i++) {
        double d = dist2(train[i].feat, q);
        /* Trouver le plus éloigné parmi les k candidats actuels */
        int worst = 0;
        for (int j = 1; j < K_NEIGHBORS; j++)
            if (best_d[j] > best_d[worst]) worst = j;
        if (d < best_d[worst]) { best_d[worst] = d; best_l[worst] = train[i].label; }
    }
    /* Vote majoritaire */
    int votes[NUM_CLASSES] = {0};
    for (int i = 0; i < K_NEIGHBORS; i++) votes[best_l[i]]++;
    int best = 0;
    for (int c = 1; c < NUM_CLASSES; c++) if (votes[c] > votes[best]) best = c;
    return best;
}

/* ============================================================
 * knn_sequential : exécute le k-NN complet séquentiellement
 * Utilisé par le Maître pour mesurer Ts (temps de référence).
 * ============================================================ */
double knn_sequential(const Point *train, int ntrain,
                      const Point *test,  int ntest, int *preds) {
    double t0 = MPI_Wtime();
    for (int i = 0; i < ntest; i++)
        preds[i] = knn_predict(train, ntrain, test[i].feat);
    return MPI_Wtime() - t0;
}

/* ============================================================
 * accuracy : taux de bonne classification (%)
 * ============================================================ */
double accuracy(const int *preds, const Point *test, int ntest) {
    int ok = 0;
    for (int i = 0; i < ntest; i++) if (preds[i] == test[i].label) ok++;
    return 100.0 * ok / ntest;
}

/* ============================================================
 *                        MAIN
 * ============================================================ */
int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    static Point train_data[MAX_POINTS];
    static Point test_data[MAX_POINTS];

    double Ts = 0.0;

    /* ================================================================
     * MAÎTRE (rang 0) : lecture, phase séquentielle, puis parallèle
     * ================================================================ */
    if (rank == 0) {
        static Point all_data[MAX_POINTS];
        const char *path = (argc > 1) ? argv[1] : "dataset_classification.csv";
        int n_total = load_csv(path, all_data, MAX_POINTS, &DIM);
        if (n_total < 0) { MPI_Abort(MPI_COMM_WORLD, 1); }

        NTRAIN = (int)(n_total * TRAIN_RATIO);
        NTEST  = n_total - NTRAIN;
        memcpy(train_data, all_data,          NTRAIN * sizeof(Point));
        memcpy(test_data,  all_data + NTRAIN, NTEST  * sizeof(Point));

        printf("=== k-NN MPI — USTHB M1 HPC | Encadrante : Mme Baba Ali ===\n");
        printf("Dataset=%s | N=%d | Ntrain=%d | Ntest=%d | dim=%d | k=%d | p=%d\n\n",
               path, n_total, NTRAIN, NTEST, DIM, K_NEIGHBORS, size);

        /* --- Phase séquentielle (référence) --- */
        int *seq_preds = (int*)malloc(NTEST * sizeof(int));
        Ts = knn_sequential(train_data, NTRAIN, test_data, NTEST, seq_preds);
        printf("[Séquentiel] Ts = %.4f s | Précision = %.2f%%\n\n",
               Ts, accuracy(seq_preds, test_data, NTEST));
        free(seq_preds);

        /* Diffusion des métadonnées et données */
        MPI_Bcast(&Ts,     1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(&DIM,    1, MPI_INT,    0, MPI_COMM_WORLD);
        MPI_Bcast(&NTRAIN, 1, MPI_INT,    0, MPI_COMM_WORLD);
        MPI_Bcast(&NTEST,  1, MPI_INT,    0, MPI_COMM_WORLD);
        MPI_Bcast(train_data, NTRAIN * sizeof(Point), MPI_BYTE, 0, MPI_COMM_WORLD);
        MPI_Bcast(test_data,  NTEST  * sizeof(Point), MPI_BYTE, 0, MPI_COMM_WORLD);

        /* Répartition des points de test */
        int *scounts = (int*)malloc(size * sizeof(int));
        int *displs  = (int*)malloc(size * sizeof(int));
        int base = NTEST / size, rem = NTEST % size;
        for (int i = 0; i < size; i++) scounts[i] = (i < rem) ? base+1 : base;
        displs[0] = 0;
        for (int i = 1; i < size; i++) displs[i] = displs[i-1] + scounts[i-1];

        int n_local = scounts[0];
        int *all_preds   = (int*)malloc(NTEST   * sizeof(int));
        int *local_preds = (int*)malloc(n_local * sizeof(int));

        /* --- Phase parallèle --- */
        MPI_Barrier(MPI_COMM_WORLD);
        double tp_start = MPI_Wtime();

        for (int i = 0; i < n_local; i++)
            local_preds[i] = knn_predict(train_data, NTRAIN, test_data[i].feat);

        MPI_Gatherv(local_preds, n_local, MPI_INT,
                    all_preds, scounts, displs, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
        double Tp = MPI_Wtime() - tp_start;

        double S = Ts / Tp;
        double E = S / size * 100.0;
        printf("[Parallèle]  p=%2d | Tp = %.4f s | Accélération S = %.3f | Efficacité E = %.1f%% | Précision = %.2f%%\n",
               size, Tp, S, E, accuracy(all_preds, test_data, NTEST));

        free(scounts); free(displs); free(all_preds); free(local_preds);

    /* ================================================================
     * ESCLAVES (rangs 1..p-1)
     * ================================================================ */
    } else {
        MPI_Bcast(&Ts,     1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(&DIM,    1, MPI_INT,    0, MPI_COMM_WORLD);
        MPI_Bcast(&NTRAIN, 1, MPI_INT,    0, MPI_COMM_WORLD);
        MPI_Bcast(&NTEST,  1, MPI_INT,    0, MPI_COMM_WORLD);
        MPI_Bcast(train_data, NTRAIN * sizeof(Point), MPI_BYTE, 0, MPI_COMM_WORLD);
        MPI_Bcast(test_data,  NTEST  * sizeof(Point), MPI_BYTE, 0, MPI_COMM_WORLD);

        int base = NTEST / size, rem = NTEST % size;
        int n_local = (rank < rem) ? base+1 : base;
        int offset  = rank * base + (rank < rem ? rank : rem);

        int *scounts = (int*)malloc(size * sizeof(int));
        int *displs  = (int*)malloc(size * sizeof(int));
        for (int i = 0; i < size; i++) scounts[i] = (i < rem) ? base+1 : base;
        displs[0] = 0;
        for (int i = 1; i < size; i++) displs[i] = displs[i-1] + scounts[i-1];

        int *local_preds = (int*)malloc(n_local * sizeof(int));

        MPI_Barrier(MPI_COMM_WORLD);
        for (int i = 0; i < n_local; i++)
            local_preds[i] = knn_predict(train_data, NTRAIN, test_data[offset+i].feat);

        MPI_Gatherv(local_preds, n_local, MPI_INT,
                    NULL, scounts, displs, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);

        free(local_preds); free(scounts); free(displs);
    }

    MPI_Finalize();
    return 0;
}
