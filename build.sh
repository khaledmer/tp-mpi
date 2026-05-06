#!/bin/bash
# build.sh — Render build script
# Installs OpenMPI and compiles the MPI C programs

set -e  # stop on any error

echo "======================================"
echo " Build TP MPI — Merbouche Abdelatif"
echo " USTHB M1 HPC 2025/2026"
echo "======================================"

# Install OpenMPI
echo "[1/3] Installing OpenMPI..."
apt-get update -qq
apt-get install -y -qq libopenmpi-dev openmpi-bin

# Verify mpicc is available
echo "[2/3] Verifying mpicc..."
which mpicc
mpicc --version

# Compile C programs
echo "[3/3] Compiling MPI programs..."
mpicc -O2 -o knn_mpi    knn_mpi.c    -lm
mpicc -O2 -o kmeans_mpi kmeans_mpi.c -lm

echo ""
echo "Build complete."
ls -lh knn_mpi kmeans_mpi
