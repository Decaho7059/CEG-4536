
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <stdio.h>


__global__ void parallelAdd(int* input, int* output, int size) {

    // Tableau partagé entre les threads d'un bloc pour stocker les données temporairement
    extern __shared__ int sharedData[];
    int tid = threadIdx.x;
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    // Initialiser le tableau partagé avec les valeurs d'entrée
    if (index < size) {
       // Charger la valeur du tableau d'entrée dans la mémoire partagée 
        sharedData[tid] = input[index];
    }
    else {
        sharedData[tid] = 0; // Pour l'addition, initialiser à 0
    }
    __syncthreads();

    // Réduction parallèle (addition) pour diviser le tableau par 2 à chaque itération
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sharedData[tid] += sharedData[tid + s];
        }
        __syncthreads();
    }

    // Écrire le résultat final dans la mémoire globale
    if (tid == 0) output[blockIdx.x] = sharedData[0];
}

int main() {
    int size = 1 << 20; // Taille de 2^20
    int blockSize = 512;
    int gridSize = (size + blockSize - 1) / blockSize;

    int* h_input = (int*)malloc(size * sizeof(int));
    int* h_output = (int*)malloc(gridSize * sizeof(int));
    int* d_input, * d_output; // Pointeurs pour les tableaux d'entrée et de sortie sur le GPU

    // Initialisation du tableau avec des valeurs non nulles
    for (int i = 0; i < size; i++) {
        h_input[i] = 2; // Remplit chaque élément du tableau avec la valeur 2
    }

    // Initialisation du tableau avec des valeurs non nulles
    for (int i = 0; i < size; i++) h_input[i] = 1; // Initiliaser le tableau avec des valeurs de 2

    cudaMalloc((void**)&d_input, size * sizeof(int));
    cudaMalloc((void**)&d_output, gridSize * sizeof(int));

    cudaMemcpy(d_input, h_input, size * sizeof(int), cudaMemcpyHostToDevice);

    // Lancer le kernel d'addition parallèle
    parallelAdd << <gridSize, blockSize, blockSize * sizeof(int) >> > (d_input, d_output, size);

    cudaMemcpy(h_output, d_output, gridSize * sizeof(int), cudaMemcpyDeviceToHost);

    // Réduction finale sur le CPU
    int totalSum = 0;
    for (int i = 0; i < gridSize; i++) {
        totalSum += h_output[i];
    }

    printf("Somme totale des éléments du tableau: %d\n", totalSum);

    // Libération de la mémoire
    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);

    return 0;
}