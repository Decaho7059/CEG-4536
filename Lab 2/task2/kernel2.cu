#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>



// Template pour l'opération d'addition
template <typename T>
struct Add {
    __device__ T operator()(T a, T b) const {
        return a + b;
    }
};

// Kernel avec unrolling des boucles 
template <typename T, typename Op> // type de données (T) et  Opération (Op)

__global__ void parallelReduceUnrolled(T* input, T* output, int size, Op op, T neutralElement) {
    extern __shared__ T sharedData[]; // Mémoire partagée pour les threads du bloc
    int tid = threadIdx.x;
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    // Charger les données dans la mémoire partagée
    if (index < size) sharedData[tid] = input[index];
    else sharedData[tid] = neutralElement; // Remplir avec l'élément neutre si l'index dépasse la taille

    __syncthreads();

    // Réduction avec un unrolling de boucle par 4 pour optimiser l’utilisation de la mémoire
 
    for (int s = blockDim.x / 4; s > 32; s /= 4) {  // Réduction de 4 éléments a chaque étape
        if (tid < s) {
            // Combiner les valeurs dans la mémoire partagée
            sharedData[tid] = op(sharedData[tid], sharedData[tid + s]);
            sharedData[tid] = op(sharedData[tid], sharedData[tid + s * 2]);
            sharedData[tid] = op(sharedData[tid], sharedData[tid + s * 3]);
        }
        __syncthreads(); // Synchroniser les threads à chaque étape de réduction
    }

    // Réduction finale par warp 
    if (tid < 32) {
        volatile int* vsmem = sharedData;
        vsmem[tid] += vsmem[tid + 32];
        vsmem[tid] += vsmem[tid + 16];
        vsmem[tid] += vsmem[tid + 8];
        vsmem[tid] += vsmem[tid + 4];
        vsmem[tid] += vsmem[tid + 2];
        vsmem[tid] += vsmem[tid + 1];
    }

    // Écrire le résultat du bloc dans la mémoire globale
    if (tid == 0) output[blockIdx.x] = sharedData[0]; // Seul le thread 0 écrit le résultat du bloc
}

int main() {
    int size = 128;
    int blockSize = 32;
    int gridSize = (size + blockSize - 1) / blockSize;

    int* h_input = (int*)malloc(size * sizeof(int));
    int* h_output = (int*)malloc(gridSize * sizeof(int));
    int* d_input, * d_output;
    cudaMalloc((void**)&d_input, size * sizeof(int));
    cudaMalloc((void**)&d_output, gridSize * sizeof(int));

    // Initialiser le tableau d'entrée
    for (int i = 0; i < size; i++) h_input[i] = i + 1;

    cudaMemcpy(d_input, h_input, size * sizeof(int), cudaMemcpyHostToDevice);

    // Chronométrage du kernel
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Enregistrer le début du kernel
    cudaEventRecord(start);
    // Lancer le kernel pour l'addition
    parallelReduceUnrolled<int, Add<int>> << <gridSize, blockSize, blockSize * sizeof(int) >> > (d_input, d_output, size, Add<int>(), 0);
    // Enregistrer la fin du kernel
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Temps d'exécution: %f ms\n", milliseconds); // Afficher le temps d'exécution
    cudaMemcpy(h_output, d_output, gridSize * sizeof(int), cudaMemcpyDeviceToHost);

    int sum = 0;
    for (int i = 0; i < gridSize; i++) sum += h_output[i]; // Additionner les résultats des blocs
    printf("Sum: %d\n", sum); // Afficher la somme totale

    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);

    return 0;
}
