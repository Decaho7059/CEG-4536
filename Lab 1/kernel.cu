#include "device_launch_parameters.h"
#include <algorithm>
#include <stdio.h>
#include <time.h>
#include <vector>
#include <random> 
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <fstream>
#include <iostream>
#include <chrono>
#include <stdlib.h>
#include <GL/glut.h>
#include <curand_kernel.h>

using namespace std;

#define N 1000  // Taille de la grille
#define BURN_DURATION 5000  // Durée de combustion d'un arbre en millisecondes (5 secondes)
#define FIRE_START_COUNT 100  // Nombre initial d'incendies

// Utilisation de vecteurs pour gérer la mémoire
std::vector<std::vector<int>> forest(N, std::vector<int>(N, 0));
std::vector<std::vector<int>> burnTime(N, std::vector<int>(N, 0));

int simulationDuration = 60000;  // Durée de la simulation (60 secondes)
int startTime = 0;  // Temps de départ en millisecondes
int elapsedTime = 0;  // Temps écoulé
float spreadProbability = 0.3f;  // Probabilité que le feu se propage à un arbre voisin

bool isPaused = false;  // Indicateur de pause
int pauseStartTime = 0;  // Temps de début de la pause

float zoomLevel = 1.0f;  // Niveau de zoom
float offsetX = 0.0f, offsetY = 0.0f;  // Décalage horizontal et vertical pour le déplacement
float moveSpeed = 0.05f;  // Vitesse de déplacement de la vue

bool dragging = false;  // Indicateur de glisser-déposer avec la souris
int lastMouseX, lastMouseY;  // Dernière position de la souris lors du clic

// Fonction pour initialiser la forêt et démarrer les incendies
void initializeForest() {
    // Génère aléatoirement des arbres (1) ou des cellules vides (0) pour chaque case de la forêt
    for (auto &row : forest) {
        std::generate(row.begin(), row.end(), []() { return rand() % 2; });
    }

    // Initialise le temps de combustion à 0 pour toutes les cellules
    std::fill(burnTime.begin(), burnTime.end(), std::vector<int>(N, 0));

    // Récupère toutes les positions des arbres disponibles (valeur 1)
    std::vector<std::pair<int, int>> availablePositions;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (forest[i][j] == 1) availablePositions.emplace_back(i, j);
        }
    }

    // Mélange les positions disponibles pour sélectionner des arbres aléatoires pour démarrer le feu
    std::shuffle(availablePositions.begin(), availablePositions.end(), std::mt19937(std::random_device{}()));
    
    // Sélectionne un certain nombre d'arbres pour allumer le feu
    for (int fire = 0; fire < FIRE_START_COUNT && fire < availablePositions.size(); fire++) {
        int fireX = availablePositions[fire].first;
        int fireY = availablePositions[fire].second;
        forest[fireX][fireY] = 2; // 2 représente un arbre en feu
        burnTime[fireX][fireY] = BURN_DURATION; // Temps de combustion initial
    }

    // Initialise le temps du début de la simulation
    startTime = glutGet(GLUT_ELAPSED_TIME);
    elapsedTime = 0;
    isPaused = false;
}


// Fonction pour initialiser les paramètres OpenGL
void initGL() {
    glClearColor(1.0, 1.0, 1.0, 1.0);  // Définit le fond blanc
    glEnable(GL_DEPTH_TEST);  // Active le test de profondeur pour un rendu 3D correct
}

// Fonction pour dessiner la forêt
void drawForest() {
    float cellSize = 2.0f / N;  // Taille de chaque cellule dans la fenêtre OpenGL

    // Fonction lambda pour définir la couleur selon l'état de la cellule
    auto setColor = [](int state) {
        switch (state) {
            case 0: glColor3f(0.8f, 0.8f, 0.8f); break; // Cellule vide (gris)
            case 1: glColor3f(0.0f, 1.0f, 0.0f); break; // Arbre (vert)
            case 2: glColor3f(1.0f, 0.0f, 0.0f); break; // Feu (rouge)
            case 3: glColor3f(0.0f, 0.0f, 0.0f); break; // Cendres (noir)
        }
    };

    // Parcourt chaque cellule pour dessiner la forêt
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            setColor(forest[i][j]);  // Détermine la couleur en fonction de l'état
            float x = -1.0f + j * cellSize, y = -1.0f + i * cellSize;
            glBegin(GL_QUADS); // Dessine chaque cellule comme un carré
            glVertex2f(x, y); glVertex2f(x + cellSize, y); 
            glVertex2f(x + cellSize, y + cellSize); glVertex2f(x, y + cellSize);
            glEnd();
        }
    }
}

// Kernel CUDA pour mettre à jour l'état de la forêt
__global__ void updateForestKernel(int* forest, int* burnTime, float spreadProbability, int A, int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x + (blockIdx.y * blockDim.y + threadIdx.y) * A;
    curandState state; curand_init(seed, idx, 0, &state);
    
    // Si l'arbre est en feu et que le temps de combustion est écoulé, il devient des cendres
    if (forest[idx] == 2 && --burnTime[idx] <= 0) forest[idx] = 3;
    // Sinon, on propage le feu aux arbres voisins avec une probabilité donnée
    else if (forest[idx] == 2) {
        auto spreadFire = [&](int i, int j) {
            int nIdx = i * A + j;
            if (forest[nIdx] == 1 && curand_uniform(&state) < spreadProbability) {
                forest[nIdx] = 2; burnTime[nIdx] = BURN_DURATION;
            }
        };
        int i = idx / A, j = idx % A;
        if (i > 0) spreadFire(i - 1, j);     // Propage le feu vers le haut
        if (i < A - 1) spreadFire(i + 1, j); // Propage le feu vers le bas
        if (j > 0) spreadFire(i, j - 1);     // Propage le feu à gauche
        if (j < A - 1) spreadFire(i, j + 1); // Propage le feu à droite
    }
}

// Fonction principale pour lancer la mise à jour de la forêt avec CUDA
void updateForestCUDA() {
    int* d_forest, * d_burnTime;
    
    // Allocation de mémoire sur le GPU pour la forêt et le temps de combustion
    cudaMalloc((void**)&d_forest, N * N * sizeof(int));
    cudaMalloc((void**)&d_burnTime, N * N * sizeof(int));

    // Copie les données de la forêt et du temps de combustion depuis l'hôte (CPU) vers le GPU
    cudaMemcpy(d_forest, forest[0].data(), N * N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_burnTime, burnTime[0].data(), N * N * sizeof(int), cudaMemcpyHostToDevice);

    // Définition de la grille et des blocs pour le lancement du kernel CUDA
    dim3 threadsPerBlock(16, 16), blocksPerGrid((N + 15) / 16, (N + 15) / 16);
    updateForestKernel <<<blocksPerGrid, threadsPerBlock>>>(d_forest, d_burnTime, spreadProbability, N, time(NULL));

    // Copie des données du GPU vers l'hôte après l'exécution du kernel
    cudaMemcpy(forest[0].data(), d_forest, N * N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(burnTime[0].data(), d_burnTime, N * N * sizeof(int), cudaMemcpyDeviceToHost);

    // Libération de la mémoire sur le GPU
    cudaFree(d_forest);
    cudaFree(d_burnTime);
}

// Fonction d'affichage principale
void display() {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // Efface la fenêtre
    glPushMatrix(); // Sauvegarde la matrice actuelle
    glScalef(zoomLevel, zoomLevel, 1.0f); // Applique le zoom
    glTranslatef(offsetX, offsetY, 0.0f); // Applique le déplacement
    drawForest(); // Dessine la forêt
    glPopMatrix(); // Restaure la matrice
    glutSwapBuffers(); // Échange les buffers pour l'affichage
}

// Fonction de mise à jour en continu (idle)
void idle() {
    if (!isPaused && elapsedTime < simulationDuration) {
        elapsedTime = glutGet(GLUT_ELAPSED_TIME) - startTime; // Met à jour le temps écoulé
        updateForestCUDA(); // Met à jour l'état de la forêt avec CUDA
        glutPostRedisplay(); // Redessine la fenêtre après la mise à jour
    }
}

// Fonction de gestion des entrées clavier
void keyboard(unsigned char key, int, int) {
    switch (key) {
        case 'r': initializeForest(); break; // Réinitialise la simulation
        case 'p': isPaused = !isPaused;      // Met en pause ou relance la simulation
                  if (isPaused) pauseStartTime = glutGet(GLUT_ELAPSED_TIME); 
                  else startTime += glutGet(GLUT_ELAPSED_TIME) - pauseStartTime;
                  break;
        case '+': zoomLevel *= 1.1f; break;  // Zoom avant
        case '-': zoomLevel *= 0.9f; break;  // Zoom arrière
        case 'w': offsetY -= moveSpeed; break; // Déplacement vers le haut
        case 's': offsetY += moveSpeed; break; // Déplacement vers le bas
        case 'a': offsetX += moveSpeed; break; // Déplacement vers la gauche
        case 'd': offsetX -= moveSpeed; break; // Déplacement vers la droite
    }
    glutPostRedisplay(); // Redessine la fenêtre après une action
}

// Fonction de gestion des clics de souris
void mouse(int button, int state, int x, int y) {
    if (button == GLUT_LEFT_BUTTON) dragging = (state == GLUT_DOWN); // Active le mode "drag" avec le bouton gauche
    if (dragging) lastMouseX = x, lastMouseY = y; // Enregistre la position initiale de la souris
}

// Fonction de gestion des mouvements de souris
void motion(int x, int y) {
    if (!dragging) return;
    offsetX += (x - lastMouseX) * 0.01f; // Déplace la vue en fonction du mouvement horizontal
    offsetY -= (y - lastMouseY) * 0.01f; // Déplace la vue en fonction du mouvement vertical
    lastMouseX = x, lastMouseY = y;      // Met à jour la dernière position de la souris
    glutPostRedisplay();                 // Redessine la fenêtre après le mouvement
}

// Fonction principale du programme
int main(int argc, char** argv) {
    // Initialisation de GLUT (bibliothèque pour la gestion de fenêtres et OpenGL)
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH); // Mode d'affichage en double buffer, RGB et test de profondeur
    glutInitWindowSize(800, 800); // Définit la taille initiale de la fenêtre
    glutCreateWindow("Simulation d'Incendie Forestier"); // Crée la fenêtre avec un titre

    
    initGL();// Initialisation des paramètres OpenGL
    initializeForest();// Initialise la forêt et démarre les feux

    glutDisplayFunc(display);// Fonction d'affichage appelée à chaque redessin de la fenêtre

    glutIdleFunc(idle);// Fonction appelée lorsque l'ordinateur est en mode inactif (utilisée pour la mise à jour continue)
    glutKeyboardFunc(keyboard);// Fonction appelée lors des événements clavier (comme les touches 'r', 'p', etc.)
    glutMouseFunc(mouse);    // Fonction appelée lors des événements de la souris (comme les clics)
    glutMotionFunc(motion);    // Fonction appelée lors du mouvement de la souris lorsque l'on fait glisser
    glutMainLoop();    // Lance la boucle principale de GLUT (infinie) qui gère les événements et redessine la fenêtre

    return 0; // Retourne 0 lorsque le programme se termine (ce qui n'arrive pas avec glutMainLoop)
}

