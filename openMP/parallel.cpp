#include <iostream>
#include <fstream>
#include <cmath>
#include <random>
#include <mutex>
#include <omp.h>
#include <sstream>
#include <chrono>
using namespace std;

// helper function to print the array to a file
void print_matrix(int size, float* matrix) {
    stringstream ss; // Use stringstream to buffer output
    for (int i = 0; i < size*size; i++) {
        ss << matrix[i];
        if (i < size*size - 1) 
            ss << ",";
    }
    ss << "\n";

    // Now write to file in one go
    ofstream myFile("outputDP.txt", ios::app); // Open file in append mode
    myFile << ss.str();
    myFile.close(); 
}

void rayTrace(int N, int numRays, int nt)
{   
    float G[N*N]; 
    mutex locks[N]; 
    for (int i=0; i<N*N; i++){
        G[i] = 0; 
    }

    // Distribution for phi
    std::uniform_real_distribution<> dis1(0, M_PI);
    // Distribution for cos(theta)
    std::uniform_real_distribution<> dis2(-1.0, 1.0);

    std::mt19937 gens[nt];

    for(int i = 0; i < nt; ++i) {
        gens[i].seed(std::random_device{}());
    }

    int c[] = {0,12,0}; int l[] = {4,4,-1}; // x,y,z
    int radius = 6; float w[] = {0,2,0}; int wMax = 2; 
    float cellSize = wMax*2.0 / float(N);
    float v[3]; float phi; float cosTheta; float sinTheta; 
    float t; float i[3];  
    float b; int row; int col; 
    float inter; int cSquare = c[0]*c[0] + c[1]*c[1] + c[2]*c[2]; 
    float magLSubI; float magISubC; 
    float n[3]; float s[3]; 
    int rayCounts[nt]; 
    for (int i=0; i<nt; i++){
        rayCounts[i] = 0; 
    }

    #pragma omp parallel for default(none) shared(N,numRays,c,radius,l,wMax,G,cellSize,rayCounts,cSquare,locks, cout) \
    private(v,phi,cosTheta,sinTheta,t,i,magISubC,n,magLSubI,s,b,row,col,inter) \
    firstprivate(dis1,dis2,gens,w) schedule(static) 
    for (int count=0; count<numRays; count++){
        int threadId = omp_get_thread_num();
        do {
            // rayCounts[threadId]++; 
            phi = dis1(gens[threadId]);
            cosTheta = dis2(gens[threadId]);
            sinTheta = sqrt(1 - cosTheta*cosTheta);

            v[0] = sinTheta * cos(phi); 
            v[1] = sinTheta * sin(phi); 
            v[2] = cosTheta; 

            w[0] = (w[1] / v[1]) * v[0]; 
            w[2] = (w[1] / v[1]) * v[2]; 

            inter = v[0]*c[0] + v[1]*c[1] + v[2]*c[2];
            t = inter*inter + radius*radius - cSquare;
        }
        while (!(abs(w[0]) < wMax && abs(w[2]) < wMax && t > 0));
        
        t = (v[0]*c[0] + v[1]*c[1] + v[2]*c[2]) - sqrt(t); 
        
        // i = t*v
        i[0] = t * v[0]; i[1] = t * v[1]; i[2] = t * v[2]; 

        // |i-c|
        magISubC = sqrt((i[0]-c[0])*(i[0]-c[0]) + (i[1]-c[1])*(i[1]-c[1]) + (i[2]-c[2])*(i[2]-c[2])); 

        // // n = (i-c) / |i-c|
        n[0] = (i[0]-c[0]) / magISubC; 
        n[1] = (i[1]-c[1]) / magISubC; 
        n[2] = (i[2]-c[2]) / magISubC; 

        // |l-i|
        magLSubI = sqrt((l[0]-i[0])*(l[0]-i[0]) + (l[1]-i[1])*(l[1]-i[1]) + (l[2]-i[2])*(l[2]-i[2])); 

        // // s = (l-i) / |l-i|
        s[0] = (l[0]-i[0]) / magLSubI;
        s[1] = (l[1]-i[1]) / magLSubI;
        s[2] = (l[2]-i[2]) / magLSubI;

        // b = max(0, s*n)
        b = (s[0]*n[0] + s[1]*n[1] + s[2]*n[2]);

        if (b > 0){
            // find (i, j) such that G(i, j) is the gridpoint of⃗ W on G 
            // use wX and wZ to calculate the point on the grid 
            row = floor((w[0]+wMax) / cellSize); 
            col = floor((w[2]+wMax) / cellSize); 

            locks[row].lock(); 
            G[row*N+col] += b; 
            locks[row].unlock(); 
        }
    }
    
    print_matrix(N, G);
    long totRays = 0;
    for (int k=0;k<nt;k++){
        totRays += rayCounts[k];
    }
    printf(" ray count is %ld\n", totRays); 
}

int 
main(int argc, char* argv[])
{
    if (argc < 4)
    {
        cout << "Not enough arguments"; 
        return 0; 
    }
    
    // usage : raytrace <numRays> <gridSize> <numThreads>
    int numRays = stoi(argv[1]);
    int N = stoi(argv[2]); 
    int nt = stoi(argv[3]);
    omp_set_num_threads(nt); 

    auto start = chrono::steady_clock::now();
    
    rayTrace(N, numRays, nt); 

    cout << "Took " << chrono::duration_cast<chrono::milliseconds>(
            chrono::steady_clock::now() - start).count() << " seconds" << endl;
}