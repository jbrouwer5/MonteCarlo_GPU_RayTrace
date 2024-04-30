#include<stdio.h>
#include <iostream>
#include <fstream>
#include<stdlib.h>
#include<cuda.h>
#include<curand.h>
#include<curand_kernel.h>
#include <sstream>
#include <chrono>
#include <mutex>

// helper function to print the array to a file
void print_matrix(int size, double* matrix) {
    std::stringstream ss; // Use stringstream to buffer output
    for (int i = 0; i < size*size; i++) {
        ss << matrix[i];
        if (i < size*size - 1) 
            ss << ",";
    }
    ss << "\n";

    // Now write to file in one go
    std::ofstream myFile("output.txt", std::ios::app); // Open file in append mode
    myFile << ss.str();
    myFile.close(); 
}

__global__ void trace(double *, curandState *, curandState *, unsigned long long int*, int); /* device function */

__global__ void trace(double *G, curandState *phiStates, curandState *thetaStates, unsigned long long int *count, int N){
     
    int c[] = {0,12,0}; int l[] = {4,4,-1}; // x,y,z
    int radius = 6; double w[] = {0,2,0}; int wMax = 2; 
    double cellSize = wMax*2.0 / double(N);
    double v[3]; double phi; double cosTheta; double sinTheta; 
    double t; double i[3];  
    double b; int row; int col; 
    double inter; int cSquare = c[0]*c[0] + c[1]*c[1] + c[2]*c[2]; 
    double magLSubI; double magISubC; 
    double n[3]; double s[3]; 

    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    for (int intCount = 0; intCount<500; intCount++){

        do {
            // atomicAdd(count,1);
            phi = curand_uniform_double(&phiStates[tid]) * M_PI;
            cosTheta = curand_uniform_double(&thetaStates[tid]) * 2.0 - 1.0; 
            sinTheta = sqrt(1 - pow(cosTheta,2));

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
            
            atomicAdd(&G[row*N+col], b);
        }
    }
}

__global__ void initCurandStates(curandState *phiStates, curandState *thetaStates) {
    int seed1 = 0; 
    int seed2 = 1; 
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed1, tid, 0, &phiStates[tid]);
    curand_init(seed2, tid, 0, &thetaStates[tid]);
}

int main(int argc, char **argv){

    if (argc < 5)
    {
        std::cout << "Not enough arguments"; 
        return 0; 
    }
    
    // usage : raytrace nrays ngrid nblocks ntpb
    int numRays = std::stoi(argv[1]);
    int N = std::stoi(argv[2]); 
    int nBlocks = std::stoi(argv[3]);
    int ntpb = std::stoi(argv[4]); 

    auto start = std::chrono::steady_clock::now();

    int i;
    double *G = new double[N*N]; 
    double *dev_G; 
    unsigned long long int*count = new unsigned long long int; 
    *count = 0; 
    unsigned long long int* dev_count;

    curandState *d_phiStates, *d_thetaStates;

    for (i = 0; i < N*N; ++i){
        G[i] = 0; 
    }
    
    cudaMalloc( (void **) &dev_G, N*N*sizeof(double));
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess) {
        std::cout << "CUDA error1: " << cudaGetErrorString(error) << std::endl;
        exit(-1);
    }
    cudaMemcpy(dev_G, G, N*N*sizeof(double), cudaMemcpyHostToDevice);
    error = cudaGetLastError();
    if(error != cudaSuccess) {
        std::cout << "CUDA error2: " << cudaGetErrorString(error) << std::endl;
        exit(-1);
    }

    cudaMalloc( (void **) &dev_count, sizeof(unsigned long long int));
    error = cudaGetLastError();
    if(error != cudaSuccess) {
        std::cout << "CUDA error3: " << cudaGetErrorString(error) << std::endl;
        exit(-1);
    }
    cudaMemcpy(dev_count, count, sizeof(unsigned long long int), cudaMemcpyHostToDevice);
    error = cudaGetLastError();
    if(error != cudaSuccess) {
        std::cout << "CUDA error4: " << cudaGetErrorString(error) << std::endl;
        exit(-1);
    }
    
    auto kstart = std::chrono::steady_clock::now();
    
    // Allocate memory for curandStates on device
    cudaMalloc(&d_phiStates, nBlocks*ntpb * sizeof(curandState));
    cudaMalloc(&d_thetaStates, nBlocks*ntpb * sizeof(curandState));

    // Initialize curandStates
    initCurandStates<<<nBlocks, ntpb>>>(d_phiStates, d_thetaStates);
    
    cudaDeviceSynchronize();
    trace<<<nBlocks, ntpb>>>(dev_G, d_phiStates, d_thetaStates, dev_count, N);
    error = cudaGetLastError();
    if(error != cudaSuccess) {
        std::cout << "CUDA error5: " << cudaGetErrorString(error) << std::endl;
        exit(-1);
    }

    cudaDeviceSynchronize();
    std::cout << "Kernel took " << std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - kstart).count() << " seconds" << std::endl;
    
    cudaMemcpy(G, dev_G, N*N*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(count, dev_count, sizeof(long), cudaMemcpyDeviceToHost);
    std::cout << "total rays is " << *count << std::endl; 
    print_matrix(N, G);

    delete[] G;
    cudaFree(dev_G); 
    cudaFree(d_phiStates);
    cudaFree(d_thetaStates);
    delete count; 
    cudaFree(dev_count); 

    std::cout << "Took " << std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start).count() << " seconds" << std::endl;

    exit(0);
}
