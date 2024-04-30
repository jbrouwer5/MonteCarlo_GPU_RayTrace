#include <iostream>
#include <fstream>
#include <cmath>
#include <random>
#include <mutex>
#include <chrono>
#include <sstream>
using namespace std;

// helper function to print the array to a file
void print_matrix(int size, double* matrix) {
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

void rayTrace(int N, int numRays)
{   
    double G[N*N]; 
    for (int i=0; i<N*N; i++){
        G[i] = 0; 
    }

    // Distribution for phi
    std::random_device rd1; std::mt19937 gen1(rd1());
    std::uniform_real_distribution<> dis1(0, M_PI);
    // Distribution for cos(theta)
    std::uniform_real_distribution<> dis2(-1.0, 1.0);

    int c[] = {0,12,0}; int l[] = {4,4,-1}; // x,y,z
    int radius = 6; double w[] = {0,2,0}; int wMax = 2; 
    double cellSize = wMax*2.0 / double(N);
    double v[3]; double phi; double cosTheta; double sinTheta; 
    double t; double i[3];  
      double b; int row; int col; 
    double inter; int cSquare = c[0]*c[0] + c[1]*c[1] + c[2]*c[2]; 
    double magnituteLSubI; double magnituteISubC; 
    double n[3]; double s[3]; 
    long totRay = 0; 

    for (int count=0; count<numRays; ++count){
        do {
            ++totRay;  

            phi = dis1(gen1);
            cosTheta = dis2(gen1);
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
        magnituteISubC = sqrt((i[0]-c[0])*(i[0]-c[0]) + (i[1]-c[1])*(i[1]-c[1]) + (i[2]-c[2])*(i[2]-c[2])); 

        // // n = (i-c) / |i-c|
        n[0] = (i[0]-c[0]) / magnituteISubC; 
        n[1] = (i[1]-c[1]) / magnituteISubC; 
        n[2] = (i[2]-c[2]) / magnituteISubC; 

        // |l-i|
        magnituteLSubI = sqrt((l[0]-i[0])*(l[0]-i[0]) + (l[1]-i[1])*(l[1]-i[1]) + (l[2]-i[2])*(l[2]-i[2])); 

        // // s = (l-i) / |l-i|
        s[0] = (l[0]-i[0]) / magnituteLSubI;
        s[1] = (l[1]-i[1]) / magnituteLSubI;
        s[2] = (l[2]-i[2]) / magnituteLSubI;

        // b = max(0, s*n)
        b = (s[0]*n[0] + s[1]*n[1] + s[2]*n[2]);

        if (b > 0){
            // find (i, j) such that G(i, j) is the gridpoint of⃗ W on G 
            // use wX and wZ to calculate the point on the grid 
            row = floor((w[0]+wMax) / cellSize); 
            col = floor((w[2]+wMax) / cellSize); 

            G[row*N+col] += b;
        }
    }
    print_matrix(N, G);
    printf(" ray count is %ld\n", totRay); 
}

int 
main(int argc, char* argv[])
{
    if (argc < 3)
    {
        cout << "Not enough arguments"; 
        return 0; 
    }
    
    // usage : raytrace <numRays> <gridSize> 
    int numRays = stoi(argv[1]);
    int N = stoi(argv[2]); 

    auto start = chrono::steady_clock::now();
    
    rayTrace(N, numRays); 

    cout << "Took " << chrono::duration_cast<chrono::milliseconds>(
            chrono::steady_clock::now() - start).count() << " seconds" << endl;
}