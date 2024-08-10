#include <iostream>
#include <mpi.h>
#include <vector>
#include <time.h>
#include <chrono>
#include <stdint.h>

using namespace std;

// generates 2 polynomials
void generate(vector<int>& a, size_t n, vector<int>& b, size_t m)
{
    a.reserve(n);
    b.reserve(m);
    for(size_t i=0 ; i<n ; ++i) {
        a.push_back(1);
    }

    for(size_t i=0 ; i<m ; ++i) {
        b.push_back(1);
    }  
}


void multiplyPolynomials(const vector<int>& poly1, const vector<int>& poly2, vector<int>& result) {
    int n = poly1.size();
    int m = poly2.size();
    result.resize(n + m - 1, 0); 
    
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            result[i + j] += poly1[i] * poly2[j]; 
        }
    }
}


std::vector<int> MultiplyPolynomialsKaratsuba(const std::vector<int>& poly1, const std::vector<int>& poly2) {
    int n = std::max(poly1.size(), poly2.size());
    int middle = n / 2;
    if (poly1.size() < 2 || poly2.size() < 2) {
        int m = poly1.size();
        int k = poly2.size();
        std::vector<int> bc(m + k - 1, 0);
        multiplyPolynomials(poly1,poly2,bc);
        return bc;
    }

    std::vector<int> Poly1Part1(poly1.begin() + middle, poly1.end()); // high Part
    std::vector<int> Poly1Part2(poly1.begin(), poly1.begin() + middle); // low Part

    std::vector<int> Poly2Part1(poly2.begin() + middle, poly2.end());
    std::vector<int> Poly2Part2(poly2.begin(), poly2.begin() + middle);

    std::vector<int> Z0 = MultiplyPolynomialsKaratsuba(Poly1Part2, Poly2Part2);
    std::vector<int> Z2 = MultiplyPolynomialsKaratsuba(Poly1Part1, Poly2Part1);


    std::vector<int> e1(std::max(Poly1Part1.size(), Poly1Part2.size()), 0);
    for (size_t i = 0; i < std::max(Poly1Part1.size(), Poly1Part2.size()); i++) {
        int value1 = (i < Poly1Part1.size()) ? Poly1Part1[i] : 0;
        int value2 = (i < Poly1Part2.size()) ? Poly1Part2[i] : 0;
        e1[i] = value1 + value2;
    }

    std::vector<int> e2(std::max(Poly2Part1.size(), Poly2Part2.size()), 0);
    for (size_t j = 0; j < std::max(Poly2Part1.size(), Poly2Part2.size()); j++) {
        int value1 = (j < Poly2Part1.size()) ? Poly2Part1[j] : 0;
        int value2 = (j < Poly2Part2.size()) ? Poly2Part2[j] : 0;
        e2[j] = value1 + value2;
    }

    std::vector<int> Z1 = MultiplyPolynomialsKaratsuba(e1, e2);

    std::vector<int> partial_rez(std::max(Z1.size(), Z0.size()), 0);
    for (size_t i = 0; i < std::max(Z1.size(), Z0.size()); i++) {
        int value1 = (i < Z1.size()) ? Z1[i] : 0;
        int value2 = (i < Z0.size()) ? Z0[i] : 0;
        partial_rez[i] = value1 - value2;
    }

    std::vector<int> rez(std::max(partial_rez.size(), Z2.size()), 0);
    for (size_t i = 0; i < std::max(partial_rez.size(), Z2.size()); i++) {
        int value1 = (i < partial_rez.size()) ? partial_rez[i] : 0;
        int value2 = (i < Z2.size()) ? Z2[i] : 0;
        rez[i] = value1 - value2;
    }

    std::vector<int> result(2 * n - 1, 0);
    for (size_t i = 0; i < Z0.size(); i++) {
        result[i] += Z0[i];
    }
    for (size_t i = 0; i < rez.size(); i++) {
        result[i + middle] += rez[i];
    }
    for (size_t i = 0; i < Z2.size(); i++) {
        result[i + 2 * middle] += Z2[i];
    }

    return result;
}


inline bool checkrez(const vector <int> &a, const vector<int>& b, vector<int>& result) {
    vector<int> expectedResult(a.size() + b.size() - 1, 0);
    multiplyPolynomials(a, b, expectedResult);
    for (int i=0;i<min(expectedResult.size(),result.size());i++)
    {
        if (expectedResult[i]!=result[i])
        return false;
    }
    return true;
}


void process(vector<int> const& a, vector<int> const& b, int me, int nrProcs, vector<int>& result)
{
    int chunkSize = a.size() / nrProcs;
    MPI_Bcast(&chunkSize, 1, MPI_INT, 0, MPI_COMM_WORLD);

    vector<int> chunk(chunkSize, 0);

    MPI_Scatter(a.data(), chunkSize, MPI_INT,
                chunk.data(), chunk.size(), MPI_INT,
                0, MPI_COMM_WORLD);

    // Perform local polynomial multiplication

    vector<int> partial_rez(chunk.size()+b.size()-1,0);
    multiplyPolynomials(chunk, b, partial_rez);
    vector<int> all_parts;
    std::cout<<"Chunk for: "<<me<<" is: "<<chunkSize<<std::endl;
    if (me == 0) {
        all_parts.resize(nrProcs* partial_rez.size());
        //std::cout<<"Master all: "<<all_parts.size()<<std::endl;
    }

    MPI_Gather(partial_rez.data(), partial_rez.size(), MPI_INT, all_parts.data(), partial_rez.size(), MPI_INT, 0, MPI_COMM_WORLD);

    if(me == 0) {

        for (int i = 0; i < nrProcs; ++i) {
        int start_index = i * chunkSize;
        int end_index = min(start_index + partial_rez.size(), all_parts.size());

        // Process the slice in result vector and reconstruct the polynomial
        int slice_index = i * partial_rez.size();
        for (int j = start_index; j < end_index; ++j) {
            result[j] += all_parts[slice_index];
            ++slice_index;
            }
        }

    }
}


void karastuba_process(vector<int> const& a, vector<int> const& b, int me, int nrProcs, vector<int>& result)
{
    int chunkSize = a.size() / nrProcs;
    MPI_Bcast(&chunkSize, 1, MPI_INT, 0, MPI_COMM_WORLD);

    vector<int> chunk(chunkSize, 0);

    MPI_Scatter(a.data(), chunkSize, MPI_INT,
                chunk.data(), chunk.size(), MPI_INT,
                0, MPI_COMM_WORLD);

    // Perform local polynomial multiplication
    vector<int> partial_rez(chunk.size()+b.size()-1,0);
    partial_rez=MultiplyPolynomialsKaratsuba(chunk, b);
    std::cout<<"Chunk for: "<<me<<" is: "<<chunkSize<<std::endl;
    vector<int> all_parts;
    
    if (me == 0) {
        all_parts.resize(nrProcs* partial_rez.size());
        //std::cout<<"Master all: "<<all_parts.size()<<std::endl;
    }

    MPI_Gather(partial_rez.data(), partial_rez.size(), MPI_INT, all_parts.data(), partial_rez.size(), MPI_INT, 0, MPI_COMM_WORLD);

    if(me == 0) {

        for (int i = 0; i < nrProcs; ++i) {
        int start_index = i * chunkSize;
        int end_index = min(start_index + partial_rez.size(), all_parts.size());

        // Process the slice in result vector and reconstruct the polynomial
        int slice_index = i * partial_rez.size();
        for (int j = start_index; j < end_index; ++j) {
            result[j] += all_parts[slice_index];
            ++slice_index;
            }
        }

    }
}

void vectorSum(vector <int> &a, vector<int>& b, int me, int nrProcs, vector<int>& result) {
    a.resize(((a.size()+nrProcs-1)/nrProcs)*nrProcs, 0);
    process(a, b, me, nrProcs, result);
}

void karastuba_vectorSum(vector <int> &a, vector<int>& b, int me, int nrProcs, vector<int>& result) {
    a.resize(((a.size()+nrProcs-1)/nrProcs)*nrProcs, 0);
    std::cout<<"KARASTRUBA"<<std::endl;
    karastuba_process(a, b, me, nrProcs, result);
}


int main(int argc, char** argv)
{
    MPI_Init(0, 0);
    int me;
    int nrProcs;
    MPI_Comm_size(MPI_COMM_WORLD, &nrProcs);
    MPI_Comm_rank(MPI_COMM_WORLD, &me);

    unsigned n;
    vector<int> a;

    unsigned m;
    vector<int> b;

    vector <int> result;

   
    if (argc != 4 || 1!=sscanf(argv[1], "%u", &n) || 1!=sscanf(argv[2], "%u", &m)) {
        fprintf(stderr, "usage: sum-mpi <n> <m>\n");
        return 1;
    }

    char *str = argv[3];

    if (me==0)
    {
        result.resize(n + m - 1, 0);
        generate(a, n, b, m);
        fprintf(stderr, "generated\n");
    }
    else
    {
        b.resize(m);
    }
    
    MPI_Bcast(b.data(), b.size(), MPI_INT, 0, MPI_COMM_WORLD);
    if(me == 0) {

        chrono::high_resolution_clock::time_point const beginTime = chrono::high_resolution_clock::now();
        
        if (strcmp(str,"kara")==0)
        {
            karastuba_vectorSum(a, b, me, nrProcs, result);
        }
        else
        {
            vectorSum(a, b, me, nrProcs, result);
        }

        chrono::high_resolution_clock::time_point const endTime = chrono::high_resolution_clock::now();
        
        printf("Result %s, time=%ldms\n", (checkrez(a, b, result) ? "ok" : "FAIL"),
            (chrono::duration_cast<chrono::milliseconds>(endTime-beginTime)).count());

        for (const auto& element : result) {
            std::cout << element << " ";
        }
        std::cout << std::endl;
    } else {
        // worker
        
        chrono::high_resolution_clock::time_point const beginTime = chrono::high_resolution_clock::now();
        if (strcmp(str,"kara")==0)
        {
            karastuba_process(a, b, me, nrProcs, result);
        }
        else
        {
            process(vector<int>() , b , me, nrProcs, result);
        }
        chrono::high_resolution_clock::time_point const endTime = chrono::high_resolution_clock::now();
        
        printf("(worker %d): time=%ldms\n", me, 
            (chrono::duration_cast<chrono::milliseconds>(endTime-beginTime)).count());
    }
    MPI_Finalize();
}
