#pragma once


// This header file includes required libraries and header files for the code.

// INCLUDES /////////////////////////////////////////////////////////////////////////////////////////////////////

// CUDA: Include cuda_runtime and device_launch_parameters libraries for CUDA support.
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// BLAS: Include cublas_v2 and cblas libraries for basic linear algebra support.
#include "cublas_v2.h"
//#include <openblas/cblas.h>

// STD
#include <iostream> // Printing
#include <fstream> // File saving
#include <vector> // Dynamic arrays
#include <map> // Dictionaries
#include <string> // String manipulation
#include <assert.h> // Error handling
#include <memory> // Memory management

// INTERNAL
#include "utilities.cuh"

// DEFINES /////////////////////////////////////////////////////////////////////////////////////////////////////

// Define two memory access types for two-dimensional matrix.
// Colone major access
#define IDXH(i, j, h) (j*h+i)
// Line major access
#define IDXL(i, j, l) (i*l+j)





// PREDECLARATION /////////////////////////////////////////////////////////////////////////////////////////////////////

namespace Cpu {
    template<typename T> class Matrix;
    template<typename T> class MatrixSave;
}

namespace Gpu {
    template<typename T> class Matrix;
    template<typename T> class MatrixSave;
}



// CPU /////////////////////////////////////////////////////////////////////////////////////////////////////

namespace Cpu {

    // Matrix class
    // The Matrix class template in C++ defines a matrix with elements of type T.
    template<typename T>
    class Matrix {

    private:
        // Height and width of the matrix
        int height_, width_;
        // Memory space for the matrix element values
        std::shared_ptr<T[]> data_;

    public:

        // Constructors and destructor
        
        // Constructor usinge existing data, height and width
        explicit Matrix(std::shared_ptr<T[]> data, int height, int width)
            : data_(data), height_(height), width_(width)
        {}
        // Constructor of a new height*width Matrix
        Matrix(int height, int width)
            : data_(new T[height * width]), height_(height), width_(width)
        {}
        // Copy constructor
        Matrix(const Matrix<T>& A)
            : data_(A.data()), height_(A.height()), width_(A.width())
        {}
        Matrix(const Gpu::Matrix<T>& A)
            : data_(new T[A.size()]), height_(A.height()), width_(A.width())
        {
            cudaMemcpy((void*)data().get(), (void*)A.data().get(), size() * sizeof(T), cudaMemcpyDeviceToHost);
        }
        // Destructor
        ~Matrix() {}


        // Getters
        
        std::shared_ptr<T[]> data() const { return data_; }

        int height() const { return height_; }

        int width() const { return width_; }

        int size() const { return height() * width(); }

        // Returns the value of the element ('i', 'j') of the matrix
        T get(int i, int j) const {
            assert(i < height() && j < width());
            return data()[height() * j + i];
        }

        // Tests if the Matrix A has the same height and width as A
        bool same_dimension(const Matrix<T>& A) {
            return A.height() == height() && A.width() == width();
        }

        // Setters

        // Set the ('i', 'j') element of the matrix to the value of 'x'
        T set(int i, int j, T x) {
            data()[height() * j + i] = x;
            return x;
        }

        // Copy elements of 'A' to the elements of the matrix
        void copy(const Matrix<T>& A) {
            assert(same_dimension(A));
            std::memcpy(data().get(), A.data().get(), size() * sizeof(T));
        }

        // Copy the SubMatrix of 'A' starting at ('i0','j0') and
        // of size 'height()' * 'width()' to the matrix
        void extract(const Matrix<T>& A, int i0, int j0) {
            assert(i0 + height() <= A.height());
            assert(j0 + width() <= A.width());
            for (int j = 0; j < width(); j++)
            {
                std::memcpy((void*)&data().get()[IDXH(0, j, height())], (void*)&A.data().get()[IDXH(i0, j0 + j, A.height())], height() * sizeof(T));
            }
        }
        void extract_to(Matrix<T>& A, int i0, int j0) {
            assert(i0 + height() <= A.height());
            assert(j0 + width() <= A.width());
            for (int j = 0; j < width(); j++)
            {
                std::memcpy((void*)&A.data().get()[IDXH(i0, j0 + j, A.height())], (void*)&data().get()[IDXH(0, j, height())], height() * sizeof(T));
            }
        }

        // Fill the matrix with the value in 'x'
        void fill(T x) {
            for (int i = 0; i < size(); i++)
            {
                data()[i] = x;
            }
        }

        // Fill the matrix with random values between 'min' and 'max'
        void randfill(T min, T max) {
            for (int i = 0; i < size(); i++)
            {
                data()[i] = Random::random<T>(min, max);
            }
        }

        // Computing 

        // Scalar function performs element-wise scaling and addition of the elements of matrix A with given alpha and beta coefficients
        void Scalar(T alpha, const Matrix<T>& A, T beta) {
            assert(same_dimension(A));
            for (int i = 0; i < size(); i++)
            {
                data()[i] = alpha * A.data()[i] + beta;
            }
        }

        // Linear function performs element-wise linear combination of the elements of matrix A and B with given alpha and beta coefficients.
        void Linear(T alpha, const Matrix<T>& A, T beta, const Matrix<T>& B) {
            assert(same_dimension(A));
            assert(same_dimension(B));
            for (int i = 0; i < size(); i++)
            {
                data()[i] = alpha * A.data()[i] + beta * B.data()[i];
            }
        }

        // Matmul function performs element-wise multiplication of the elements of matrices A and B.
        void Matmul(const Matrix<T>& A, const Matrix<T>& B) {
            assert(same_dimension(A));
            assert(same_dimension(B));
            for (int i = 0; i < size(); i++)
            {
                data()[i] = A.data()[i] * B.data()[i];
            }
        }

        // Matprod function performs matrix product of matrices A and B, with possible transpositions specified by transa and transb.
        void Matprod(const Matrix<T>& A, char transa, const Matrix<T>& B, char transb);

        // Power function performs element-wise power operation of the elements of matrix A with given power p.
        void Power(const Matrix<T>& A, T p) {
            assert(same_dimension(A));
            for (int i = 0; i < size(); i++)
            {
                data()[i] = (T)pow((double)A.data()[i], (double)p);
            }
        }

        // Reverse a square matrix
        void Reverse(const Matrix<T>& A) {
            int n = height();
            Gpu::Matrix<T> RES(height(), width());
            Gpu::Matrix<T> TMP(A.height(), A.width());
            RES.copy(*this);
            TMP.copy(A);
            RES.Reverse(TMP);
            RES.copy_to(*this);
        }

    };

    
    /*template<> void Matrix<double>::Matprod(const Matrix<double>& A, char transa, const Matrix<double>& B, char transb) {

        if (transa != 't' && transb != 't') {
            for (int i = 0; i < height(); i++)
            {
                for (int j = 0; j < width(); j++)
                {
                    double tmp = 0;
                    for (int k = 0; k < A.width(); k++)
                    {
                        tmp += A.get(i, k) * B.get(k, j);
                    }
                    set(i, j, tmp);
                }
            }
        }
        else if (transa == 't' && transb == 't') {
            for (int i = 0; i < height(); i++)
            {
                for (int j = 0; j < width(); j++)
                {
                    double tmp = 0;
                    for (int k = 0; k < A.height(); k++)
                    {
                        tmp += A.get(k, i) * B.get(j, k);
                    }
                    set(i, j, tmp);
                }
            }
        }
        else if (transa == 't') {
            for (int i = 0; i < height(); i++)
            {
                for (int j = 0; j < width(); j++)
                {
                    double tmp = 0;
                    for (int k = 0; k < A.height(); k++)
                    {
                        tmp += A.get(k, i) * B.get(k, j);
                    }
                    set(i, j, tmp);
                }
            }
        }
        else {
            for (int i = 0; i < height(); i++)
            {
                for (int j = 0; j < width(); j++)
                {
                    double tmp = 0;
                    for (int k = 0; k < A.width(); k++)
                    {
                        tmp += A.get(i, k) * B.get(j, k);
                    }
                    set(i, j, tmp);
                }
            }
        }
        
    }
    template<> void Matrix<double>::Matprod(const Matrix<double>& A, char transa, const Matrix<double>& B, char transb) {

        CBLAS_TRANSPOSE op_a = transa == 't' || transa == 'T' ? CblasTrans : CblasNoTrans;
        CBLAS_TRANSPOSE op_b = transb == 't' || transb == 'T' ? CblasTrans : CblasNoTrans;

        // On vérifie que les dimensions sont correctes
        int k = 0;
        if (op_a == CblasTrans) {
            k = A.height();
            assert(A.width() == height());
        }
        else {
            k = A.width();
            assert(A.height() == height());
        }

        if (op_b == CblasTrans) {
            assert(B.height() == width());
            assert(B.width() == k);
        }
        else {
            assert(B.width() == width());
            assert(B.height() == k);
        }

        int h1 = A.height();
        int h2 = B.height();



        cblas_dgemm(CblasColMajor, op_a, op_b, height(), width(), k, 1, A.data().get(), h1, B.data().get(), h2, 0, data().get(), height());

    }
    template<> void Matrix<float>::Matprod(const Matrix<float>& A, char transa, const Matrix<float>& B, char transb) {

        CBLAS_TRANSPOSE op_a = transa == 't' || transa == 'T' ? CblasTrans : CblasNoTrans;
        CBLAS_TRANSPOSE op_b = transb == 't' || transb == 'T' ? CblasTrans : CblasNoTrans;

        // On vérifie que les dimensions sont correctes
        int k = 0;
        if (op_a == CblasTrans) {
            k = A.height();
            assert(A.width() == height());
        }
        else {
            k = A.width();
            assert(A.height() == height());
        }

        if (op_b == CblasTrans) {
            assert(B.height() == width());
            assert(B.width() == k);
        }
        else {
            assert(B.width() == width());
            assert(B.height() == k);
        }

        int h1 = A.height();
        int h2 = B.height();

        cblas_sgemm(CblasColMajor, op_a, op_b, height(), width(), k, 1, A.data().get(), h1, B.data().get(), h2, 0, data().get(), height());

    }*/

    // Print function of the class Cpu::Matrix when calling std::cout << A
    template<typename T> std::ostream& operator<<(std::ostream& os, const Matrix<T>& A)
    {
        for (int i = 0; i < A.height(); i++) {
            for (int j = 0; j < A.width(); j++)
            {
                os << A.get(i, j) << " ";
            }
            os << std::endl;
        }
        return os;
    }




    // SAVE /////////////////////////////////////////////////////////////////////////////////////////////////////

    template<typename T> class MatrixSave {

    private:

        std::map<std::string, Cpu::Matrix<T>> map_matrix_;

    public:

        MatrixSave()
        {}
        MatrixSave(std::map<std::string, Cpu::Matrix<T>> map_matrix)
            : map_matrix_(map_matrix)
        {}

        void add_matrix(Matrix<T> t, std::string name) {
            auto it = map_matrix_.find(name);
            if (it != map_matrix_.end()) {
                std::cout << "Error : Name already ussed !\n";
                assert(0);
            }
            if (name.find("\n") != std::string::npos) {
                std::cout << "Error : Invalid name !\n";
                assert(0);
            }
            map_matrix_[name] = t;
        }

        void save(std::string file) {
            std::ofstream fichier(file, std::ios::out | std::ios::binary);

            for (auto it = map_matrix_.begin(); it != map_matrix_.end(); it++) {

                fichier.write(it->first.c_str(), it->first.size() * sizeof(char));
                fichier.write("\n", sizeof(char));

                T* buffer = it->second.data().get();
                fichier.write(reinterpret_cast<char*>(buffer), it->second.size() * sizeof(T));
            }

            fichier.close();
        }

        void load(std::string file) {
            std::ifstream fichier(file, std::ios::in | std::ios::binary);

            for (std::string name; std::getline(fichier, name);) {

                int taille = map_matrix_[name].size();

                T* buffer = map_matrix_[name].data().get();
                fichier.read(reinterpret_cast<char*>(buffer), taille * sizeof(T));

            }

            fichier.close();

        }

        void print() {
            std::cout << "\n############\n Print begin \n############\n\n";
            for (auto it = map_matrix_.begin(); it != map_matrix_.end(); it++) {
                std::cout << "## " << it->first << "\n";
                std::cout << it->second << std::endl;
                std::cout << "##\n\n";
            }
            std::cout << "############\n Print end \n############\n";
        }

    };


    template<typename T> struct MatrixPair {

        Matrix<T> X;
        Matrix<T> Y;

    };

}
















// GPU /////////////////////////////////////////////////////////////////////////////////////////////////////


namespace Gpu {

    // A global cublas environnement (ie handle) to provide fast matrix product perfomed on the GPU
    cublasHandle_t ENV_CUBLAS__;
    // A counter to ensure creation and destruction of the cublas enrironnements
    size_t COUNT_ENV_CUBLAS__ = 0;


    template<typename T>
    class Matrix {

    private:

        int height_, width_;
        Gpu::SafePointer<T> data_;

        // Privates functions to manage the global cublas environnement creation and destruction
        void add_env_cublas() {
            if (COUNT_ENV_CUBLAS__ == 0) {
                cublasCreate(&ENV_CUBLAS__);
            }
            COUNT_ENV_CUBLAS__++;
        }
        void suppr_env_cublas() {
            COUNT_ENV_CUBLAS__--;
            if (COUNT_ENV_CUBLAS__ == 0) {
                cublasDestroy(ENV_CUBLAS__);
            }
        }

    public:

        // Constructors and destructor

        // Constructor using existing data, height and width
        Matrix() : data_(nullptr), height_(0), width_(0) {}
        // Constructor of a new height*width Matrix
        Matrix(int height, int width)
            : data_(height*width), height_(height), width_(width)
        {
            add_env_cublas();
        }
        // Copy constructor
        Matrix(const Matrix<T>& A)
            : data_(A.data()), height_(A.height()), width_(A.width())
        {
            add_env_cublas();
        }
        Matrix(const Cpu::Matrix<T>& A)
            : data_(A.size()), height_(A.height()), width_(A.width())
        {
            add_env_cublas();
            cudaMemcpy((void*)data().get(), (void*)A.data().get(), size() * sizeof(T), cudaMemcpyHostToDevice);
        }
        // Destructor
        ~Matrix() {
            suppr_env_cublas();
        }


        // Getters

        Gpu::SafePointer<T> data() const { return data_; }

        int height() const { return height_; }

        int width() const { return width_; }

        int size() const { return height() * width(); }

        // Returns the value of the element ('i', 'j') of the matrix
        T get(int i, int j) const {
            assert(i < height() && j < width());
            T tmp;
            cudaMemcpy((void*)&tmp, (void*)&(data()[height() * j + i]), sizeof(T), cudaMemcpyDeviceToHost);
            Gpu::__gpu_test__();
            return tmp;
        }

        // Tests de la classe

        // Tests if the matrix has the same height and width as 'A'
        bool same_dimension(const Matrix<T>& A) const {
            return A.height() == height() && A.width() == width();
        }
        // Cpu::Matrix version
        bool same_dimension(const Cpu::Matrix<T>& A) const {
            return A.height() == height() && A.width() == width();
        }


        // Setters

        // Set the ('i', 'j') element of the matrix to the value of 'x'
        T set(int i, int j, T x) {
            cudaMemcpy((void*)&(data()[height() * j + i]), &x, sizeof(T), cudaMemcpyHostToDevice);
            Gpu::__gpu_test__();
            return x;
        }

        // Copy elements of 'A' to the elements of the matrix
        void copy(const Matrix<T>& A) {
            assert(same_dimension(A));
            cudaMemcpy((void*)data().get(), (void*)A.data().get(), size() * sizeof(T), cudaMemcpyDeviceToDevice);
            Gpu::__gpu_test__();
        }

        // Copy the SubMatrix of 'A' starting at ('i0','j0') and
        // of size 'height()' * 'width()' to the matrix
        void extract(const Matrix<T>& A,  int i0, int j0) {
            assert(i0 + height() <= A.height());
            assert(j0 + width() <= A.width());
            void* dst = (void*)data().get();
            void* src = (void*)&(A.data().get()[i0+j0 * A.height()]);
            cudaMemcpy2D(dst, (size_t)height()*sizeof(T), src, (size_t)A.height()*sizeof(T), (size_t)height() * sizeof(T), (size_t)width(), cudaMemcpyDeviceToDevice);
            
        }
        void extract_to(Matrix<T>& A, int i0, int j0) {
            assert(i0 + height() <= A.height());
            assert(j0 + width() <= A.width());
            for (int j = 0; j < width(); j++)
            {
                cudaMemcpy( (void*)&A.data().get()[i0 + (j0 + j) * A.height()], (void*)&data().get()[j * height()], height() * sizeof(T), cudaMemcpyDeviceToDevice);
            }
        }

        // Import 'A' in the SubMatrix of the current matrix starting at ('i0','j0') and
        // of size 'A.height()' * 'B.width()'
        void rev_extract(Matrix<T>& A, int i0, int j0) {
            assert(i0 + A.height() <= height());
            assert(j0 + A.width() <= width());
            void* dst = (void*)&(data().get()[i0 + j0 * height()]);
            void* src = (void*)A.data().get();
            cudaMemcpy2D(dst, (size_t)height() * sizeof(T), src, (size_t)A.height() * sizeof(T), (size_t)A.height() * sizeof(T), (size_t)A.width(), cudaMemcpyDeviceToDevice);
            Gpu::__gpu_test__();
        }

        // Cpu::Matrix version of copy in order to have compatibility
        void copy(const Cpu::Matrix<T>& A) {
            assert(same_dimension(A));
            cudaMemcpy((void*)data().get(), (void*)A.data().get(), size() * sizeof(T), cudaMemcpyHostToDevice);
            Gpu::__gpu_test__();
        }
        void copy_to(const Cpu::Matrix<T>& A) const {
            assert(same_dimension(A));
            cudaMemcpy((void*)A.data().get(), (void*)data().get(), size() * sizeof(T), cudaMemcpyDeviceToHost);
            Gpu::__gpu_test__();
        }

        // Fill the matrix with the value in 'x'
        void fill(T x) {
            Cpu::Matrix<T> A(height(), width());
            A.fill(x);
            copy(A);
        }

        // Fill the matrix with random values between 'min' and 'max'
        void randfill(T min, T max) {
            Cpu::Matrix<T> A(height(), width());
            A.randfill(min, max);
            copy(A);
        }

        // Fonctions de calcul


        // Scalar function performs element-wise scaling and addition of the elements of matrix A with given alpha and beta coefficients
        void Scalar(T alpha, const Matrix<T>& A, T beta) {
            assert(same_dimension(A));
            T* my_data = data().get();
            T* A_data = A.data().get();
            T alpha_ = alpha;
            T beta_ = beta;
            auto fptr = [=] __device__(int idx) {
                my_data[idx] = alpha_ * A_data[idx] + beta_;
            };
            Gpu::call_gpu_iter(size(), fptr);
        }

        // Linear function performs element-wise linear combination of the elements of matrix A and B with given alpha and beta coefficients.
        void Linear(T alpha, const Matrix<T>& A, T beta, const Matrix<T>& B) {
            assert(same_dimension(A));
            assert(same_dimension(B));
            T* my_data = data().get();
            T* A_data = A.data().get();
            T* B_data = B.data().get();
            T alpha_ = alpha;
            T beta_ = beta;
            auto fptr = [=] __device__(int idx) {
                my_data[idx] = alpha_ * A_data[idx] + beta_ * B_data[idx];
            };
            Gpu::call_gpu_iter(size(), fptr);
            Gpu::__gpu_test__();
        }

        // Sum function performs the element-wise sum of the elements from all the matrix in As vector
        void Sum(std::vector<Matrix<T>> As) {

            std::vector<T*> As_ptr;
            for (int i = 0; i < As.size(); i++)
            {
                As_ptr.push_back(As[i].data().get());
            }

            Gpu::SafePointer<T*> As_data(As.size());
            cudaMemcpy((void*)As_data.get(), (void*)As_ptr.data(), As.size() * sizeof(T*), cudaMemcpyHostToDevice);

            T* my_data = data().get();
            T** datas = As_data.get();
            int n = As.size();

            auto fptr = [=] __device__(int idx) {
                T acc = 0;
                for (int i = 0; i < n; i++)
                {
                    acc += datas[i][idx];
                }
                my_data[idx] = acc;
            };

            Gpu::call_gpu_iter(size(), fptr);
            Gpu::__gpu_test__();
        }

        // Matmul function performs element-wise multiplication of the elements of matrices A and B.
        void Matmul(const Matrix<T>& A, const Matrix<T>& B) {
            assert(same_dimension(A));
            assert(same_dimension(B));
            T* my_data = data().get();
            T* A_data = A.data().get();
            T* B_data = B.data().get();
            auto fptr = [=] __device__(int idx) {
                my_data[idx] = A_data[idx] * B_data[idx];
            };
            Gpu::call_gpu_iter(size(), fptr);
            Gpu::__gpu_test__();
        }

        // Matprod function performs matrix product of matrices A and B, with possible transpositions specified by transa and transb.
        void Matprod(const Matrix<T>& A, char transa, const Matrix<T>& B, char transb, T alpha = 1, T beta = 0);

        // Power function performs element-wise power operation of the elements of matrix A with given power p.
        void Power(const Matrix<T>& A, T p);

        // Reverse a square matrix
        void Reverse(const Matrix<T>& A) {

            assert(same_dimension(A));
            assert(height() == width());

            int n = height();

            Gpu::Matrix<T> TMP(n, n);
            TMP.copy(A);

            Cpu::Matrix<T> Id(n, n);
            Id.fill(0);
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    Id.set(i, j, i == j ? (T)1 : (T)0);
                }
            }
            copy(Id);

            for (int i = 0; i < n; i++)
            {
                __cancel_column__ << <n, BLOCK_SIZE >> > (TMP.data().get(), data().get(), n, i);
            }
            __normalize_column__ << <n, BLOCK_SIZE >> > (TMP.data().get(), data().get(), n);

        }

        // Classic functions

        // Element-wise sigmoid function of the elements of matrix A
        void Sigmoid(const Matrix<T>& A);

        // Element-wise tanh function of the elements of matrix A
        void Tanh(const Matrix<T>& A);

        // Element-wise exp function of the elements of matrix A
        void Exp(const Matrix<T>& A);


    };

    template<> void Matrix<float>::Power(const Matrix<float>& A, float p) {
        // Check if the dimensions of the matrices match
        assert(same_dimension(A));

        // Get the pointers to the data of this matrix and A
        float* my_data = data().get();
        float* A_data = A.data().get();

        // Copy the power scalar to a variable that can be used in the lambda function
        float p_ = p;

        // Define a lambda function to calculate the power of each element in the matrix A
        auto fptr = [=] __device__(int idx) {
            my_data[idx] = powf(A_data[idx], p_);
        };

        // Call the GPU function to perform the calculation
        Gpu::call_gpu_iter(size(), fptr);

        // Test the GPU function
        Gpu::__gpu_test__();
    }
    template<> void Matrix<double>::Power(const Matrix<double>& A, double p) {
        // For commentaries see the float version
        assert(same_dimension(A));
        double* my_data = data().get();
        double* A_data = A.data().get();
        double p_ = p;
        auto fptr = [=] __device__(int idx) {
            my_data[idx] = pow(A_data[idx], p_);
        };
        Gpu::call_gpu_iter(size(), fptr);
        Gpu::__gpu_test__();
    }

    template<> void Matrix<float>::Sigmoid(const Matrix<float>& A) {
        // Check if the dimensions of the input matrix and the current matrix are the same
        assert(same_dimension(A));
        // Get the data pointers for the current matrix and the input matrix
        float* my_data = data().get();
        float* A_data = A.data().get();
        // Create a lambda function to calculate the sigmoid function on each element
        auto fptr = [=] __device__(int idx) {
            // Calculate the exponential of the input value
            float tmp = expf(A_data[idx]);
            // Calculate the sigmoid function for the current element
            my_data[idx] = tmp / (tmp + 1);
        };
        // Call the GPU to execute the lambda function on each element
        Gpu::call_gpu_iter(size(), fptr);
        // Check if the GPU execution was successful
        Gpu::__gpu_test__();
    }
    template<> void Matrix<float>::Tanh(const Matrix<float>& A) {
        // For commentaries see the float version of Sigmoid
        assert(same_dimension(A));
        float* my_data = data().get();
        float* A_data = A.data().get();
        auto fptr = [=] __device__(int idx) {
            my_data[idx] = tanhf(A_data[idx]);
        };
        Gpu::call_gpu_iter(size(), fptr);
        Gpu::__gpu_test__();
    }
    template<> void Matrix<float>::Exp(const Matrix<float>& A) {
        // For commentaries see the float version of Sigmoid
        assert(same_dimension(A));
        float* my_data = data().get();
        float* A_data = A.data().get();
        auto fptr = [=] __device__(int idx) {
            my_data[idx] = expf(A_data[idx]);
        };
        Gpu::call_gpu_iter(size(), fptr);
        Gpu::__gpu_test__();
    }

    template<> void Matrix<double>::Sigmoid(const Matrix<double>& A) {
        // For commentaries see the float version of Sigmoid
        assert(same_dimension(A));
        double* my_data = data().get();
        double* A_data = A.data().get();
        auto fptr = [=] __device__(int idx) {
            double tmp = exp(A_data[idx]);
            my_data[idx] = tmp / (tmp + 1);
        };
        Gpu::call_gpu_iter(size(), fptr);
        Gpu::__gpu_test__();
    }
    template<> void Matrix<double>::Tanh(const Matrix<double>& A) {
        // For commentaries see the float version of Sigmoid
        assert(same_dimension(A));
        double* my_data = data().get();
        double* A_data = A.data().get();
        auto fptr = [=] __device__(int idx) {
            my_data[idx] = tanh(A_data[idx]);
        };
        Gpu::call_gpu_iter(size(), fptr);
        Gpu::__gpu_test__();
    }
    template<> void Matrix<double>::Exp(const Matrix<double>& A) {
        // For commentaries see the float version of Sigmoid
        assert(same_dimension(A));
        double* my_data = data().get();
        double* A_data = A.data().get();
        auto fptr = [=] __device__(int idx) {
            my_data[idx] = exp(A_data[idx]);
        };
        Gpu::call_gpu_iter(size(), fptr);
        Gpu::__gpu_test__();
    }


    template<> void Matrix<double>::Matprod(const Matrix<double>& A, char transa, const Matrix<double>& B, char transb, double alpha, double beta) {

        cublasOperation_t op_a = transa == 't' || transa == 'T' ? cublasOperation_t::CUBLAS_OP_T : cublasOperation_t::CUBLAS_OP_N;
        cublasOperation_t op_b = transb == 't' || transb == 'T' ? cublasOperation_t::CUBLAS_OP_T : cublasOperation_t::CUBLAS_OP_N;

        // Check dimensions
        int k = 0;
        if (op_a == cublasOperation_t::CUBLAS_OP_T) {
            k = A.height();
            assert(A.width() == height());
        }
        else {
            k = A.width();
            assert(A.height() == height());
        }

        if (op_b == cublasOperation_t::CUBLAS_OP_T) {
            assert(B.height() == width());
            assert(B.width() == k);
        }
        else {
            assert(B.width() == width());
            assert(B.height() == k);
        }

        // Call the cublas matrix product functions
        cublasDgemm_v2(ENV_CUBLAS__, op_a, op_b, height(), width(), k, &alpha, A.data().get(), A.height(), B.data().get(), B.height(), &beta, data().get(), height());
        Gpu::__gpu_test__();

    }
    template<> void Matrix<float>::Matprod(const Matrix<float>& A, char transa, const Matrix<float>& B, char transb, float alpha, float beta) {

        // For commentaries see the float version

        cublasOperation_t op_a = transa == 't' || transa == 'T' ? cublasOperation_t::CUBLAS_OP_T : cublasOperation_t::CUBLAS_OP_N;
        cublasOperation_t op_b = transb == 't' || transb == 'T' ? cublasOperation_t::CUBLAS_OP_T : cublasOperation_t::CUBLAS_OP_N;

        // Check dimensions
        int k = 0;
        if (op_a == cublasOperation_t::CUBLAS_OP_T) {
            k = A.height();
            assert(A.width() == height());
        }
        else {
            k = A.width();
            assert(A.height() == height());
        }

        if (op_b == cublasOperation_t::CUBLAS_OP_T) {
            assert(B.height() == width());
            assert(B.width() == k);
        }
        else {
            assert(B.width() == width());
            assert(B.height() == k);
        }

        int h1 = A.height();
        int h2 = B.height();

        // Call the cublas matrix product function
        cublasSgemm_v2(ENV_CUBLAS__, op_a, op_b, height(), width(), k, &alpha, A.data().get(), h1, B.data().get(), h2, &beta, data().get(), height());
        Gpu::__gpu_test__();

    }

    // __global__ functions used in reverse
    // Precondition : a_i0_i0 != 0
    template<class T> __global__ void __cancel_column__(T* A, T* A_inverse, int taille, int i0) {

        int j = blockIdx.x;
        int nb_thread = blockDim.x;
        int idx_thread = threadIdx.x;

        if (j != i0) {

            T a_i0_j = A[IDXH(i0, j, taille)];
            T a_i0_i0 = A[IDXH(i0, i0, taille)];

            T* Cj = &A[IDXH(0, j, taille)];
            T* Cj_inverse = &A_inverse[IDXH(0, j, taille)];

            T* Ci0 = &A[IDXH(0, i0, taille)];
            T* Ci0_inverse = &A_inverse[IDXH(0, i0, taille)];

            for (int i = idx_thread; i < taille; i += nb_thread) {
                Cj[i] = (a_i0_i0 * Cj[i] - a_i0_j * Ci0[i]) / a_i0_i0;
                Cj_inverse[i] = (a_i0_i0 * Cj_inverse[i] - a_i0_j * Ci0_inverse[i]) / a_i0_i0;
            }

        }

    }
    // Precondition : a_i0_i0 != 0
    template<class T> __global__ void __normalize_column__(T* A, T* A_inverse, int taille) {

        int j = blockIdx.x;
        int nb_thread = blockDim.x;
        int idx_thread = threadIdx.x;

        T a_j_j = A[IDXH(j, j, taille)];
        T* Cj_inverse = &A_inverse[IDXH(0, j, taille)];

        for (int i = idx_thread; i < taille; i += nb_thread) {
            Cj_inverse[i] = Cj_inverse[i] / a_j_j;
        }

    }

    // Optimisation functions

    template<class T> void __opti_vect_scal__(T alpha, T* vector, int n, int inc = 0);
    template<> void __opti_vect_scal__(double alpha, double* vector, int n, int inc) {
        cublasDscal_v2(ENV_CUBLAS__, n, &alpha, vector, inc);
    }
    template<> void __opti_vect_scal__(float alpha, float* vector, int n, int inc) {
        cublasSscal_v2(ENV_CUBLAS__, n, &alpha, vector, inc);
    }

    // Print function of the class Gpu::Matrix when calling std::cout << A
    template<typename T> std::ostream& operator<<(std::ostream& os, const Gpu::Matrix<T>& A)
    {
        using namespace Cpu;
        Cpu::Matrix<T> TMP(A.height(), A.width());
        A.copy_to(TMP);
        std::cout << TMP;
        return os;
    }





    // MatrixSave class
    // This class store a set of matrix and is able of save them to a file
    // or to load them from a file
    template<typename T>
    class MatrixSave {
    private:
        std::map<std::string, Gpu::Matrix<T>> map_matrix_;  // map to store the matrices with their names as keys

    public:
        MatrixSave() {}  // default constructor

        MatrixSave(std::map<std::string, Cpu::Matrix<T>> map_matrix)
            : map_matrix_(map_matrix) {}  // constructor that takes a map of matrices as input

        // This function add a new matrix to set indexing it by 'name' in 'map_matrix_'s
        void add_matrix(Matrix<T> t, std::string name) {
            // check if the name already exists in the map
            auto it = map_matrix_.find(name);
            if (it != map_matrix_.end()) {
                std::cout << "Error : Name already ussed !\n";
                assert(0);  // stop the program execution if the name is already used
            }
            if (name.find("\n") != std::string::npos) {  // check if the name contains a newline character
                std::cout << "Error : Invalid name !\n";
                assert(0);  // stop the program execution if the name contains a newline character
            }
            map_matrix_[name] = t;  // add the matrix to the map
        }

        // This function saves data from matrices stored in `map_matrix_` to a binary file of name 'file'.
        void save(std::string file) {
            // open the file for writing in binary mode
            std::ofstream fichier(file, std::ios::out | std::ios::binary);

            for (auto it = map_matrix_.begin(); it != map_matrix_.end(); it++) {
                // write the name of the matrix to the file
                fichier.write(it->first.c_str(), it->first.size() * sizeof(char));
                fichier.write("\n", sizeof(char));  // write a newline character after the name

                // allocate a buffer to store the data from the matrix on the host
                T* buffer = new T[it->second.size()];
                // copy the data from the GPU to the host
                cudaMemcpy(buffer, it->second.data().get(), it->second.size() * sizeof(T), cudaMemcpyDeviceToHost);
                // write the data to the file
                fichier.write(reinterpret_cast<char*>(buffer), it->second.size() * sizeof(T));

                delete[] buffer;  // free the memory used by the buffer
            }

            fichier.close();  // close the file
        }

        // This  function loads data from a binary file into the matrices stored in `map_matrix_`.
        void load(std::string file) {
            // Open the file for reading in binary mode
            std::ifstream fichier(file, std::ios::in | std::ios::binary);

            // Loop over each matrix stored in the file
            for (std::string name; std::getline(fichier, name);) {

                // Get the size of the matrix to be loaded
                int size = map_matrix_[name].size();

                // Get the data of the matrix to be loaded
                T* data = map_matrix_[name].data().get();

                // Allocate a buffer to store the data from the file
                T* buffer = new T[size];

                // Read the data from the file into the buffer
                fichier.read(reinterpret_cast<char*>(buffer), size * sizeof(T));

                // Copy the data from the buffer to the GPU
                cudaMemcpy(data, buffer, size * sizeof(T), cudaMemcpyHostToDevice);

                // Free the memory used by the buffer
                delete[] buffer;
            }

            // Close the file
            fichier.close();
        }

        // This function print matrices in 'map_matrix_'s
        void print() {
            std::cout << "\n############\n Print begin \n############\n\n";
            for (auto it = map_matrix_.begin(); it != map_matrix_.end(); it++) {
                std::cout << "## " << it->first << "\n";
                std::cout << it->second << std::endl;
                std::cout << "##\n\n";
            }
            std::cout << "############\n Print end \n############\n";
        }

    };


    template<typename T> struct MatrixPair {

        Matrix<T> X;
        Matrix<T> Y;

        MatrixPair(Gpu::Matrix<T> X, Gpu::Matrix<T> Y) : X(X), Y(Y) {}
        MatrixPair(const Cpu::MatrixPair<T>& pair) : X(pair.X), Y(pair.Y) {}

    };

}

namespace Cpu {
    template<> void Matrix<double>::Matprod(const Matrix<double>& A, char transa, const Matrix<double>& B, char transb) {

        Gpu::Matrix<double> a(A), b(B), c(*this);
        c.Matprod(a, transa, b, transb);
        c.copy_to(*this);

    }
}