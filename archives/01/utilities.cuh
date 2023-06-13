#pragma once

// CUDA libraries
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// Standard libraries
#include <assert.h>
#include <fstream>
#include <iostream>
#include <vector>

// Define block size for CUDA kernel function calls
#define BLOCK_SIZE 1024
#define BLOCK_SIZE_2D_1 16
#define BLOCK_SIZE_2D_2 32
#define BLOCK_SIZE_2D_3 64

// Macro for optimized division calculation
#define OPTI_DIV(n, m) ((n + m - 1) / m)



// GPU namespace
// This namespace provides utility functions and objects for GPU operations
namespace Gpu {

    // This function checks for any errors that may have occurred in the GPU and ends the program if necessary.
    void __gpu_test__() {
        // Synchronize the GPU with the CPU
        cudaDeviceSynchronize();

        // Retrieve any potential error
        cudaError_t cuda_err = cudaGetLastError();
        if (cuda_err != cudaSuccess) {
            std::cout << "#!# Error : " << cudaGetErrorName(cuda_err) << std::endl;
            assert(0);
        }
    }

    // This function executes the lambda __device__ function 'fptr' for each i from 0 to (n-1) on the GPU
    template<class F> __global__ void __gpu_iter__(int n, F fptr) {

        // Calculate the index of the current thread
        int idx = blockDim.x * blockIdx.x + threadIdx.x;

        // Check if the index is within the range of n
        if (idx < n) {
            // If the index is valid, call the lambda __device__ function `fptr` 
            // with the current index as its argument.
            fptr(idx);
        }
    }

    // This function launches the `__gpu_iter__`  __global__ function on the GPU with optimized dimensions of call
    template<class F> void call_gpu_iter(int n, F fptr) {
        __gpu_iter__ << <OPTI_DIV(n, BLOCK_SIZE), BLOCK_SIZE >> > (n, fptr);
    }

    // This function retrieves the value stored in a GPU memory location and returns it.
    template<class T> T gpu_get(T* ptr_gpu) {
        T elt_cpu;
        cudaMemcpy((void*)&elt_cpu, (void*)ptr_gpu, sizeof(T), cudaMemcpyDeviceToHost);
        __gpu_test__();
        return elt_cpu;
    }

    // This function copies a value to a GPU memory location.
    template<class T> T gpu_set(T* ptr_gpu, T valeur) {
        cudaMemcpy((void*)ptr_gpu, (void*)&valeur, sizeof(T), cudaMemcpyHostToDevice);
        __gpu_test__();
        return valeur;
    }


    // SafePointer class
    // This class provides a GPU memory management using a reference counting mechanism to keep track of
    // the number of references to the same memory allocation to avoid leaks
    template <typename T>
    class SafePointer {
    private:
        // Pointer to GPU memory
        T* ptr_;

        // Reference counter
        int* count_;

    public:
        // Constructor allocating GPU memory of size `size` and pointing to it
        SafePointer(size_t size) : ptr_(nullptr), count_(new int(1)) {
            // TODO revenir à l'état précédent si ne fonctionne pas
            /*
            cudaMalloc((void**)&ptr_, size * sizeof(T));
            Gpu::__gpu_test__();*/
            if (size > 0) {
                cudaMalloc((void**)&ptr_, size * sizeof(T));
                Gpu::__gpu_test__();
            }
        }

        // Constructor taking a pointer to GPU memory and setting reference counter to 1
        explicit SafePointer(T* ptr) : ptr_(ptr), count_(new int(1)) {}

        // Copy constructor incrementing the reference counter of `other`
        SafePointer(const SafePointer<T>& other)
            : ptr_(other.ptr_), count_(other.count_) {
            (*count_)++;
        }

        // Assignment operator freeing the previous pointer and copying the data from `other`
        SafePointer<T>& operator=(const SafePointer<T>& other) {
            if (this != &other) {
                // Decrement and free previous pointer if reference count reaches 0
                (*count_)--;
                if (*count_ == 0) {
                    // TODO remplacer par : cudaFree((void*)ptr_); si ne fonctionne pas
                    if (ptr_ != nullptr) { cudaFree((void*)ptr_); }
                    delete count_;
                    Gpu::__gpu_test__();
                }

                // Copy new data
                ptr_ = other.ptr_;
                count_ = other.count_;
                (*count_)++;
            }
            return *this;
        }

        // Destructor decrementing the reference counter and freeing the pointer if needed
        ~SafePointer() {
            (*count_)--;
            if (*count_ == 0) {
                // TODO remplacer par : cudaFree((void*)ptr_); si ne fonctionne pas
                if (ptr_ != nullptr) { cudaFree((void*)ptr_); }
                delete count_;
                Gpu::__gpu_test__();
            }
        }

        // Overloading dereference operator
        //TODO : enlever les assert si ne fonctionne pas
        T& operator*() const { assert(ptr_ != nullptr); return *ptr_; }
        T* operator->() const { assert(ptr_ != nullptr); return ptr_; }
        T& operator[](size_t index) const { assert(ptr_ != nullptr); return ptr_[index]; }

        // Getter for the pointer
        T* get() const { return ptr_; }

        // Getter for the reference count
        int use_count() const { return *count_; }
    };

}



// RANDOM namespace
// This namespace provides utility functions for random number generation.
namespace Random {

    // random function template
    // Generates a random number between the `low` and `up` values, both inclusive.
    // The type of the generated number is the same as the type of the `low` and `up` values.
    // The type must support being cast to a `double`.
    template<typename T> T random(T low, T up) {
        unsigned int rd = rand();
        double x = (double)rd / (double)RAND_MAX;
        return (T)((double)low + x * ((double)up - (double)low));
    }

    // knuth_shuffle function
    // Performs a Knuth shuffle on the input `tab` vector.
    // The shuffle randomizes the order of the elements in the vector.
    void knuth_shuffle(std::vector<int>& tab) {
        for (int i = 0; i < tab.size(); i++)
        {
            // Generate a random index in the remaining unshuffled portion of the vector.
            int j = rand() % (tab.size() - i);
            j = j + i;

            // Swap the elements at indices i and j.
            int tmp = tab[i];
            tab[i] = tab[j];
            tab[j] = tmp;
        }
    }

}

// STRING namespace
// Customs functions to manage string
namespace String {

    std::vector<std::string> parse(std::string str, std::string separateur) {
        std::vector<std::string> vecteur;
        size_t pos = 0;
        std::string token;
        while ((pos = str.find(separateur)) != std::string::npos) {
            token = str.substr(0, pos);
            vecteur.push_back(token);
            str.erase(0, pos + separateur.length());
        }
        vecteur.push_back(str);
        return vecteur;
    }
    std::string replace(std::string& chaine_de_base, const std::string& chaine_a_remplacer, const std::string& nouvelle_chaine) {

        std::string chaine(chaine_de_base);

        size_t found = chaine.find(chaine_a_remplacer);
        while (found != std::string::npos) {
            chaine.replace(found, chaine_a_remplacer.size(), nouvelle_chaine);
            found = chaine.find(chaine_a_remplacer, found + nouvelle_chaine.size());
        }

        return chaine;
    }

    char MAJUSCULE(char min) // minuscule --> MAJUSCULE
    {
        char maj;
        maj = (char)min;
        if ((int)min <= 90 && (int)min >= 65) // majuscule
        {
            return (char)maj;
        }
        else
        {
            if ((int)min <= 122 && (int)min >= 97) // minuscule
            {
                return (char)((int)maj - 32);
            }
            else
            {
                return min; // ce n'est pas une lettre
            }
        }
    }
    char minuscule(char maj) // MAJUSCULE --> minuscule
    {
        char min;
        min = (char)maj;
        if ((int)maj <= 90 && (int)maj >= 65) // majuscule
        {
            return (char)((int)min + 32);
        }
        else
        {
            if ((int)maj <= 122 && (int)maj >= 97) // minuscule
            {
                return (char)min;
            }
            else
            {
                return maj; // ce n'est pas une lettre
            }
        }
    }
    std::string minuscule(std::string& str) {
        std::string new_str(str);
        for (size_t i = 0; i < new_str.size(); i++)
        {
            new_str[i] = minuscule(new_str[i]);
        }
        return new_str;
    }
    std::string MAJUSCULE(std::string str) {
        std::string new_str(str);
        for (size_t i = 0; i < new_str.size(); i++)
        {
            new_str[i] = MAJUSCULE(new_str[i]);
        }
        return new_str;
    }
    
}


namespace File {

    void add_to_csv(std::string file_name, std::vector<std::string> vec, bool end_line = true) {
        
        std::ofstream fichier(file_name, std::ios::app);

        for (int i = 0; i < vec.size(); i++) {
            fichier << vec[i] << ';';
        }

        if (end_line) {
            fichier << std::endl;
        }

        fichier.close();

    }
}
