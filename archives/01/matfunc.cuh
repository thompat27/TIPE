#pragma once

// INCLUDES /////////////////////////////////////////////////////////////////////////////////////////////////////

// STD
#include <iostream> // Standard input and output streams library for console output
#include <fstream> // Standard file stream library for file input and output
#include <vector> // Standard template library for dynamic arrays
#include <map> // Standard template library for associative arrays
#include <stack> // Standard template library for stacks
#include <string> // Standard string library
#include <stdarg.h> // Header for variable number of arguments in a function
#include <assert.h> // Header for error handling
#include <memory> // Standard memory management library

// INTERNAL
#include "matrix.cuh"

// DEFINES /////////////////////////////////////////////////////////////////////////////////////////////////////

#define EPSILON 1e-8


// GPU /////////////////////////////////////////////////////////////////////////////////////////////////////

namespace Gpu {

    /*
    
    This part is defining a set of classes and functions related to matrix operations for symbolic differentiation of matrix functions using reverse automatic differentiation (RAD).
    The ObjMatFunc class is a base class for matrix functions and it's templated on a data type T.
    The MatFunc is a typedef for a shared pointer of ObjMatFunc<T>, which is used to store the matrix functions and is the main object used by the user.


    There are several classes derived from ObjMatFunc class that each represent a matrix operation :

      - ObjMatFuncScalar represents a scalar matrix function : alpha*A+beta with alpha, beta some numbers and A a matrix

      - ObjMatFuncLinear represents a linear matrix function : alpha*A + beta*B

      - ObjMatFuncSum represents the sum of matrix functions : A1+A2+A3 ...

      - ObjMatFuncMatMul represents the element-wise matrix multiplication of matrix functions : A1*A2

      - ObjMatFuncMatProd represents the product of two matrices functions : A^B

      - ObjMatFuncPower represents the element-wise power of a matrix function : A^p

      - ObjMatFuncSigmoid and ObjMatFuncTanh represent the sigmoid and tanh activation of matrix a function respectively

      - ObjMatFuncExp represents the exponential of a matrix function

      - ObjMatFuncCopy allow the creation of 'cycle' in the functional graphs and is very specific to situation like implementing LSTM units in deep learning
     

    MatFuncSet, MatFuncMap, MatFuncIndex, and MatFuncArr are typedefs for map, map, map and vector of MatFunc respectively.

    */

    // Predeclaration of the classes

    template<typename T> class ObjMatFunc;
    template<typename T> class ObjMatFuncCopy;
    template<typename T> class ObjMatFuncScalar;
    template<typename T> class ObjMatFuncLinear;
    template<typename T> class ObjMatFuncSum;
    template<typename T> class ObjMatFuncMatMul;
    template<typename T> class ObjMatFuncMatProd;
    template<typename T> class ObjMatFuncPower;
    template<typename T> class ObjMatFuncSigmoid;
    template<typename T> class ObjMatFuncTanh;
    template<typename T> class ObjMatFuncExp;
    template<typename T> class ObjMatFuncExtract;
    template<typename T> class ObjMatFuncReverseExtract;
    template<typename T> class ObjMatFuncConcat;
    template<typename T> class ObjMatFuncBatchConcat;



    // Typedef of the different type of structure

    template<typename T> using MatFunc = std::shared_ptr<ObjMatFunc<T>>;
    template<typename T> using MatFuncSet = std::map<MatFunc<T>, int>;
    template<typename T> using MatFuncMap = std::map<MatFunc<T>, MatFunc<T>>;
    template<typename T> using MatFuncIndex = std::map<std::string, MatFunc<T>>;
    template<typename T> using MatFuncArr = std::vector<MatFunc<T>>;



    // Low levels functions to create and manage MatFunc

    // TOCHANGE if problems
    
    // #!# Lors de rajout de nouvelles fonctions, bien vérifiés la ligne __add_parents__ pour tous les enfants !
    template<typename T> bool is_matfunc_null(MatFunc<T> mf) {
        return mf.get() == nullptr;
    }
    template<typename T> MatFunc<T> matfunc_null() {
        std::shared_ptr<ObjMatFunc<T>> mf_null;
        return mf_null;
    }
    template<typename T> MatFunc<T> matfunc(int height, int width, T x = (T)0) {
        MatFunc<T> matfunc(new ObjMatFunc<T>(height, width));
        matfunc->set_owner();
        matfunc->matrix().fill(x);
        return matfunc;
    }
    template<typename T> MatFunc<T> matfunc(Matrix<T> matrix) {
        MatFunc<T> matfunc(new ObjMatFunc<T>(matrix));
        matfunc->set_owner();
        return matfunc;
    }
    template<typename T> MatFunc<T> matfunc_scalar(T alpha, MatFunc<T> A, T beta) {
        MatFunc<T> matfunc((ObjMatFunc<T>*) new ObjMatFuncScalar<T>(alpha, A, beta));
        A->__add_parent__(matfunc);
        return matfunc;
    }
    template<typename T> MatFunc<T> matfunc_linear(T alpha, MatFunc<T> A, T beta, MatFunc<T> B) {
        MatFunc<T> matfunc((ObjMatFunc<T>*) new ObjMatFuncLinear<T>(alpha, A, beta, B));
        A->__add_parent__(matfunc);
        B->__add_parent__(matfunc);
        return matfunc;
    }
    template<typename T> MatFunc<T> matfunc_sum(std::vector<MatFunc<T>> As) {
        MatFunc<T> matfunc((ObjMatFunc<T>*) new ObjMatFuncSum<T>(As));
        for (int i = 0; i < As.size(); i++)
        {
            As[i]->__add_parent__(matfunc);
        }
        return matfunc;
    }
    template<typename T> MatFunc<T> matfunc_matmul(MatFunc<T> A, MatFunc<T> B) {
        MatFunc<T> matfunc((ObjMatFunc<T>*) new ObjMatFuncMatMul<T>(A, B));
        A->__add_parent__(matfunc);
        B->__add_parent__(matfunc);
        return matfunc;
    }
    template<typename T> MatFunc<T> matfunc_matprod(MatFunc<T> A, char transa, MatFunc<T> B, char transb) {
        MatFunc<T> matfunc((ObjMatFunc<T>*) new ObjMatFuncMatProd<T>(A, transa, B, transb));
        A->__add_parent__(matfunc);
        B->__add_parent__(matfunc);
        return matfunc;
    }
    template<typename T> MatFunc<T> matfunc_power(MatFunc<T> A, T p) {
        MatFunc<T> matfunc((ObjMatFunc<T>*) new ObjMatFuncPower<T>(A, p));
        A->__add_parent__(matfunc);
        return matfunc;
    }
    template<typename T> MatFunc<T> matfunc_sigmoid(MatFunc<T> A) {
        MatFunc<T> matfunc((ObjMatFunc<T>*) new ObjMatFuncSigmoid<T>(A));
        A->__add_parent__(matfunc);
        return matfunc;
    }
    template<typename T> MatFunc<T> matfunc_tanh(MatFunc<T> A) {
        MatFunc<T> matfunc((ObjMatFunc<T>*) new ObjMatFuncTanh<T>(A));
        A->__add_parent__(matfunc);
        return matfunc;
    }
    template<typename T> MatFunc<T> matfunc_exp(MatFunc<T> A) {
        MatFunc<T> matfunc((ObjMatFunc<T>*) new ObjMatFuncExp<T>(A));
        A->__add_parent__(matfunc);
        return matfunc;
    }
    template<typename T> MatFunc<T> matfunc_extract(MatFunc<T> A, int i0, int j0, int height, int width) {
        MatFunc<T> matfunc((ObjMatFunc<T>*) new ObjMatFuncExtract<T>(A, std::make_shared<int>(i0), std::make_shared<int>(j0), height, width));
        A->__add_parent__(matfunc);
        return matfunc;
    }
    template<typename T> MatFunc<T> matfunc_rev_extract(MatFunc<T> A, int i0, int j0, int height, int width) {
        MatFunc<T> matfunc((ObjMatFunc<T>*) new ObjMatFuncReverseExtract<T>(A, std::make_shared<int>(i0), std::make_shared<int>(j0), height, width));
        A->__add_parent__(matfunc);
        return matfunc;
    }
    template<typename T> MatFunc<T> matfunc_concat(MatFunc<T> A, MatFunc<T> B, char type) {
        MatFunc<T> matfunc((ObjMatFunc<T>*) new ObjMatFuncConcat<T>(A, B, type));
        A->__add_parent__(matfunc);
        B->__add_parent__(matfunc);
        return matfunc;
    }
    template<typename T> MatFunc<T> matfunc_batch_concat(std::vector<MatFunc<T>> As, char type) {
        MatFunc<T> matfunc((ObjMatFunc<T>*) new ObjMatFuncBatchConcat<T>(As, type));
        for (int i = 0; i < As.size(); i++)
        {
            As[i]->__add_parent__(matfunc);
        }
        return matfunc;
    }

    // Dynamic arguments
    // Arguments are stored in shared_ptr in order to allow direct modifications of them
    template<typename T> MatFunc<T> matfunc_extract(MatFunc<T> A, std::shared_ptr<int> i0, std::shared_ptr<int> j0, int height, int width) {
        MatFunc<T> matfunc((ObjMatFunc<T>*) new ObjMatFuncExtract<T>(A, i0, j0, height, width));
        A->__add_parent__(matfunc);
        return matfunc;
    }
    template<typename T> MatFunc<T> matfunc_rev_extract(MatFunc<T> A, std::shared_ptr<int> i0, std::shared_ptr<int> j0, int height, int width) {
        MatFunc<T> matfunc((ObjMatFunc<T>*) new ObjMatFuncReverseExtract<T>(A, i0, j0, height, width));
        A->__add_parent__(matfunc);
        return matfunc;
    }


    // High level functions and overloaded functions to create and manage MatFunc
    // #!# : A ^ B is used to create a Matrix Product MatFunc if A and B are some MatFunc

    template<typename T> Gpu::MatFunc<T> nullmf() {
        return matfunc_null<T>();
    }
    template<typename T> Gpu::MatFunc<T> isnullmf(MatFunc<T> mf) {
        return is_matfunc_null<T>(mf);
    }
    template<typename T> Gpu::MatFunc<T> newmf(int h, int w, T val = (T)0) {
        return Gpu::matfunc<T>(h, w, val);
    }
    template<typename T> Gpu::MatFunc<T> newmf(Gpu::Matrix<T> matrix) {
        auto mf = Gpu::newmf<T>(matrix.height(), matrix.width());
        mf->matrix().copy(matrix);
        return mf;
    }
    template<typename T> Gpu::MatFunc<T> newmf(Cpu::Matrix<T> matrix) {
        auto mf = Gpu::newmf<T>(matrix.height(), matrix.width());
        mf->matrix().copy(matrix);
        return mf;
    }
    template<typename T> Gpu::MatFunc<T> newmf_of_matrix(Gpu::Matrix<T> matrix) {
        return Gpu::matfunc<T>(matrix);
    }
    template<typename T> Gpu::MatFunc<T> operator+(Gpu::MatFunc<T> A, Gpu::MatFunc<T> B) {
        return Gpu::matfunc_linear<T>(1, A, 1, B);
    }
    template<typename T> Gpu::MatFunc<T> operator-(Gpu::MatFunc<T> A, Gpu::MatFunc<T> B) {
        return Gpu::matfunc_linear<T>(1, A, -1, B);
    }
    template<typename T> Gpu::MatFunc<T> operator*(Gpu::MatFunc<T> A, Gpu::MatFunc<T> B) {
        return Gpu::matfunc_matmul<T>(A, B);
    }
    template<typename T> Gpu::MatFunc<T> operator*(T alpha, Gpu::MatFunc<T> A) {
        return Gpu::matfunc_scalar<T>(alpha, A, (T)0);
    }
    template<typename T> Gpu::MatFunc<T> operator*(Gpu::MatFunc<T> A, T alpha) {
        return Gpu::matfunc_scalar<T>(alpha, A, (T)0);
    }
    template<typename T> Gpu::MatFunc<T> operator/(Gpu::MatFunc<T> A, T alpha) {
        return Gpu::matfunc_scalar<T>((T)1/(T)alpha, A, (T)0);
    }
    template<typename T> Gpu::MatFunc<T> operator+(T alpha, Gpu::MatFunc<T> A) {
        return Gpu::matfunc_scalar<T>((T)1, A, alpha);
    }
    template<typename T> Gpu::MatFunc<T> operator+(Gpu::MatFunc<T> A, T alpha) {
        return Gpu::matfunc_scalar<T>((T)1, A, alpha);
    }
    template<typename T> Gpu::MatFunc<T> operator-(T alpha, Gpu::MatFunc<T> A) {
        return Gpu::matfunc_scalar<T>((T)-1, A, alpha);
    }
    template<typename T> Gpu::MatFunc<T> operator-(Gpu::MatFunc<T> A, T alpha) {
        return Gpu::matfunc_scalar<T>((T)1, A, -alpha);
    }
    template<typename T> Gpu::MatFunc<T> operator^(Gpu::MatFunc<T> A, T p) {
        return Gpu::matfunc_power<T>(A, p);
    }
    template<typename T> Gpu::MatFunc<T> operator^(Gpu::MatFunc<T> A, int p) {
        return Gpu::matfunc_power<T>(A, (T)p);
    }
    template<typename T> Gpu::MatFunc<T> operator^(Gpu::MatFunc<T> A, Gpu::MatFunc<T> B)
    {
        return Gpu::matfunc_matprod<T>(A, 'n', B, 'n');
    }
    template<typename T> Gpu::MatFunc<T> sigmoidmf(Gpu::MatFunc<T> A) {
        return Gpu::matfunc_sigmoid(A);
    }
    template<typename T> Gpu::MatFunc<T> tanhmf(Gpu::MatFunc<T> A) {
        return Gpu::matfunc_tanh(A);
    }
    template<typename T> Gpu::MatFunc<T> expmf(Gpu::MatFunc<T> A) {
        return Gpu::matfunc_exp(A);
    }
    template<typename T> Gpu::MatFunc<T> softmaxmf(Gpu::MatFunc<T> A) {
        int hauteur = A->get_height();
        Gpu::MatFunc<T> tmp1 = expmf(A);
        Gpu::MatFunc<T> tmp2 = newmf<T>(hauteur, hauteur, 1) ^ tmp1;
        return tmp1 * (tmp2 ^ (-1));

    }
    template<typename T> Gpu::MatFunc<T> normalizemf(Gpu::MatFunc<T> A, char type = 'l') {
        
        assert(type == 'l' || type == 'c');

        if (type == 'l') {

            int largeur = A->get_width();

            auto thick1 = newmf<T>(largeur, 1, 1);
            auto thick2 = newmf<T>(1, largeur, (T)1 / (T)largeur);

            auto mean = (A ^ thick1) ^ thick2;
            auto corrected = A - mean;
            auto var = ((corrected ^ 2) ^ thick1) ^ thick2;
            return corrected * ((var + (T)EPSILON) ^ ((T)-0.5));

        }
        else {

            int hauteur = A->get_height();

            auto thick1 = newmf<T>(1, hauteur, 1);
            auto thick2 = newmf<T>(hauteur, 1, (T)1 / (T)largeur);

            auto mean = thick2 ^ (thick1 ^ A);
            auto corrected = A - mean;
            auto var = thick2 ^ (thick1 ^ (corrected ^ 2));
            return corrected * ((var + (T)EPSILON) ^ ((T)-0.5));

        }


        


    }
    template<typename T> Gpu::MatFunc<T> msemf(Gpu::MatFunc<T> A, Gpu::MatFunc<T> B) {
        Gpu::MatFunc<T> tmp = (A - B) ^ 2;
        int n = tmp->get_width();
        return (tmp ^ newmf<T>(n, 1, 1)) / (T)n;
    }
    template<typename T> Gpu::MatFunc<T> concatmf(Gpu::MatFunc<T> A, Gpu::MatFunc<T> B, char type) {
        return matfunc_concat(A, B, type);
    }

    // The classes definiions 

    template<typename T> class ObjMatFunc {

    private :

        Gpu::Matrix<T> matrix_;
        std::vector<MatFunc<T>> childs_;
        std::vector<std::weak_ptr<ObjMatFunc<T>>> parents_;

        // Optimisation
        bool has_to_own_matrix_{ false };

    protected :

        // Optimisation
        // Return a matrix of the childs that can be shared with the current matfunc
        // If all childs cannot it return a new matrix of the size of the first child
        /*Matrix<T> __shared_matrix__(MatFuncArr<T> childs) {
            for (int i = 0; i < childs.size(); i++)
            {
                if (childs[i]->can_be_shared()) {
                    return childs[i]->matrix();
                }
            }
            Matrix<T> res(childs[0]->get_height(), childs[0]->get_width());
            return res;
        }*/

    public :

        /// Constructors

        // Void constructor : Creates a null ObjMatFunc
        ObjMatFunc()
            : matrix_(0, 0) /* TODO : mettre (1,1) si ne fonctionne pas */
        {}
        // Constructor of a new ObjMatFunc of size height*width
        ObjMatFunc(int height, int width)
            : matrix_(height, width)
        {}
        //Constructor from a Gpu::Matrix<T>
        ObjMatFunc(Gpu::Matrix<T> matrix) : matrix_(matrix)
        {}
        // copy constructor
        ObjMatFunc(const ObjMatFunc<T>& other)
            : matrix_(other.matrix()), childs_(other.childs()), parents_(other.parents())
        {}


        /// Getters 

        Gpu::Matrix<T> matrix() const { return matrix_; }
        int get_height() { return matrix_.height(); }
        int get_width() { return matrix_.width(); }

        int nb_childs() {
            return childs_.size();
        }
        std::vector<MatFunc<T>> childs() const { return childs_; }
        MatFunc<T> get_child(int i) { return childs_[i]; }
        std::weak_ptr<ObjMatFunc<T>> get_weak_child(int i) {
            std::weak_ptr<ObjMatFunc<T>> wp(childs_[i]);
            return wp;
        }

        int nb_parents() {
            return parents_.size();
        }
        std::vector<std::weak_ptr<ObjMatFunc<T>>> parents() const { return parents_; }
        MatFunc<T> get_parent(int i) { return parents_[i].lock(); }
        std::weak_ptr<ObjMatFunc<T>> get_weak_parent(int i) { return parents_[i]; }


        /// Virtual main methods

        virtual void compute() {
            /* Dependent on the inherited class */
            // When called, the inherited class object compute in 'matrix_' the result of the
            // matrix operation it represent with the matrices the MatFunc in 'childs_'
        }
        virtual MatFunc<T> childgrad(MatFunc<T> child, MatFunc<T> grad, MatFunc<T> matfunc_this) {
            /* Dependent on the inherited class */
            // This function assumes that grad represent a the matrix function dY/d(this) ie the gradient of a 1*1 MatFunc (Y) seen as a function of the calling MatFunc (this)
            // If child belongs to childs_ it then returns a MatFunc representing dY/d(child) using reverse auto differentiation rules
            return std::make_shared<ObjMatFunc<T>>();
        }


        /// Optimisation

        // Constructor from the childs
        // If a child is able to shared its matrix share the matrix with it
        // In order to be called the sharing of matrix between the child and the matfunc
        // must be possible in term of dimension
        ObjMatFunc(MatFuncArr<T> childs) : matrix_(0, 0) /* TODO : mettre (1,1) si ne fonctionne pas */ {
            bool done = false;
            for (int i = 0; i < childs.size() && !done; i++)
            {
                if (childs[i]->can_be_shared() && childs[i]->nb_parents() == 0) {
                    matrix_ = childs[i]->matrix();
                    done = true;
                }
            }
            if (!done) {
                Matrix<T> res(childs[0]->get_height(), childs[0]->get_width());
                matrix_ = res;
            }
        }

        // virtual destructor
        virtual ~ObjMatFunc() {
            for (int i = 0; i < nb_childs(); i++) {
                get_child(i)->__del_parent__(this);
            }

        };

        // Return true if the matrix of the matfunc can be shared
        bool can_be_shared() {
            return (!is_owner()) && (nb_parents() < 2);
        }

        // Reset the matrix and the memory of the parent if necessary in
        // order to be the unique owner of its matrix
        void set_owner() {

            has_to_own_matrix_ = true;
            Matrix<T> new_matrix(matrix_.height(), matrix_.width());
            new_matrix.copy(matrix_);
            matrix_ = new_matrix;

            if (nb_parents() == 1) {
                auto p = get_parent(0);
                if (p) {
                    p->__reset_matrix__();
                }
            }

        }

        // Return true if the matfunc has to own is own matrix
        // and so can't share it
        bool is_owner()
        {
            return has_to_own_matrix_;
        }

        // Reset the matrix of the matfunc, sharing the new one with
        // a child if possible
        // It is virtual because matfunc such as matprod hasn't the same dimension 
        // of their child and so can't share matrix with them and need to have a different
        // __reset_matrix__ function
        virtual void __reset_matrix__() {

            bool done = false;
            for (int i = 0; i < nb_childs() && !done; i++)
            {
                if (get_child(i)->can_be_shared()) {
                    matrix_ = get_child(i)->matrix();
                    done = true;
                }
            }
            if (!done) {
                Matrix<T> res(get_child(0)->get_height(), get_child(0)->get_width());
                matrix_ = res;
            }


            if (nb_parents() == 1) {
                auto p = get_parent(0);
                if (p) {
                    p->__reset_matrix__();
                }
            }

        }

        // graph managing functions 
        void __add_child__(MatFunc<T> A) {   
            childs_.push_back(A);
        }
        void __add_parent__(MatFunc<T> A) {

            std::weak_ptr<ObjMatFunc<T>> wp(A);
            parents_.push_back(wp);

            // Optimisation
            if (nb_parents() == 2) {
                auto p = get_parent(0);
                if (p) {
                    p->__reset_matrix__();
                }
            }

        }
        void __del_parent__(ObjMatFunc<T>* pA) {

            // Clean the parents_ array
            std::vector<std::weak_ptr<ObjMatFunc<T>>> new_parent;
            for (int i = 0; i < parents_.size(); i++)
            {
                auto p = get_parent(i);
                if (p && p.get() != pA) {
                    new_parent.push_back(get_weak_parent(i));
                }
            }
            parents_ = new_parent;

            // If sharing matrix is now possible reset_matrix of the parent
            if (nb_parents() == 1) {
                auto p = get_parent(0);
                if (p) {
                    p->__reset_matrix__();
                }
            }

        }



    };





    template<typename T> class  ObjMatFuncCopy : public ObjMatFunc<T> {
    private: std::weak_ptr<ObjMatFunc> other_;
    public:
        ObjMatFuncCopy(int height, int width) : ObjMatFunc<T>(height, width)
        {}
        virtual void compute() {
            auto p = other_.lock();
            assert(p);
            matrix().copy(p->matrix());
        }
        void choose_copy(MatFunc<T> other) {
            other_ = other;
            other->set_owner();
        }

    };






    template<typename T> class  ObjMatFuncScalar : public ObjMatFunc<T> {
    private: T alpha_, beta_;
    public:
        ObjMatFuncScalar(T alpha, MatFunc<T> A, T beta) : ObjMatFunc<T>({A}), alpha_(alpha), beta_(beta)
        {
            __add_child__(A);
        }
        virtual void compute() {
            matrix().Scalar(alpha_, get_child(0)->matrix(), beta_);
        }
        virtual MatFunc<T> childgrad(MatFunc<T> child, MatFunc<T> grad, MatFunc<T> matfunc_this) {

            if (child.get() == get_child(0).get()) {
                if (alpha_ == (T)0) {
                    return matfunc<T>(matrix().height(), matrix().width(), (T)0);
                }
                else if (alpha_ == (T)1) {
                    return grad;
                }
                else {
                    return matfunc_scalar<T>(alpha_, grad, (T)0);
                }
            }

            assert(0);
            return matfunc_null<T>();

        }
    };






    template<typename T> class  ObjMatFuncLinear : public ObjMatFunc<T> {
    private: T alpha_, beta_;
    public:
        ObjMatFuncLinear(T alpha, MatFunc<T> A, T beta, MatFunc<T> B) : ObjMatFunc<T>({A, B}), alpha_(alpha), beta_(beta)
        {
            assert(A->matrix().same_dimension(B->matrix()));
            __add_child__(A);
            __add_child__(B);
        }
        virtual void compute() {
            matrix().Linear(alpha_, get_child(0)->matrix(), beta_, get_child(1)->matrix());
        }
        virtual MatFunc<T> childgrad(MatFunc<T> child, MatFunc<T> grad, MatFunc<T> matfunc_this) {

            if (child.get() == get_child(0).get()) {
                if (alpha_ == (T)0) {
                    return matfunc<T>(matrix().height(), matrix().width(), (T)0);
                }
                else if (alpha_ == (T)1) {
                    return grad;
                }
                else {
                    return matfunc_scalar<T>(alpha_, grad, (T)0);
                }
            }
            else if (child.get() == get_child(1).get()) {
                if (beta_ == (T)0) {
                    return matfunc<T>(matrix().height(), matrix().width(), (T)0);
                }
                else if (beta_ == (T)1) {
                    return grad;
                }
                else {
                    return matfunc_scalar<T>(beta_, grad, (T)0);
                }
            }

            assert(0);
            return matfunc_null<T>();

        }
    };






    template<typename T> class  ObjMatFuncSum : public ObjMatFunc<T> {
    public:
        ObjMatFuncSum(std::vector<MatFunc<T>> As) : ObjMatFunc<T>(As)
        {
            for (int i = 0; i < As.size(); i++)
            {
                assert(matrix().same_dimension(As[i]->matrix()));
                __add_child__(As[i]);
            }
        }
        virtual void compute() {
            std::vector<Gpu::Matrix<T>> As;
            for (int i = 0; i < nb_childs(); i++)
            {
                As.push_back(get_child(i)->matrix());
            }
            matrix().Sum(As);
        }
        virtual MatFunc<T> childgrad(MatFunc<T> child, MatFunc<T> grad, MatFunc<T> matfunc_this) {
            for (int i = 0; i < nb_childs(); i++)
            {
                if (child.get() == get_child(i).get()) {
                    return grad;
                }
            }

            assert(0);
            return matfunc_null<T>();

        }
    };






    template<typename T> class  ObjMatFuncMatMul : public ObjMatFunc<T> {
    public:
        ObjMatFuncMatMul(MatFunc<T> A, MatFunc<T> B) : ObjMatFunc<T>({ A, B })
        {
            assert(A->matrix().same_dimension(B->matrix()));
            __add_child__(A);
            __add_child__(B);
        }
        virtual void compute() {
            matrix().Matmul(get_child(0)->matrix(), get_child(1)->matrix());
        }
        virtual MatFunc<T> childgrad(MatFunc<T> child, MatFunc<T> grad, MatFunc<T> matfunc_this) {

            if (child.get() == get_child(0).get()) {
                return matfunc_matmul(get_child(1), grad);
            }
            else if (child.get() == get_child(1).get()) {
                return matfunc_matmul(get_child(0), grad);
            }

            assert(0);
            return matfunc_null<T>();

        }

    };






    template<typename T> class  ObjMatFuncMatProd : public ObjMatFunc<T> {
    private: char transa_, transb_;
    public:
        ObjMatFuncMatProd(MatFunc<T> A, char transa, MatFunc<T> B, char transb) : ObjMatFunc<T>(transa == 'n' ? A->get_height() : A->get_width(), transb == 'n' ? B->get_width() : B->get_height()), transa_(transa), transb_(transb)
        {
            assert(transa == 't' || transa == 'n');
            assert(transb == 't' || transb == 'n');
            __add_child__(A);
            __add_child__(B);
        }
        virtual void compute() {
            matrix().Matprod(get_child(0)->matrix(), transa_, get_child(1)->matrix(), transb_, (T)1, (T)0);
        }
        virtual MatFunc<T> childgrad(MatFunc<T> child, MatFunc<T> grad, MatFunc<T> matfunc_this) {


            if (child.get() == get_child(0).get()) {
                if (transa_ == 'n' && transb_ == 'n') {
                    return matfunc_matprod<T>(grad, 'n', get_child(1), 't');
                }
                else if (transa_ == 't' && transb_ == 'n') {
                    return matfunc_matprod<T>(get_child(1), 'n', grad, 't');
                }
                else if (transa_ == 'n' && transb_ == 't') {
                    return matfunc_matprod<T>(grad, 'n', get_child(1), 'n');
                }
                else {
                    return matfunc_matprod<T>(get_child(1), 't', grad, 't');
                }
            }
            else if (child.get() == get_child(1).get()) {
                if (transa_ == 'n' && transb_ == 'n') {
                    return matfunc_matprod<T>(get_child(0), 't', grad, 'n');
                }
                else if (transa_ == 't' && transb_ == 'n') {
                    return matfunc_matprod<T>(get_child(0), 'n', grad, 'n');
                }
                else if (transa_ == 'n' && transb_ == 't') {
                    return matfunc_matprod<T>(grad, 't', get_child(0), 'n');
                }
                else {
                    return matfunc_matprod<T>(grad, 't', get_child(0), 't');
                }
            }

            assert(0);
            return matfunc_null<T>();
        }
        virtual void __reset_matrix__() {}
    };






    template<typename T> class  ObjMatFuncPower : public ObjMatFunc<T> {
    private: T p_;
    public:
        ObjMatFuncPower(MatFunc<T> A, T p) : ObjMatFunc<T>({ A }), p_(p)
        {
            __add_child__(A);
        }
        virtual void compute() {
            matrix().Power(get_child(0)->matrix(), p_);
        }
        virtual MatFunc<T> childgrad(MatFunc<T> child, MatFunc<T> grad, MatFunc<T> matfunc_this) {

            if (child.get() == get_child(0).get()) {
                if (p_ == (T)0) {
                    return matfunc<T>(child->get_height(), child->get_width(), (T)1);
                }
                if (p_ == (T)1) {
                    return grad;
                }
                else {
                    return matfunc_scalar<T>(p_, matfunc_matmul<T>(grad, matfunc_power<T>(get_child(0), p_ - (T)1)), (T)0);
                }
            }

            assert(0);
            return matfunc_null<T>();
        }
    };






    template<typename T> class  ObjMatFuncSigmoid : public ObjMatFunc<T> {
    public:
        ObjMatFuncSigmoid(MatFunc<T> A) : ObjMatFunc<T>({ A })
        {
            __add_child__(A);
        }
        virtual void compute() {
            matrix().Sigmoid(get_child(0)->matrix());
        }
        virtual MatFunc<T> childgrad(MatFunc<T> child, MatFunc<T> grad, MatFunc<T> matfunc_this) {

            if (child.get() == get_child(0).get()) {
                return matfunc_matmul<T>(grad, matfunc_matmul<T>(matfunc_this, matfunc_scalar<T>(-1, matfunc_this, 1)));
            }

            assert(0);
            return matfunc_null<T>();
        }
    };






    template<typename T> class  ObjMatFuncTanh : public ObjMatFunc<T> {
    public:
        ObjMatFuncTanh(MatFunc<T> A) : ObjMatFunc<T>({ A })
        {
            __add_child__(A);
        }
        virtual void compute() {
            matrix().Tanh(get_child(0)->matrix());
        }
        virtual MatFunc<T> childgrad(MatFunc<T> child, MatFunc<T> grad, MatFunc<T> matfunc_this) {

            if (child.get() == get_child(0).get()) {
                return matfunc_matmul<T>(grad, matfunc_scalar<T>(-1, matfunc_power<T>(matfunc_this, 2), 1));
            }

            assert(0);
            return matfunc_null<T>();
        }
    };






    template<typename T> class  ObjMatFuncExp : public ObjMatFunc<T> {
    public:
        ObjMatFuncExp(MatFunc<T> A) : ObjMatFunc<T>({ A })
        {
            __add_child__(A);
        }
        virtual void compute() {
            matrix().Exp(get_child(0)->matrix());
        }
        virtual MatFunc<T> childgrad(MatFunc<T> child, MatFunc<T> grad, MatFunc<T> matfunc_this) {

            if (child.get() == get_child(0).get()) {
                return matfunc_matmul<T>(grad, matfunc_this);
            }

            assert(0);
            return matfunc_null<T>();
        }
    };





    template<typename T> class  ObjMatFuncExtract : public ObjMatFunc<T> {
    protected: std::shared_ptr<int> i0_, j0_;
    public:
        ObjMatFuncExtract(MatFunc<T> A, std::shared_ptr<int> i0, std::shared_ptr<int> j0, int height, int width) : ObjMatFunc<T>(height, width)
        {
            __add_child__(A);
            i0_ = i0;
            j0_ = j0;
        }
        virtual void compute() {
            matrix().extract(get_child(0)->matrix(), *i0_, *j0_);
        }
        virtual MatFunc<T> childgrad(MatFunc<T> child, MatFunc<T> grad, MatFunc<T> matfunc_this) {

            if (child.get() == get_child(0).get()) {
                return matfunc_rev_extract<T>(grad, i0_, j0_, get_child(0)->get_height(), get_child(0)->get_width());
            }

            assert(0);
            return matfunc_null<T>();
        }
        virtual void __reset_matrix__() {}
    };







    // TODO : Derniers tests
    template<typename T> class  ObjMatFuncReverseExtract : public ObjMatFunc<T> {
    protected : std::shared_ptr<int> i0_, j0_;
    public:
        ObjMatFuncReverseExtract(MatFunc<T> A, std::shared_ptr<int> i0, std::shared_ptr<int> j0, int height, int width) : ObjMatFunc<T>(height, width)
        {
            __add_child__(A);
            i0_ = i0;
            j0_ = j0;
        }
        virtual void compute() {
            matrix().fill((T)0); // Peut-être à déplacer dans le compute
            matrix().rev_extract(get_child(0)->matrix(), *i0_, *j0_);
        }
        virtual MatFunc<T> childgrad(MatFunc<T> child, MatFunc<T> grad, MatFunc<T> matfunc_this) {

            if (child.get() == get_child(0).get()) {
                return matfunc_extract<T>(grad, i0_, j0_, get_child(0)->get_height(), get_child(0)->get_width());
            }

            assert(0);
            return matfunc_null<T>();
        }
        virtual void __reset_matrix__() {}
    };



    template<typename T> class  ObjMatFuncConcat : public ObjMatFunc<T> {
    protected: int i1_, j1_;
    public:
        ObjMatFuncConcat(MatFunc<T> A, MatFunc<T> B, char arg) : ObjMatFunc<T>(arg == 'c' ? A->get_height() + B->get_height() : A->get_height(), arg == 'c' ? A->get_width() : A->get_width() + B->get_width())
        {
            assert(arg == 'l' || arg == 'c');
            __add_child__(A);
            __add_child__(B);
            if (arg == 'l') {
                assert(A->get_height() == B->get_height());
                i1_ = 0;
                j1_ = get_child(0)->get_width();
            }
            else {
                assert(A->get_width() == B->get_width());
                i1_ = get_child(0)->get_height();
                j1_ = 0;
            }
        }
        virtual void compute() {
            matrix().rev_extract(get_child(0)->matrix(), 0, 0);
            matrix().rev_extract(get_child(1)->matrix(), i1_, j1_);
        }
        virtual MatFunc<T> childgrad(MatFunc<T> child, MatFunc<T> grad, MatFunc<T> matfunc_this) {

            if (child.get() == get_child(0).get()) {
                return matfunc_extract<T>(grad, 0, 0, get_child(0)->get_height(), get_child(0)->get_width());
            }
            else if (child.get() == get_child(1).get()) {
                return matfunc_extract<T>(grad, i1_, j1_, get_child(1)->get_height(), get_child(1)->get_width());
            }

            assert(0);
            return matfunc_null<T>();
        }
        virtual void __reset_matrix__() {}
    };




    // TODO : A vérifier
    template<typename T> class  ObjMatFuncBatchConcat : public ObjMatFunc<T> {
    private:
        int __sum_heights__(std::vector<MatFunc<T>> As) {
            int h = 0;
            for (int i = 0; i < As.size(); i++)
            {
                h += As[i]->get_height();
            }
            return h;
        }
        int __sum_widths__(std::vector<MatFunc<T>> As) {
            int h = 0;
            for (int i = 0; i < As.size(); i++)
            {
                h += As[i]->get_width();
            }
            return h;
        }
    protected: std::vector<int> idx_tab_;
    public:
        ObjMatFuncBatchConcat(std::vector<MatFunc<T>> As, char arg) : ObjMatFunc<T>(arg == 'c' ? __sum_heights__(As) : As[0]->get_height(), arg == 'c' ? As[0]->get_width() : __sum_widths__(As))
        {
            assert(arg == 'l' || arg == 'c');
            int count = 0;
            for (int i = 0; i < As.size(); i++)
            {
                idx_tab_.push_back(count);
                __add_child__(As[i]);
                if (arg == 'l') {
                    assert(As[0]->get_height() == As[i]->get_height());
                    count += As[i]->get_width();
                }
                else {
                    assert(As[0]->get_width() == As[i]->get_width());
                    count += As[i]->get_height();
                }
            }
        }
        virtual void compute() {
            if (arg == 'l') {
                for (int i = 0; i < nb_childs(); i++) { matrix().rev_extract(get_child(i)->matrix(), 0, idx_tab_[i]); }
            }
            else {
                for (int i = 0; i < nb_childs(); i++) { matrix().rev_extract(get_child(i)->matrix(), idx_tab_[i], 0); }
            }
        }
        virtual MatFunc<T> childgrad(MatFunc<T> child, MatFunc<T> grad, MatFunc<T> matfunc_this) {

            for (int i = 0; i < nb_childs(); i++)
            {
                if (child.get() == get_child(i).get()) {
                    if (arg == 'l') {
                        return matfunc_extract<T>(grad, 0, idx_tab_[i], get_child(i)->get_height(), get_child(i)->get_width());
                    }
                    else {
                        return matfunc_extract<T>(grad, idx_tab_[i], 0, get_child(i)->get_height(), get_child(i)->get_width());
                    }
                }
            }

            assert(0);
            return matfunc_null<T>();
        }
        virtual void __reset_matrix__() {}
    };





////////////////////////////////////////////////////////////////////////////////////////////
// IMPLEMENTATION DIRECTE DES NOEUDS TYPIQUES DE RESEAUX DE NEURONES POUR PLUS D'EFFICACITE
////////////////////////////////////////////////////////////////////////////////////////////

    template<typename T> class  ObjMatFuncNeuralNetworkDense : public ObjMatFunc<T> {
    public:
        ObjMatFuncNeuralNetworkDense(MatFunc<T> Input, MatFunc<T> Weights, MatFunc<T> Bias) : ObjMatFunc<T>(Weights->get_height(), Input->get_width())
        {
            assert(A->matrix().same_dimension(B->matrix())); 
            assert(Weights->get_width() == Input.get_height());
            assert(Bias->get_width() == 1 && Bias->get_height() == Weights.get_height());
            __add_child__(Input);
            __add_child__(Weights);
            __add_child__(Bias);
            __add_child__(newmf<T>(1, Input->get_width(), (T)1));
        }
        virtual void compute() {
            matrix().Matprod(get_child(1)->matrix(), 'n', get_child(0)->matrix(), 'n');
            matrix().Matprod(get_child(2)->matrix(), 'n', get_child(3)->matrix(), 'n', (T)1, (T)1);
        }
        virtual MatFunc<T> childgrad(MatFunc<T> child, MatFunc<T> grad, MatFunc<T> matfunc_this) {

            if (child.get() == get_child(0).get()) {
                return matfunc_matprod<T>(get_child(1), 't', grad, 'n');
            }
            else if (child.get() == get_child(1).get()) {
                return matfunc_matprod<T>(grad, 'n', get_child(0), 't');
            }
            else if (child.get() == get_child(2).get()) {
                return matfunc_matprod<T>(grad, 'n', get_child(3), 't');
            }

            assert(0);
            return matfunc_null<T>();
        }
        virtual void __reset_matrix__() {}
    };






    // The mains functions to use MatFuncs

    template<typename T> MatFuncSet<T> compute(MatFunc<T> mf, MatFuncSet<T>& computed) {

        // Effectue un parcour en profondeur à partir de mf afin de la calculer

        // Computed associe à une MatFunc un état :
        //  - Pas présent = pas été visité
        //  - 1 = A "compute" cad à calcul
        //  - 0 = "Computed" cad calculé

        std::stack<MatFunc<T>> call_stack;
        call_stack.push(mf);

        while (!call_stack.empty())
        {

            MatFunc<T> current = call_stack.top();
            auto it = computed.find(current);

            // On regarde l'état de current dans computed

            if (it == computed.end()) { // Il n'a pas encore été visité

                computed[current] = 1; // On le met à calculer

                // On ajoute ses enfants par dessus dans la call stack pour les calculer avant
                auto childs = current->childs();
                for (int i = 0; i < childs.size(); i++) {
                    call_stack.push(childs[i]);
                }

            }
            else if (it->second == 1) { // Il est à calculer
                current->compute(); // On le calcule
                it->second = 0; // On le met comme calculé
                call_stack.pop(); // On l'enlève de la call stack
            }
            else { // Il est calculer
                call_stack.pop(); // On l'enlève de la call stack
            }

        }

        // On test les possibles erreurs survenue sur le gpu
        __gpu_test__();

        return computed;

    }
    template<typename T> MatFuncSet<T> compute(MatFunc<T> mf) {
        MatFuncSet<T> computed;
        compute<T>(mf, computed);
        return computed;
    }
    template<typename T> MatFuncSet<T> compute(MatFuncIndex<T> mfs) {
        MatFuncSet<T> computed;
        for (auto it = mfs.begin(); it != mfs.end(); it++)
        {
            compute<T>(it->second, computed);
        }
        return computed;
    }

    template<typename T> MatFunc<T> gradient(MatFunc<T> mf, MatFuncMap<T>& map_grads) {

        // Préconditions :
        // 
        // - mf est une matfunc
        // - Il existe Y une matfonc de taille 1*1 telle que pour tout matfunc X, ou bien X n'est pas dans map_grads ou bien map_grads[X] = dY/dX
        // - map_grads[Y] = matfunc(1, 1, 1);
        // 
        // 
        // Description :
        // 
        // Créer dans map_grads les matfunc nécessaires au calcul de dY/dmf avec Y et mf tels que décris dans les préconditions
        // Le but étant d'obtenir le gradient par accumulation arrière. Puis renvoie dY/dmf.
        // 
        // ( Si dictgradients_ est vide la fonction retourne nullptr )
        //

        // Principe :
        //
        // Le principe de base est faire un parcour en profondeur vers les parents de mf jusqu'à trouver une matfunc X dans map_grads.
        // Puis on redenscend jusqu'à mf en créant des matfunc de gradient par accumulation arrière de proche en proche afin d'optenir
        // la matfonc de dY/dmf à la fin
        //

        // On créé une pile d'appels avec mf dedans
        std::stack<MatFunc<T>> call_stack;
        call_stack.push(mf);


        // Tant que la pile n'est pas vide
        while (!call_stack.empty())
        {

            // On traite le matrice en haut de la pile
            MatFunc<T> current = call_stack.top();
            // On regarde s'il est présent dans dictgradients_
            auto it = map_grads.find(current);

            // S'il n'est pas présent
            if (it == map_grads.end()) {

                // On le met dans dictgradients_ mais avec dY/dmatrice = nullptr pour signifier qu'il faut encore le calculer
                map_grads[current] = matfunc_null<T>();

                // On ajoute ses parents à la pile des appels pour les traiter
                auto parents = current->parents();
                for (int i = 0; i < parents.size(); i++) {
                    auto parent = parents[i].lock();
                    if (parent) {
                        call_stack.push(parent);
                    }
                }

            }

            // Sinon s'il est présent mais que dY/dmatrice = nullptr
            else if (is_matfunc_null<T>(it->second)) {

                // Alors on connais le gradient pour tout ses parents
                // On récupére ces gradients
                std::vector<MatFunc<T>> mf_arr;

                auto parents = current->parents();

                for (int i = 0; i < parents.size(); i++) {
                    // Si le parent n'a pas de gradient c'est que Y ne dépend pas de ce parent et donc
                    // matrice ne recois aucun gradient de la part de ce parent
                    // Sinon on récupére le gradient qu'associe le parent à matrice
                    auto parent = parents[i].lock();
                    if (parent) {
                        auto grad = map_grads[parent];
                        if (!is_matfunc_null(grad)) {
                            mf_arr.push_back(parent->childgrad(current, grad, parent));
                        }
                    }
                }

                // On somme
                if (mf_arr.size() == 1) {
                    it->second = mf_arr[0];
                }
                else if (mf_arr.size() > 0) {
                    it->second = matfunc_sum<T>(mf_arr);
                }

                call_stack.pop();

            }

            else {
                call_stack.pop();
            }

        }

        Gpu::__gpu_test__();

        return map_grads[mf];

    }
    template<typename T> MatFuncMap<T> gradient(MatFunc<T> Y, std::vector<MatFunc<T>> Xs) {

        // Description : Calcul le gradient de t par rapport aux matrices de E
        // 
        // Précontions :
        // - y un matrice scalaire quelconque du graphe
        //  (si y n'est pas scalaire alors c'est le gradient de la somme de ses composantes qui est calculé)
        // - E un ensemble de matrices du graphe
        //

        MatFuncMap<T> D;
        D[Y] = matfunc(Y->get_height(), Y->get_width(), (T)1);

        // Dans le cas ou les préconditions sont vérifiés la dimension est 1*1
        // Sinon c'est une matrice remplie de 1

        for (int i = 0; i < Xs.size(); i++)
        {
            gradient<T>(Xs[i], D);
        }

        return D;
    }
    template<typename T> MatFunc<T> gradient(MatFunc<T> Y, MatFunc<T> X) {
        MatFuncMap<T> D;
        D[Y] = matfunc(Y->get_height(), Y->get_width(), (T)1);
        return gradient<T>(X, D);
    }


}