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
#include "utilities.cuh"
#include "matrix.cuh"
#include "matfunc.cuh"



// DEFINES /////////////////////////////////////////////////////////////////////////////////////////////////////

#define EPSILON 1e-8


// NEURALNETWORK ///////////////////////////////////////////////////////////////////////////////////////////////

namespace Gpu {

    // TYPENAMES ///////////////////////////////////////////////////////////////////////////////////////////////

    template<typename T> using HyperParameters = std::map<std::string, T>;



    // OPTIMIZERS //////////////////////////////////////////////////////////////////////////////////////////////

    template<typename T> class Optimizer {

    protected:

        MatFuncArr<T> inputs_;
        MatFuncArr<T> gradients_;

        MatFunc<T> output_;

        void init() {
            auto map_gradients = gradient<T>(output_, inputs_);
            for (int i = 0; i < inputs_.size(); i++)
            {
                gradients_.push_back(map_gradients[inputs_[i]]);
            }
        }

    public:

        Optimizer(MatFunc<T> output, std::vector<MatFunc<T>> inputs)
            : output_(output), inputs_(inputs)
        {
            init();
        }
        Optimizer(MatFunc<T> output, MatFuncIndex<T> inputs)
            : output_(output)
        {
            for (auto it = inputs.begin(); it != inputs.end(); it++) {
                inputs_.push_back(it->second);
            }
            init();
        }

        void compute_gradients() {
            MatFuncSet<T> E;
            for (int i = 0; i != inputs_.size(); i++)
            {
                compute<T>(gradients_[i], E);
            }
            compute<T>(output_, E);
        }
        virtual void optimize() {}

    };

    template<typename T> class OptimizerGradientDescent : public Optimizer<T> {

    protected:

        T learning_rate_;

    public:

        OptimizerGradientDescent(MatFunc<T> output, std::vector<MatFunc<T>> inputs, T learning_rate)
            : Optimizer<T>(output, inputs), learning_rate_(learning_rate)
        {}
        OptimizerGradientDescent(MatFunc<T> output, MatFuncIndex<T> inputs, T learning_rate)
            : Optimizer<T>(output, inputs), learning_rate_(learning_rate)
        {}
        OptimizerGradientDescent(MatFunc<T> output, MatFuncIndex<T> inputs, std::map<std::string, T> hyper_parameters)
            : Optimizer<T>(output, inputs), learning_rate_(hyper_parameters["learning_rate"])
        {
            assert(hyper_parameters.find("learning_rate") != hyper_parameters.end());
        }

        virtual void optimize() {
            compute_gradients();
            for (int i = 0; i != inputs_.size(); i++)
            {
                Matrix<T> P = inputs_[i]->matrix();
                Matrix<T> G = gradients_[i]->matrix();
                P.Linear(1, P, -learning_rate_, G);
            }
        }

    };

    template<typename T> class OptimizerAdam : public Optimizer<T> {

    protected:

        T learning_rate_, decay1_, decay2_;
        std::vector<Matrix<T>> moments1_, moments2_;
        int t_;
        T decay1_t_, decay2_t_;

        void init() {
            for (int i = 0; i < inputs_.size(); i++)
            {
                Matrix<T> A1(inputs_[i]->get_height(), inputs_[i]->get_width());
                Matrix<T> A2(inputs_[i]->get_height(), inputs_[i]->get_width());
                A1.fill(0);
                A2.fill(0);
                moments1_.push_back(A1);
                moments2_.push_back(A2);
            }
        }

    public:

        OptimizerAdam(MatFunc<T> output, std::vector<MatFunc<T>> inputs, T learning_rate, T decay1, T decay2)
            : Optimizer<T>(output, inputs), learning_rate_(learning_rate), decay1_(decay1), decay2_(decay2)
        {
            init();
            reset_time();
        }
        OptimizerAdam(MatFunc<T> output, MatFuncIndex<T> inputs, T learning_rate, T decay1, T decay2)
            : Optimizer<T>(output, inputs), learning_rate_(learning_rate), decay1_(decay1), decay2_(decay2)
        {
            init();
            reset_time();
        }
        OptimizerAdam(MatFunc<T> output, MatFuncIndex<T> inputs, std::map<std::string, T> hyper_parameters)
            : Optimizer<T>(output, inputs), learning_rate_(hyper_parameters["learning_rate"]), decay1_(hyper_parameters["decay1"]), decay2_(hyper_parameters["decay2"])
        {
            assert(hyper_parameters.find("learning_rate") != hyper_parameters.end());
            assert(hyper_parameters.find("decay1") != hyper_parameters.end());
            assert(hyper_parameters.find("decay2") != hyper_parameters.end());
            init();
            reset_time();
        }

        void reset_time() {
            t_ = 0;
            decay1_t_ = (T)1;
            decay2_t_ = (T)1;
        }
        virtual void optimize() {

            t_++;
            decay1_t_ *= decay1_;
            decay2_t_ *= decay2_;

            compute_gradients();

            for (int i = 0; i < inputs_.size(); i++)
            {
                T* P = inputs_[i]->matrix().data().get();
                T* G = gradients_[i]->matrix().data().get();
                T* M = moments1_[i].data().get();
                T* V = moments2_[i].data().get();

                T alpha = learning_rate_;
                T beta1 = decay1_;
                T beta2 = decay2_;
                T beta1_t = decay1_t_;
                T beta2_t = decay2_t_;

                auto fptr = [=] __device__(int idx) {

                    T g = G[idx];

                    T m = beta1 * M[idx] + ((T)1 - beta1) * g;
                    T v = beta2 * V[idx] + ((T)1 - beta2) * g * g;
                    M[idx] = m;
                    V[idx] = v;

                    m = m / ((T)1 - beta1_t);
                    v = v / ((T)1 - beta2_t);

                    g = m / ((T)sqrt((double)v) + EPSILON);

                    P[idx] -= alpha * g;

                };


                Gpu::call_gpu_iter(inputs_[i]->matrix().size(), fptr);
                Gpu::__gpu_test__();

            }
        }

    };



    // NEURALNETWORK ///////////////////////////////////////////////////////////////////////////////////////////

    template<typename T> class NeuralNetwork {

    protected:

        /// Attributes 

        Gpu::MatFuncIndex<T> parameters_;
        Gpu::MatrixSave<T> save_;


        /// Basic methods of construction

        // Add a paramter matrix of named 'name' and of dimension
        // 'h'*'w'
        void add_param(std::string name, int h, int w) {
            MatFunc<T> param = newmf<T>(h, w, (T)0);
            parameters_[name] = param;
            save_.add_matrix(param->matrix(), name);
        }

        // Add the parameters necessary for the creation of a dense layer from 'input_size' to 'output_size'
        // the parameters are named : 'name' + "_" + dense_parameter_name("w" for weights and "b" for biais)
        void add_dense_params(std::string name, int input_size, int output_size) {
            add_param(name + "_w", output_size, input_size);
            add_param(name + "_b", output_size, 1);
        }
        // Create in network with the name 'name' a dense layer using parameters created before with 'add_dense_params(name, ...)' and the 'activation' function of activation taking 'input' as input
        void create_dense(MatFuncIndex<T>& network, std::string name, MatFunc<T> input, std::string activation = "id") {
            MatFunc<T> A = parameter(name + "_w") ^ input;
            MatFunc<T> B = parameter(name + "_b") ^ newmf<T>(1, input->get_width(), 1);
            if (activation == "id") {
                network[name] = A + B;
            }
            else if (activation == "sigmoid") {
                network[name] = sigmoidmf(A + B);
            }
            else if (activation == "tanh") {
                network[name] = tanhmf(A + B);
            }
            else {
                std::cout << "#!# Error : Invalid activation ! \n";
                assert(0);
            }

        }

        // Add the parameters necessary for the creation of a lstm layer from 'input_size' to 'output_size'
        // the parameters are named : 'name' + "_" + lstm_parameter_name(
        //      "wf_in", "wf_hs" and "bf" for the forget gate
        //      "wi_in", "wi_hs" and "bi" for the input gate
        //      "wc_in", "wc_hs" and "bc" for the state gate
        //      "wo_in", "wo_hs" and "bo" for the output gate gate)
        void add_lstm_params(std::string name, int input_size, int output_size) {

            add_param(name + "_wf_in", output_size, input_size);
            add_param(name + "_wf_hs", output_size, output_size);
            add_param(name + "_bf", output_size, 1);

            add_param(name + "_wi_in", output_size, input_size);
            add_param(name + "_wi_hs", output_size, output_size);
            add_param(name + "_bi", output_size, 1);

            add_param(name + "_wc_in", output_size, input_size);
            add_param(name + "_wc_hs", output_size, output_size);
            add_param(name + "_bc", output_size, 1);

            add_param(name + "_wo_in", output_size, input_size);
            add_param(name + "_wo_hs", output_size, output_size);
            add_param(name + "_bo", output_size, 1);

        }
        // Create in network with the name 'name' a dense layer using parameters created before with 'add_lstm_params(name, ...)'  taking 'input' as input
        void create_lstm(MatFuncIndex<T>& network, std::string name, MatFunc<T> input) {

            int input_size = parameter(name + "_wf_in")->get_width();
            int output_size = parameter(name + "_wf_in")->get_height();
            int sample_size = input->get_width();

            ObjMatFuncCopy<T>* CS = new ObjMatFuncCopy<T>(output_size, sample_size);
            ObjMatFuncCopy<T>* HS = new ObjMatFuncCopy<T>(output_size, sample_size);
            MatFunc<T> cs((ObjMatFunc<T>*)CS);
            MatFunc<T> hs((ObjMatFunc<T>*)HS);
            MatFunc<T> thick = newmf<T>(1, sample_size, (T)1);

            auto wf_in = parameter(name + "_wf_in");
            auto wf_hs = parameter(name + "_wf_hs");
            auto bf = parameter(name + "_bf");

            auto wi_in = parameter(name + "_wi_in");
            auto wi_hs = parameter(name + "_wi_hs");
            auto bi = parameter(name + "_bi");

            auto wc_in = parameter(name + "_wc_in");
            auto wc_hs = parameter(name + "_wc_hs");
            auto bc = parameter(name + "_bc");

            auto wo_in = parameter(name + "_wo_in");
            auto wo_hs = parameter(name + "_wo_hs");
            auto bo = parameter(name + "_bo");

            auto f_gate = sigmoidmf(matfunc_sum<T>({ wf_in ^ input,  wf_hs ^ hs, bf ^ thick }));
            auto i_gate = sigmoidmf(matfunc_sum<T>({ wi_in ^ input,  wi_hs ^ hs, bi ^ thick }));
            auto c_gate = tanhmf(matfunc_sum<T>({ wc_in ^ input,  wc_hs ^ hs, bc ^ thick }));
            auto o_gate = sigmoidmf(matfunc_sum<T>({ wo_in ^ input,  wo_hs ^ hs, bo ^ thick }));

            auto new_cs = (cs * f_gate) + (i_gate * c_gate);
            auto new_hs = (tanhmf(new_cs)) * o_gate;
            new_cs->matrix().fill(0);
            new_hs->matrix().fill(0);

            CS->choose_copy(new_cs);
            HS->choose_copy(new_hs);

            network[name + "_c"] = new_cs;
            network[name] = new_hs;

        }
        // Initialize the memory of the lstm layer 'name' in network to 0
        void init_lstm(MatFuncIndex<T>& network, std::string name) {
            network[name]->matrix().fill(0);
            network[name + "_c"]->matrix().fill(0);
        }

        // Same for GRU layer

        void add_gru_params(std::string name, int input_size, int output_size) {

            add_param(name + "_wf_in", output_size, input_size);
            add_param(name + "_wf_hs", output_size, output_size);
            add_param(name + "_bf", output_size, 1);

            add_param(name + "_wh_in", output_size, input_size);
            add_param(name + "_wh_hs", output_size, output_size);
            add_param(name + "_bh", output_size, 1);

        }
        void create_gru(MatFuncIndex<T>& network, std::string name, MatFunc<T> input) {

            int input_size = parameter(name + "_wf_in")->get_width();
            int output_size = parameter(name + "_wf_in")->get_height();
            int sample_size = input->get_width();

            ObjMatFuncCopy<T>* HS = new ObjMatFuncCopy<T>(output_size, sample_size);
            MatFunc<T> hs((ObjMatFunc<T>*)HS);
            MatFunc<T> thick = newmf<T>(1, sample_size, (T)1);

            auto wf_in = parameter(name + "_wf_in");
            auto wf_hs = parameter(name + "_wf_hs");
            auto bf = parameter(name + "_bf");

            auto wh_in = parameter(name + "_wh_in");
            auto wh_hs = parameter(name + "_wh_hs");
            auto bh = parameter(name + "_bh");

            auto f_gate = sigmoidmf(matfunc_sum<T>({ wf_in ^ input,  wf_hs ^ hs, bf ^ thick }));
            auto h_gate = sigmoidmf(matfunc_sum<T>({ wh_in ^ input,  wh_hs ^ hs, bh ^ thick }));

            auto new_hs = (((T)1 - f_gate) * hs) + (f_gate * h_gate);

            HS->choose_copy(new_hs);

            network[name] = new_hs;
        }
        void init_gru(MatFuncIndex<T>& network, std::string name) {
            network[name]->matrix().fill(0);
        }

        // Same for Scaling layer
        void add_scaling_params(std::string name, int input_size) {
            add_param(name, input_size, 1);
        }
        void create_scaling(MatFuncIndex<T>& network, std::string name, MatFunc<T> input) {
            network[name] = input * (parameter(name) ^ newmf<T>(1, input->get_width(), 1));
        }

        // Same for attention layer

        void add_attention_params(std::string name, int input_size, int dim, MatFunc<T> Mask = Gpu::nullmf<T>()) {
            add_param(name + "_wq", dim, input_size);
            add_param(name + "_wk", dim, input_size);
            add_param(name + "_wv", dim, input_size);
            parameters_[name + "_mask"] = Mask;
        }
        void create_attention(MatFuncIndex<T>& network, std::string name, MatFunc<T> Q, MatFunc<T> K, MatFunc<T> V) {

            int dim = parameter(name + "_wq")->get_height();
            int h = parameter(name + "_wq")->get_width();
            int w = Q->get_width();

            assert(Q->get_height() == h && Q->get_width() == w);
            assert(K->get_height() == h && K->get_width() == w);
            assert(V->get_height() == h && V->get_width() == w);

            T scale = (T)1 / (T)sqrt((double)dim);

            MatFunc<T> Q1 = parameter(name + "_wq") ^ Q;
            MatFunc<T> K1 = parameter(name + "_wk") ^ K;
            MatFunc<T> V1 = parameter(name + "_wv") ^ V;
            MatFunc<T> Mask = parameter(name + "_mask");

            if (isnullmf(Mask)) {
                network[name] = V1 ^ softmaxmf(scale * matfunc_matprod(Q1, 't', K1, 'n'));
            }
            else {
                assert(Mask->get_height() == w && Mask->get_width() == w);
                network[name] = V1 ^ softmaxmf((scale * matfunc_matprod(Q1, 't', K1, 'n')) + Mask );
            }


        }

        void add_multi_head_self_attention_params(std::string name, MatFunc<T> input, int input_size, int dim, int nb_head, MatFunc<T> Mask = Gpu::nullmf<T>()) {
            for (int i = 0; i < nb_head; i++)
            {
                add_attention_params(name + "_head_" + std::to_string(i), input_size, dim, Mask);
            }
        }
        void create_multi_head_self_attention(MatFuncIndex<T>& network, std::string name, MatFunc<T> input, int nb_head) {
            for (int i = 0; i < nb_head; i++)
            {
                std::string name_head = name + "_head_" + std::to_string(i);
                create_attention(network, name_head, input, input, input);
            }
            std::vector<MatFunc<T>> vec;
            for (int i = 0; i < nb_head; i++)
            {
                std::string name_head = name + "_head_" + std::to_string(i);
                vec.push_back(network[name_head]);
            }
            network[name] = matfunc_batch_concat<T>(vec, 'c');
        }

        MatFunc<T> mask_for_transofmer(int h) {

            MatFunc<T> Mask = newmf(h, h, (T)-INFINITY);

            T* tmp = new T[h];
            for (int i = 0; i < h; i++) { tmp[i] = 0; }

            T* dst = Mask->matrix().data().get();

            // On remplit colonne de 0 par colonne de 0
            for (int i = 0; i < h; i++)
            {
                cudaMemcpy((void*)&dst[i * h + i], (void*)tmp, (h - i) * sizeof(T), cudaMemcpyHostToDevice);
            }

            return Mask;

        }
        void add_transformer_decoder_layer_params(std::string name, int input_size, int dim, int nb_head, int output_size) {
            MatFunc<T> mask = mask_for_transofmer(dim);
            add_multi_head_self_attention_params(name + "_masked_multi_head_self_attention", input_size, dim, nb_head, mask);
            add_dense_params(name + "_dense_layer", dim * nb_head, output_size);
        }
        void add_transformer_decoder_layer(MatFuncIndex<T>& network, std::string name, MatFunc<T> input, int nb_head) {
            create_multi_head_self_attention(network, name + "_masked_multi_head_self_attention", input, nb_head);
            network[name + "add_and_norm_1"] = normalizemf(network[name + "_masked_multi_head_self_attention"] + input, 'l');
            create_dense(network, name + "_dense_layer", network[name + "add_and_norm_1"]);
        }

    public:

        NeuralNetwork() {}

        /// Getters

        MatFunc<T> parameter(std::string name) {
            return parameters_[name];
        }
        MatFuncIndex<T> parameters() {
            return parameters_;
        }

        /// Main methods

        // Return an instance of the network constructed using the paramaters set of the NeuralNetwork and 'input' as initial input
        virtual MatFuncIndex<T> create_network(MatFunc<T> input) {
            return MatFuncIndex<T>();
        }
        // Initialize the instance 'network' (for example the memory of lstm layer)
        virtual void init_network(MatFuncIndex<T>& network) {
        }

        /// Saves and loads

        // Save the parameters values in the file 'name' + ".neuralnet"
        void save(std::string name) {
            save_.save(name + ".neuralnet");
        }
        // Load the parameters values in the file 'name' + ".neuralnet"
        void load(std::string name) {
            save_.load(name + ".neuralnet");
        }

        // Print of the parameters
        void print() {
            save_.print();
        }


        /// LEARNING Methods

        // Initialize all parameters taking a coef of range 'coef'
        void init_parameters(T coef = (T)1) {
            for (auto it = parameters_.begin(); it != parameters_.end(); it++)
            {
                int k = it->second->get_width();
                it->second->matrix().randfill((T)-coef / (T)k, (T)coef / (T)k);
            }
        }
        // Initialize the parameter 'name' taking a range 'coef'
        void init_parameter(std::string name, T coef = (T)1) {
            parameters_[name]->matrix().randfill((T)-coef / (T)k, (T)coef / (T)k);
        }

        // Return the error matfunc of the neuralnetwork from an instance 'network' and a wanted output matfunc 'output'
        virtual MatFunc<T> create_error(MatFunc<T> output, MatFuncIndex<T>& network) {
            // Propre au réseau de neurone concerné
            return Gpu::nullmf<T>();
        }


        void __display_errors__(T training_error, T test_error, int iteration) {

            std::cout << "Iteration " << iteration << " : " << training_error << " - " << test_error << "\t";
            std::cout << std::endl;

        }

        template<template<typename> class Opt> void learning(MatrixPair<T> training, MatrixPair<T> test, HyperParameters<T> hyper_parameters, int nb_epoch, std::vector<double>* training_error_curve = nullptr, std::vector<double>* test_error_curve = nullptr, std::string prefix_saves = "", int freq_save = 100) {

            if (prefix_saves == "") { prefix_saves = "default_save/n_"; }

            // Training Error
            MatFunc<T> full_training_input = newmf_of_matrix<T>(training.X);
            MatFuncIndex<T> full_training_network = create_network(full_training_input);
            MatFunc<T> full_training_error = create_error(training.Y, full_training_network);



            // Test Error
            MatFunc<T> full_test_input = newmf_of_matrix<T>(test.X);
            MatFuncIndex<T> full_test_network = create_network(full_test_input);
            MatFunc<T> full_test_error = create_error(test.Y, full_test_network);



            // Training network instance creation

            int training_input_size = training.X.height();
            int training_output_size = training.Y.height();
            int training_nb_sample = training.Y.width();

            MatFunc<T> training_input = newmf<T>(training_input_size, 1, (T)0);
            MatFunc<T> training_output = newmf<T>(training_output_size, 1, (T)0);

            MatFuncIndex<T> training_network = create_network(training_input);
            MatFunc<T> training_error = create_error(training_output, training_network);

            Opt<T> optimizer(training_error, parameters(), hyper_parameters);



            // Iterate over the number of epoch

            for (int i = 0; i < nb_epoch; i++)
            {

                init_network(training_network);
                init_network(full_training_network);
                init_network(full_test_network);

                for (int j = 0; j < training_nb_sample; j++)
                {
                    training_input->matrix().extract(training.X, 0, j);
                    training_output->matrix().extract(training.Y, 0, j);
                    optimizer.optimize();
                }

                // Display and save of errors
                compute<T>(full_training_error);
                compute<T>(full_test_error);
                T err_train = full_training_error->matrix().get(0, 0);
                T err_test = full_test_error->matrix().get(0, 0);
                std::cout << "Iteration " << i << " : " << err_train << "\t\t" << err_test << "\n";
                if (training_error_curve != nullptr) { training_error_curve->push_back(err_train); }
                if (test_error_curve != nullptr) { test_error_curve->push_back(err_test); }

                // Save of network parameters
                if (i % freq_save == 0) {
                    save(prefix_saves + std::to_string(i / freq_save));
                }
            }


        }
        template<template<typename> class Opt> void batch_learning(MatrixPair<T> training, MatrixPair<T> test, HyperParameters<T> hyper_parameters, int nb_epoch, std::vector<double>* training_error_curve = nullptr, std::vector<double>* test_error_curve = nullptr, std::string prefix_saves = "", int freq_save = 100) {

            if (prefix_saves == "") { prefix_saves = "default_save/n_"; }

            // Test Error
            MatFunc<T> full_test_input = newmf_of_matrix<T>(test.X);
            MatFuncIndex<T> full_test_network = create_network(full_test_input);
            MatFunc<T> full_test_error = create_error(Gpu::newmf_of_matrix(test.Y), full_test_network);

            // Training network instance creation

            MatFunc<T> training_input = newmf_of_matrix<T>(training.X);
            MatFunc<T> training_output = newmf_of_matrix<T>(training.Y);

            MatFuncIndex<T> training_network = create_network(training_input);
            MatFunc<T> training_error = create_error(training_output, training_network);

            Opt<T> optimizer(training_error, parameters(), hyper_parameters);



            // Iterate over the number of epoch

            for (int i = 0; i < nb_epoch; i++)
            {
                init_network(training_network);
                init_network(full_test_network);

                optimizer.optimize();

                // Display and save of errors
                compute<T>(full_test_error);
                T err_train = training_error->matrix().get(0, 0);
                T err_test = full_test_error->matrix().get(0, 0);
                std::cout << "Iteration " << i << " : " << err_train << "\t\t" << err_test << "\n";
                if (training_error_curve != nullptr) { training_error_curve->push_back(err_train); }
                if (test_error_curve != nullptr) { test_error_curve->push_back(err_test); }

                // Save of network parameters
                if (i % freq_save == 0) {
                    save(prefix_saves + std::to_string(i / freq_save));
                }

            }


        };
        template<template<typename> class Opt> void batch_seq_learning(MatrixPair<T> training, MatrixPair<T> test, HyperParameters<T> hyper_parameters, int nb_epoch, int time_steps, std::vector<double>* training_error_curve = nullptr, std::vector<double>* test_error_curve = nullptr, std::string prefix_saves = "", int freq_save = 100) {

            if (prefix_saves == "") { prefix_saves = "default_save/n_"; }

            /// Training

            int training_input_size = training.X.height();
            int training_output_size = training.Y.height();
            int training_nb_sample = training.Y.width();
            int training_batch_size = training_nb_sample - time_steps;

            MatFunc<T> training_input = newmf<T>(training_input_size, training_batch_size);
            MatFunc<T> training_output = newmf<T>(training_output_size, training_batch_size);

            MatFuncIndex<T> training_network = create_network(training_input);
            MatFunc<T> training_error = create_error(training_output, training_network);

            Opt<T> optimizer(training_error, parameters(), hyper_parameters);



            /// Test

            int test_input_size = test.X.height();
            int test_output_size = test.Y.height();
            int test_nb_sample = test.Y.width();
            int test_batch_size = test_nb_sample - time_steps;

            MatFunc<T> test_input = newmf<T>(test_input_size, test_batch_size);
            MatFunc<T> test_output = newmf<T>(test_output_size, test_batch_size);

            MatFuncIndex<T> test_network = create_network(test_input);
            MatFunc<T> test_error = create_error(test_output, test_network);




            // Iterates over the number of epoch

            for (int i = 0; i < nb_epoch; i++)
            {

                /// Training

                init_network(training_network);

                for (int j = 0; j < time_steps; j++)
                {
                    training_input->matrix().extract(training.X, 0, j);
                    compute<T>(training_network);
                }

                training_input->matrix().extract(training.X, 0, time_steps);
                training_output->matrix().extract(training.Y, 0, time_steps);
                optimizer.optimize();




                /// Test

                init_network(test_network);

                for (int j = 0; j < time_steps; j++)
                {
                    test_input->matrix().extract(test.X, 0, j);
                    compute<T>(test_network);
                }

                test_input->matrix().extract(test.X, 0, time_steps);
                test_output->matrix().extract(test.Y, 0, time_steps);
                compute<T>(test_error);

                // Display and save of errors
                compute<T>(full_test_error);
                T err_train = training_error->matrix().get(0, 0);
                T err_test = test_error->matrix().get(0, 0);
                std::cout << "Iteration " << i << " : " << err_train << "\t\t" << err_test << "\n";
                if (training_error_curve != nullptr) { training_error_curve->push_back(err_train); }
                if (test_error_curve != nullptr) { test_error_curve->push_back(err_test); }

                // Save of network parameters
                if (i % freq_save == 0) {
                    save(prefix_saves + std::to_string(i / freq_save));
                }

            }
        }
        /*template<template<typename> class Opt> void batch_seq_learning2(MatrixPair<T> training, MatrixPair<T> test, HyperParameters<T> hyper_parameters, int nb_epoch, int time_steps, int time_steps_learning, T obj_error = 0, int freq_save = 100) {


            /// Training

            int training_input_size = training.X.height();
            int training_output_size = training.Y.height();
            int training_nb_sample = training.Y.width();
            int training_batch_size = training_nb_sample - time_steps - time_steps_learning;

            MatFunc<T> training_input = newmf<T>(training_input_size, training_batch_size);
            MatFunc<T> training_output = newmf<T>(training_output_size, training_batch_size);

            MatFuncIndex<T> training_network = create_network(training_input);
            MatFunc<T> training_error = create_error(training_output, training_network);

            Opt<T> optimizer(training_error, parameters(), hyper_parameters);



            /// Test

            int test_input_size = test.X.height();
            int test_output_size = test.Y.height();
            int test_nb_sample = test.Y.width();
            int test_batch_size = test_nb_sample - time_steps - time_steps_learning;

            MatFunc<T> test_input = newmf<T>(test_input_size, test_batch_size);
            MatFunc<T> test_output = newmf<T>(test_output_size, test_batch_size);

            MatFuncIndex<T> test_network = create_network(test_input);
            MatFunc<T> test_error = create_error(test_output, test_network);




            // Iterates over the number of epoch

            for (int i = 0; i < nb_epoch; i++)
            {

                /// Training

                init_network(training_network);
                T training_tot_err = 0;

                for (int j = 0; j < time_steps; j++)
                {
                    training_input->matrix().extract(training.X, 0, j);
                    compute<T>(training_network);
                }
                for (int j = 0; j < time_steps_learning; j++)
                {
                    training_input->matrix().extract(training.X, 0, time_steps + j);
                    training_output->matrix().extract(training.Y, 0, time_steps + j);
                    optimizer.optimize();
                    training_tot_err += training_error->matrix().get(0, 0);
                }


                /// Test

                init_network(test_network);
                T test_tot_err = 0;

                for (int j = 0; j < time_steps; j++)
                {
                    test_input->matrix().extract(test.X, 0, j);
                    compute<T>(test_network);
                }
                for (int j = 0; j < time_steps_learning; j++)
                {
                    test_input->matrix().extract(test.X, 0, time_steps + j);
                    test_output->matrix().extract(test.Y, 0, time_steps + j);
                    compute<T>(test_error);
                    test_tot_err += test_error->matrix().get(0, 0);
                }

                // Display
                __display_errors__(test_tot_err, training_tot_err, i);

                if (test_tot_err <= obj_error) {
                    i = nb_epoch + 1; // We end the iterations
                }

                if (i % freq_save == 0) {
                    save("tmp_save_" + std::to_string(i / freq_save));
                }

            }
        }*/

    };


    template<typename T, template<typename> class Opt> void learning(NeuralNetwork<T>* neuralnet, MatFunc<T> input, MatFunc<T> output, MatFunc<T> input_test, MatFunc<T> output_test, int nb_epoch, std::map<std::string, T> hyper_parameters) {

        int input_size = input->get_height();
        int output_size = output->get_height();
        int nb_sample = input->get_width();

        // Création de l'erreur totale
        MatFuncIndex<T> model = neuralnet->create_network(input);
        MatFunc<T> err = neuralnet->create_error(output, model);

        // Création des batchs
        MatFunc<T> IN = newmf<T>(input_size, 1);
        MatFunc<T> OUT = newmf<T>(output_size, 1);
        MatFuncIndex<T> MODEL = neuralnet->create_network(IN);
        MatFunc<T> ERR = neuralnet->create_error(OUT, MODEL);
        Opt<T> OPT(ERR, neuralnet->parameters(), hyper_parameters);

        // Affichage de l'erreur au début
        neuralnet->init_network(model);
        compute<T>(err);
        std::cout << "Debut :" << err->matrix();

        // Itération sur le nombre d'époque
        for (int i = 0; i < nb_epoch; i++)
        {
            neuralnet->init_network(MODEL);
            neuralnet->init_network(model);

            for (int j = 0; j < nb_sample; j++)
            {
                IN->matrix().extract(input->matrix(), 0, j);
                OUT->matrix().extract(output->matrix(), 0, j);
                OPT.optimize();
            }

            compute<T>(err);
            std::cout << "Epoque " << i << " :" << err->matrix();

        }

        neuralnet->init_network(model);
        compute<T>(err);
        std::cout << "Fin :" << err->matrix();

    }

    template<typename T, template<typename> class Opt> void batch_learning(NeuralNetwork<T>* neuralnet, MatFunc<T> input, MatFunc<T> output, MatFunc<T> input_test, MatFunc<T> outpu_test, int nb_epoch, std::map<std::string, T> hyper_parameters) {

        // Création de l'erreur 
        MatFuncIndex<T> network = neuralnet->create_network(input);
        MatFunc<T> error = neuralnet->create_error(output, network);

        // Préparation de l'optimiseur 
        Opt<T> opt(error, neuralnet->parameters(), hyper_parameters);

        // Affichage de l'erreur au début
        neuralnet->init_network(network);
        compute<T>(error);
        std::cout << "Debut :" << error->matrix();

        // Itération sur le nombre d'époque
        for (int i = 0; i < nb_epoch; i++)
        {
            neuralnet->init_network(network);
            opt.optimize();
            std::cout << "Epoque " << i << " :" << error->matrix();

        }

        neuralnet->init_network(network);
        compute<T>(error);
        std::cout << "Fin :" << error->matrix();

    }

    template<typename T, template<typename> class Opt> void seq_batch_learning(NeuralNetwork<T>* neuralnet, MatFunc<T> input, MatFunc<T> output, int nb_epoch, int time_steps, std::map<std::string, T> hyper_parameters) {

        /// Pour le train

        int input_size = input->get_height();
        int output_size = output->get_height();
        int nb_sample = input->get_width();
        int batch_size = nb_sample - time_steps;

        MatFunc<T> IN = newmf<T>(input_size, batch_size);
        MatFunc<T> OUT = newmf<T>(output_size, batch_size);

        MatFuncIndex<T> MODEL = neuralnet->create_network(IN);
        MatFunc<T> ERR = neuralnet->create_error(OUT, MODEL);

        Opt<T> OPT(ERR, neuralnet->parameters(), hyper_parameters);

        // Itération sur le nombre d'époque
        for (int i = 0; i < nb_epoch; i++)
        {

            neuralnet->init_network(MODEL);
            for (int j = 0; j < time_steps; j++)
            {
                IN->matrix().extract(input->matrix(), 0, j);
                compute<T>(MODEL);
            }
            IN->matrix().extract(input->matrix(), 0, time_steps);
            OUT->matrix().extract(output->matrix(), 0, time_steps);
            OPT.optimize();

            std::cout << "Epoque " << i << " : " << ERR->matrix();

        }

    }
    template<typename T, template<typename> class Opt> void seq_batch_learning(NeuralNetwork<T>* neuralnet, MatFunc<T> input, MatFunc<T> output, MatFunc<T> input_test, MatFunc<T> output_test, int nb_epoch, int time_steps, std::map<std::string, T> hyper_parameters) {

        /// Pour le train

        int input_size = input->get_height();
        int output_size = output->get_height();
        int nb_sample = input->get_width();
        int batch_size = nb_sample - time_steps;

        MatFunc<T> IN = newmf<T>(input_size, batch_size);
        MatFunc<T> OUT = newmf<T>(output_size, batch_size);

        MatFuncIndex<T> MODEL = neuralnet->create_network(IN);
        MatFunc<T> ERR = neuralnet->create_error(OUT, MODEL);

        Opt<T> OPT(ERR, neuralnet->parameters(), hyper_parameters);


        /// Pour le test

        int input_size_test = input_test->get_height();
        int output_size_test = output_test->get_height();
        int nb_sample_test = input_test->get_width();
        int batch_size_test = nb_sample_test - time_steps;

        MatFunc<T> IN_test = newmf<T>(input_size_test, batch_size_test);
        MatFunc<T> OUT_test = newmf<T>(output_size_test, batch_size_test);

        MatFuncIndex<T> MODEL_test = neuralnet->create_network(IN_test);
        MatFunc<T> ERR_test = neuralnet->create_error(OUT_test, MODEL_test);

        // Itération sur le nombre d'époque
        for (int i = 0; i < nb_epoch; i++)
        {

            /// Pour le train

            neuralnet->init_network(MODEL);
            for (int j = 0; j < time_steps; j++)
            {
                IN->matrix().extract(input->matrix(), 0, j);
                compute<T>(MODEL);
            }
            IN->matrix().extract(input->matrix(), 0, time_steps);
            OUT->matrix().extract(output->matrix(), 0, time_steps);
            OPT.optimize();

            /// Pour le test

            neuralnet->init_network(MODEL_test);
            for (int j = 0; j < time_steps; j++)
            {
                IN_test->matrix().extract(input_test->matrix(), 0, j);
                compute<T>(MODEL_test);
            }
            IN_test->matrix().extract(input->matrix(), 0, time_steps);
            OUT_test->matrix().extract(output->matrix(), 0, time_steps);
            compute<T>(ERR_test);

            std::cout << "Epoque " << i << " : " << ERR->matrix() << " - " << ERR_test->matrix();

        }

    }





}