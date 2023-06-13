#include <assert.h>
#include "sqlite3.h"
#include <string>
#include <vector>
#include <fstream>

#include <stdio.h>
#include <time.h>

#include "matfunc.cuh"
#include "data.cuh"
#include "neuralnetwork.cuh"

// Constantes /////////////////////////////////////////////////////////////////////////////////////////////////

BaseDeDonnees BDD(BDD_PATH);
context_donnees_t TRAINING_CONTEXT{ LIEU_LYON, ID_STATION_LYON, Date(2017, 01, 01), Date(2022, 01, 01) };
context_donnees_t TEST_CONTEXT{ LIEU_LYON, ID_STATION_LYON, Date(2022, 01, 01), Date(2023, 01, 01) };

Donnees TRAINING(&BDD, &TRAINING_CONTEXT);
Donnees TEST(&BDD, &TEST_CONTEXT);

const int INTERVALLE_QUART_HEURE = 3 * 4; // Le pas des séries temporelles en quart d'heure

Donnees TRAINING2(&BDD, &TRAINING_CONTEXT);
Donnees TEST2(&BDD, &TEST_CONTEXT);

const int INTERVALLE_QUART_HEURE2 = 2; // Le pas des séries temporelles en quart d'heure

// Donnees normalisées

Donnees_2 TRAINING_2(&BDD, &TRAINING_CONTEXT);
Donnees_2 TEST_2(&BDD, &TEST_CONTEXT);


// Print dans un fichier ///////////////////////////////////////////////////////////////////////////////////////

void print_results(std::string file, std::vector<Date> time, std::vector<Cpu::Matrix<double>> As) {

    std::ofstream fichier(file);

    for (int i = 0; i < time.size(); i++) {
        fichier << time[i] << ';';
        for (int j = 0; j < As.size(); j++)
        {
            for (int k = 0; k < As[j].height(); k++) {
                fichier << As[j].get(k, i) << ';';
            }

        }
        fichier << '\n';
    }

    fichier.close();
}




// Regressions linéaires ///////////////////////////////////////////////////////////////////////////////////////

void print_error(Cpu::Matrix<double> reelle, Cpu::Matrix<double> predic) {
    double sum = 0;
    for (int i = 0; i < reelle.width(); i++)
    {
        double tmp = reelle.get(0, i) - predic.get(0, i);
        sum += tmp * tmp;
    }
    std::cout << "\nErreur : " << sum / reelle.width() << "\n";
}

Cpu::Matrix<double> regression_lineaire(Cpu::MatrixPair<double> dataset) {
    auto X = dataset.X;
    auto Y = dataset.Y;
    Cpu::Matrix<double> X_XT(X.height(), X.height()), Y_XT(Y.height(), X.height()), RES(Y.height(), X.height());
    X_XT.Matprod(X, 'n', X, 't');
    X_XT.Reverse(X_XT);
    Y_XT.Matprod(Y, 'n', X, 't');
    RES.Matprod(Y_XT, 'n', X_XT, 'n');
    return RES;
}
Cpu::Matrix<double> resultats_regression_lineaire(Cpu::MatrixPair<double> dataset) {
    Cpu::Matrix<double> beta = regression_lineaire(dataset);
    Cpu::Matrix<double> TRY(dataset.Y.height(), dataset.Y.width());
    TRY.Matprod(beta, 'n', dataset.X, 'n');
    return TRY;
}
void resultats_regression_lineaire(std::string file_name, selection_donnees_t selection) {

    // On récupère les données
    Cpu::MatrixPair<double> trainset(TRAINING.dataset(&selection));
    Cpu::MatrixPair<double> testset(TEST.dataset(&selection));

    // On calcul beta sur le trainset
    Cpu::Matrix<double> beta = regression_lineaire(trainset);

    // On calcul la dérivée de la consommation avec le model sur le testset : derive_conso.get(0, i) = consommation(i)-consommation(i-1).
    Cpu::Matrix<double> derive_trouve(testset.Y.height(), testset.Y.width());
    derive_trouve.Matprod(beta, 'n', testset.X, 'n');

    // On récupère la derive réelle
    Cpu::Matrix<double> derive_reelle(1, derive_trouve.width());
    TEST.recup_consommation(derive_reelle, 0, 0, selection.nb_derniere_conso, derive_trouve.width(), 1);

    // On récupère la consommation réelle
    Cpu::Matrix<double> conso_reel(1, derive_trouve.width());
    TEST.recup_consommation(conso_reel, 0, 0, selection.nb_derniere_conso, derive_trouve.width(), 0);

    // On calcul la consommation trouvée par le modèle (la case 0 ne compte pas)
    Cpu::Matrix<double> conso_trouve(1, derive_trouve.width());
    conso_trouve.set(0, 0, 0);
    for (int i = 1; i < derive_trouve.width(); i++) {
        conso_trouve.set(0, i, conso_reel.get(0, i - 1) + derive_trouve.get(0, i));
    }

    // On marque les résultats dans un .csv
    print_results(file_name, TEST.times(selection.nb_derniere_conso), { derive_reelle, derive_trouve, conso_reel, conso_trouve });

}
void resultats_regression_lineaire_2(std::string file_name, selection_donnees_2_t selection) {

    // On récupère les données
    Cpu::MatrixPair<double> trainset(TRAINING_2.dataset(&selection));
    Cpu::MatrixPair<double> testset(TEST_2.dataset(&selection));

    // On calcul beta sur le trainset
    Cpu::Matrix<double> beta = regression_lineaire(trainset);
    //std::cout << trainset.X;

    // On calcul la dérivée de la consommation avec le model sur le testset : derive_conso.get(0, i) = consommation(i)-consommation(i-1).
    Cpu::Matrix<double> derive_trouve(testset.Y.height(), testset.Y.width());
    derive_trouve.Matprod(beta, 'n', testset.X, 'n');

    // On récupère la derive réelle
    Cpu::Matrix<double> derive_reelle(1, derive_trouve.width());
    TEST_2.recup_consommation(derive_reelle, 0, 0, selection.max_nb_par_var(), derive_trouve.width(), 1);

    // On récupère la consommation réelle
    Cpu::Matrix<double> conso_reel(1, derive_trouve.width());
    TEST_2.recup_consommation(conso_reel, 0, 0, selection.max_nb_par_var(), derive_trouve.width(), 0);

    // On calcul la consommation trouvée par le modèle (la case 0 ne compte pas)
    Cpu::Matrix<double> conso_trouve(1, derive_trouve.width());
    conso_trouve.set(0, 0, 0);
    for (int i = 1; i < derive_trouve.width(); i++) {
        conso_trouve.set(0, i, conso_reel.get(0, i - 1) + derive_trouve.get(0, i));
    }

    derive_reelle.Scalar(TEST_2.consommation_ecart_type(), derive_reelle, 0.);
    derive_trouve.Scalar(TEST_2.consommation_ecart_type(), derive_trouve, 0.);
    conso_reel.Scalar(TEST_2.consommation_ecart_type(), conso_reel, TEST_2.consommation_moyenne());
    conso_trouve.Scalar(TEST_2.consommation_ecart_type(), conso_trouve, TEST_2.consommation_moyenne());

    print_error(derive_reelle, derive_trouve);

    // On marque les résultats dans un .csv
    print_results(file_name, TEST.times(selection.max_nb_par_var()), { derive_reelle, derive_trouve, conso_reel, conso_trouve });

}

void creer_regressions_lineaires() {

    // Regression 1

    selection_donnees_t selection1{

    true, // Biais

    false, // Annee
    true,  // Moi
    true,  // Jour
    true,  // Heure
    true,  // Minutes
    true,  // Jour Semaine

    false, // dérivation de la temperature
    true,  // Temperature
    false, // dérivation de l'humidité
    true,  // Humidité
    false, // dérivation de la nébulosité
    true,  // Nébulosité
    false, // dérivation de la pluviometrie
    true,  // Pluviométrie

    1,     // Ordre de dérivation de la Consommation
    100,     // Nb dernière consommation prise en compte en entrée


    };
    resultats_regression_lineaire("Regression_1.csv", selection1);

    // Regression 2

    selection_donnees_t selection2{

    true, // Biais

    false, // Annee
    true,  // Moi
    true,  // Jour
    true,  // Heure
    true,  // Minutes
    true,  // Jour Semaine

    false, // dérivation de la temperature
    false,  // Temperature
    false, // dérivation de l'humidité
    false,  // Humidité
    false, // dérivation de la nébulosité
    false,  // Nébulosité
    false, // dérivation de la pluviometrie
    false,  // Pluviométrie

    1,     // Ordre de dérivation de la Consommation
    100,   // Nb dernière consommation prise en compte en entrée


    };
    resultats_regression_lineaire("Regression_2.csv", selection2);

    // Regression 3

    selection_donnees_t selection3{

    true, // Biais

    false, // Annee
    true,  // Moi
    true,  // Jour
    true,  // Heure
    true,  // Minutes
    true,  // Jour Semaine

    false, // dérivation de la temperature
    true,  // Temperature
    false, // dérivation de l'humidité
    true,  // Humidité
    false, // dérivation de la nébulosité
    true,  // Nébulosité
    false, // dérivation de la pluviometrie
    true,  // Pluviométrie

    1,     // Ordre de dérivation de la Consommation
    0,     // Nb dernière consommation prise en compte en entrée


    };
    resultats_regression_lineaire("Regression_3.csv", selection3);

    // Regression 4

    selection_donnees_t selection4{

    true, // Biais

    false, // Annee
    true,  // Moi
    true,  // Jour
    true,  // Heure
    true,  // Minutes
    true,  // Jour Semaine

    true, // dérivation de la temperature
    true,  // Temperature
    true, // dérivation de l'humidité
    true,  // Humidité
    true, // dérivation de la nébulosité
    true,  // Nébulosité
    true, // dérivation de la pluviometrie
    true,  // Pluviométrie

    1,     // Ordre de dérivation de la Consommation
    0,     // Nb dernière consommation prise en compte en entrée


    };
    resultats_regression_lineaire("Regression_4.csv", selection4);

    // Regression 5

    selection_donnees_t selection5{

    true, // Biais

    false, // Annee
    true,  // Moi
    true,  // Jour
    true,  // Heure
    true,  // Minutes
    true,  // Jour Semaine

    true, // dérivation de la temperature
    true,  // Temperature
    true, // dérivation de l'humidité
    true,  // Humidité
    true, // dérivation de la nébulosité
    true,  // Nébulosité
    true, // dérivation de la pluviometrie
    true,  // Pluviométrie

    1,     // Ordre de dérivation de la Consommation
    100,     // Nb dernière consommation prise en compte en entrée


    };
    resultats_regression_lineaire("Regression_5.csv", selection5);



}

void creer_regressions_lineaires_2() {

    // Regression 1

    int nb_derniere_donnees = 8 * 7;
    selection_donnees_2_t selection1{

    true, // Biais

    false, // Annee
    true,  // Moi
    true,  // Jour
    true,  // Heure
    false,  // Minutes
    true,  // Jour Semaine

    0, // nb_temperature
    0,  // nb_humidite
    0, // nb_nebulosite
    nb_derniere_donnees,  // nb_variation_temperature
    nb_derniere_donnees, // nb_variation_humidite
    nb_derniere_donnees,  // nb_nebulosite

    0,     // nb_conso
    nb_derniere_donnees,     // nb_variation_conso


    };

    resultats_regression_lineaire_2("Regression_1.csv", selection1);
}


class MLP_2_1 : public Gpu::NeuralNetwork<double> {

private:

    int nb_derniere_donnees = 8 * 7;
    selection_donnees_2_t selection{

    false, // Biais

    false, // Annee
    false,  // Mois
    true, // Mois 2
    false,  // Jour
    true,  // Heure
    false,  // Minutes
    true,  // Jour Semaine

    1, // nb_temperature
    1,  // nb_humidite
    1, // nb_nebulosite
    1,  // nb_variation_temperature
    1, // nb_variation_humidite
    1,  // nb_nebulosite

    1,     // nb_conso
    1,     // nb_variation_conso


    };

    //int time_steps{ 10 };
    int taille_entree{ selection.nb_donnees_entrees() };
    int taille_cachee{ 7 };
    //int taille_lstm{ 50 };
    int taille_sortie{ 1 };

    double facteur_sortie = 2.0;

    Gpu::MatrixPair<double> trainset, testset;

public:

    MLP_2_1() : trainset(TRAINING_2.dataset(&selection)), testset(TEST_2.dataset(&selection)) {

        // Architecture :
        // (entree)->(dense0(tanh))->(dense1(tanh))->(scaling2(* 1500))->(sortie)

        // Erreur : MSE

        //add_scaling_params("scaling1", taille_entree);
        add_dense_params("dense0", taille_entree, taille_cachee);
        add_dense_params("dense1", taille_cachee, taille_sortie);

    }

    virtual Gpu::MatFuncIndex<double> create_network(Gpu::MatFunc<double> input) {

        Gpu::MatFuncIndex<double> network;

        network["input"] = input;

        //create_scaling(network, "scaling1", input);

        create_dense(network, "dense0", network["input"], "tanh");

        create_dense(network, "dense1", network["dense0"], "tanh");

        network["sortie"] = network["dense1"];

        return network;

    }
    virtual Gpu::MatFunc<double> create_error(Gpu::MatFunc<double> output, Gpu::MatFuncIndex<double>& network) {

        return Gpu::msemf(output / facteur_sortie, network["sortie"]);

    }

    void entrainement_phase1(double learning_rate, int nb_epoch, double obj_error = 0) {

        Gpu::HyperParameters<double> hyper_parameters{
            {"learning_rate", learning_rate },
            { "decay1", 0.9 },
            { "decay2", 0.999 }
        };

        batch_seq_learning<Gpu::OptimizerAdam>(trainset, testset, hyper_parameters, nb_epoch, 0);

    }
    void resultats(std::string file_name) {

        int input_size = testset.X.height();
        int nb_sample = testset.X.width();
        int batch_size = nb_sample;

        // On créer une entrée pour le réseau et une instance du MLP
        Gpu::MatFuncIndex<double> network = create_network(Gpu::newmf_of_matrix(testset.X));

        // On calcule la sortie du réseau
        compute(network["sortie"]);
        Cpu::Matrix<double> derive_trouve(1, batch_size);
        derive_trouve.copy(network["sortie"]->matrix());
        derive_trouve.Scalar(facteur_sortie, derive_trouve, 0.);

        Cpu::Matrix<double> derive_reelle(1, derive_trouve.width());
        Cpu::Matrix<double> conso_reelle(1, derive_trouve.width());
        Cpu::Matrix<double> conso_trouve(1, derive_trouve.width());

        // On récupère la dérivée réelle
        TEST_2.recup_consommation(derive_reelle, 0, 0, selection.max_nb_par_var(), derive_trouve.width(), 1);

        // On récupère la consommation réelle et on calcule la consommation trouvée par le réseau
        TEST_2.recup_consommation(conso_reelle, 0, 0, selection.max_nb_par_var(), derive_trouve.width(), 0);
        TEST_2.recup_consommation(conso_trouve, 0, 0, selection.max_nb_par_var() - 1, derive_trouve.width(), 0);
        conso_trouve.Linear(1, conso_trouve, 1, derive_trouve);

        // On rescale les sortie pour avoir des valeurs en MW
        derive_reelle.Scalar(TEST_2.consommation_ecart_type(), derive_reelle, 0.);
        derive_trouve.Scalar(TEST_2.consommation_ecart_type(), derive_trouve, 0.);
        conso_reelle.Scalar(TEST_2.consommation_ecart_type(), conso_reelle, TEST_2.consommation_moyenne());
        conso_trouve.Scalar(TEST_2.consommation_ecart_type(), conso_trouve, TEST_2.consommation_moyenne());

        print_error(derive_reelle, derive_trouve);

        // On marque les résultats dans un .csv
        print_results(file_name, TEST_2.times(selection.max_nb_par_var()), { derive_reelle, derive_trouve, conso_reelle, conso_trouve });

        // On marque les résultats d'une regression linéaire sur la même selection
        selection.biais = true;
        resultats_regression_lineaire_2("mlp_2_1_regression_lineaire.csv", selection);

    }


};






int main() {

    srand(time(nullptr));

    TRAINING.creer_requete();
    TEST.creer_requete();
    TRAINING.recuperer_donnees(INTERVALLE_QUART_HEURE);
    TEST.recuperer_donnees(INTERVALLE_QUART_HEURE);

    TRAINING2.creer_requete();
    TEST2.creer_requete();
    TRAINING2.recuperer_donnees(INTERVALLE_QUART_HEURE2);
    TEST2.recuperer_donnees(INTERVALLE_QUART_HEURE2);

    TRAINING_2.creer_requete();
    TEST_2.creer_requete();
    TRAINING_2.recuperer_donnees();
    TEST_2.recuperer_donnees();

    MLP_2_1 mlp1;
    mlp1.init_parameters();
    mlp1.entrainement_phase1(0.001, 100);
    mlp1.save("MLP_2_1_0");
    mlp1.resultats("test_mlp_2_1.csv");



    return 0;
}