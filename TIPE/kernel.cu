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

// Donnees normalisées
Donnees TRAINING(&BDD, &TRAINING_CONTEXT);
Donnees TEST(&BDD, &TEST_CONTEXT);

void init_donnees() {
    TRAINING.creer_requete();
    TEST.creer_requete();
    TRAINING.recuperer_donnees();
    TEST.recuperer_donnees();
}

// Fonctions pour les résultats des modèles ///////////////////////////////////////////////////////////////////////////////////////

void print_resultats(std::string file, std::vector<Date> time, std::vector<Cpu::Matrix<double>> As) {

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
void print_vecteurs(std::string file, std::vector<std::vector<double>> As) {
    // On suppose qu'ils ont tous la même taille
    std::ofstream fichier(file);

    for (int i = 0; i < As[0].size(); i++) {
        for (int j = 0; j < As.size(); j++)
        {
            fichier << As[j][i] << ';';
        }
        fichier << '\n';
    }

    fichier.close();
}
double erreur_modele(Cpu::Matrix<double> reelle, Cpu::Matrix<double> predic) {
    double sum = 0;
    for (int i = 0; i < reelle.width(); i++)
    {
        double tmp = reelle.get(0, i) - predic.get(0, i);
        sum += tmp * tmp;
    }
    return sum / reelle.width();
}




// Regressions linéaires /////////////////////////////////////////////////////////////////////////////////////////////////////////

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
Cpu::Matrix<double> calcul_regression_lineaire(Cpu::Matrix<double> beta, Cpu::Matrix<double> X) {
    Cpu::Matrix<double> res(beta.height(), X.width());
    res.Matprod(beta, 'n', X, 'n');
    return res;
}
double resultats_regression_lineaire(selection_donnees_t selection, Cpu::Matrix<double>* reelle = nullptr, Cpu::Matrix<double>* predic = nullptr) {
     
    // On calcul le modèle
    Cpu::Matrix<double> beta = regression_lineaire(TRAINING.dataset(&selection));

    // On récupère le set de test
    Cpu::MatrixPair<double> testset(TEST.dataset(&selection));

    // On calcul la dérivée prédite par le modèle
    Cpu::Matrix<double> Yp = calcul_regression_lineaire(beta, testset.X);

    // On ramène en MW car les données étaient normalisée jusqu'ici
    testset.Y.Scalar(TEST.consommation_ecart_type(), testset.Y, 0.);
    Yp.Scalar(TEST.consommation_ecart_type(), Yp, 0.);
    
    // On passe le résultat dans les pointeurs
    if (reelle != nullptr) { *reelle = testset.Y; }
    if (predic != nullptr) { *predic = Yp; }

    return erreur_modele(testset.Y, Yp);

}




// Réseau /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class MLP : public Gpu::NeuralNetwork<double> {

// Architecture :
// (entree)->(dense0(tanh))->(dense1(tanh))->(sortie)
// Erreur : MSE

private:

    selection_donnees_t _selection;
    int _taille_entree;
    int _taille_cachee;
    int _taille_sortie{ 1 };

    double _facteur_sortie = 4.0;
    Gpu::MatrixPair<double> _trainset, _testset;

public:

    MLP(selection_donnees_t selection, int taille_cachee)
        : _trainset(TRAINING.dataset(&_selection)), _testset(TEST.dataset(&_selection)),
        _selection(selection), _taille_entree(selection.nb_donnees_entrees()), _taille_cachee(taille_cachee) {

        add_dense_params("dense0", _taille_entree, _taille_cachee);
        add_dense_params("dense1", _taille_cachee, _taille_sortie);

    }

    virtual Gpu::MatFuncIndex<double> create_network(Gpu::MatFunc<double> input) {

        Gpu::MatFuncIndex<double> network;

        network["input"] = input;

        create_dense(network, "dense0", network["input"], "tanh");

        create_dense(network, "dense1", network["dense0"], "tanh");

        network["sortie"] = network["dense1"];

        return network;

    }
    virtual Gpu::MatFunc<double> create_error(Gpu::MatFunc<double> output, Gpu::MatFuncIndex<double>& network) {

        return Gpu::msemf(output / _facteur_sortie, network["sortie"]);

    }

    void entrainement(double learning_rate, int nb_epoch, std::vector<double>* training_error_curve, std::vector<double>* test_error_curve, std::string prefix_saves, int freq_saves = 100) {

        Gpu::HyperParameters<double> hyper_parameters{
            {"learning_rate", learning_rate },
            { "decay1", 0.9 },
            { "decay2", 0.999 }
        };

        batch_learning<Gpu::OptimizerAdam>(_trainset, _testset, hyper_parameters, nb_epoch, training_error_curve, test_error_curve, prefix_saves, freq_saves);
    }

    double resultats(Cpu::Matrix<double>* reelle = nullptr, Cpu::Matrix<double>* predic = nullptr) {

        Gpu::MatFuncIndex<double> network = create_network(Gpu::newmf_of_matrix(_testset.X)); // On créer une instance du MLP sur l'entrée
        compute(network["sortie"]); // On calcul la sortie

        Cpu::Matrix<double> Yp(network["sortie"]->matrix()); // On récupère le résultat sur le gpu
        Yp.Scalar(_facteur_sortie * TEST.consommation_ecart_type(), Yp, 0.); // On multiplie par le facteur de sortie * ecart type pour obtenir la sortie en MW

        Cpu::Matrix<double> Y(_testset.Y); // On récupère le résultat réelle sur le cpu
        Y.Scalar(TEST.consommation_ecart_type(), Y, 0.); // On multiplie par l'ecart type pour l'obtenir en MW

        // On transmet les résultats dans les pointeurs
        if (reelle != nullptr) { *reelle = Y; }
        if (predic != nullptr) { *predic = Yp; }

        return erreur_modele(Y, Yp);

    }


};



// Tests /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


void ml_tests_entrees() {

    std::vector<std::string> noms_tests;
    std::vector<double> erreurs;
    selection_donnees_t selection;

    noms_tests.push_back("mois-jour-heure-1*température-1*humdité-1*nébulosité-1*var_consommation");
    selection = {

            true,   // Biais

            false,  // Annee
            true,   // Mois
            false,  // Mois 2
            true,   // Jour
            false,  // Jour Semaine
            true,   // Heure
            false,  // Heure_2
            false,  // Minutes


            1,  // nb_temperature
            1,  // nb_humidite
            1,  // nb_nebulosite
            0,  // nb_variation_temperature
            0,  // nb_variation_humidite
            0,  // nb_nebulosite

            0,  // nb_conso
            1,  // nb_variation_conso


    };
    erreurs.push_back(resultats_regression_lineaire(selection));

    noms_tests.push_back("mois_2-jour_semaine-heure_2-1*température-1*humdité-1*nébulosité-1*var_consommation");
    selection = {

            true,   // Biais

            false,  // Annee
            false,   // Mois
            true,  // Mois 2
            false,   // Jour
            true,  // Jour Semaine
            false,   // Heure
            true,  // Heure_2
            false,  // Minutes


            1,  // nb_temperature
            1,  // nb_humidite
            1,  // nb_nebulosite
            0,  // nb_variation_temperature
            0,  // nb_variation_humidite
            0,  // nb_nebulosite

            0,  // nb_conso
            1,  // nb_variation_conso


    };
    erreurs.push_back(resultats_regression_lineaire(selection));

    int n = 7 * 8 + 1;
    noms_tests.push_back("mois_2-jour_semaine-heure_2-n*température-n*humdité-n*nébulosité-n*var_consommation");
    selection = {

            true,   // Biais

            false,  // Annee
            false,   // Mois
            true,  // Mois 2
            false,   // Jour
            true,  // Jour Semaine
            false,   // Heure
            true,  // Heure_2
            false,  // Minutes


            n,  // nb_temperature
            n,  // nb_humidite
            n,  // nb_nebulosite
            0,  // nb_variation_temperature
            0,  // nb_variation_humidite
            0,  // nb_nebulosite

            0,  // nb_conso
            n,  // nb_variation_conso


    };
    erreurs.push_back(resultats_regression_lineaire(selection));

    noms_tests.push_back("mois_2-jour_semaine-heure_2-0*température-0*humdité-0*nébulosité-n*var_consommation");
    selection = {

            true,   // Biais

            false,  // Annee
            false,   // Mois
            true,  // Mois 2
            false,   // Jour
            true,  // Jour Semaine
            false,   // Heure
            true,  // Heure_2
            false,  // Minutes


            0,  // nb_temperature
            0,  // nb_humidite
            0,  // nb_nebulosite
            0,  // nb_variation_temperature
            0,  // nb_variation_humidite
            0,  // nb_nebulosite

            0,  // nb_conso
            n,  // nb_variation_conso


    };
    erreurs.push_back(resultats_regression_lineaire(selection));

    noms_tests.push_back("mois_2-jour_semaine-heure_2-1*température-1*humdité-1*nébulosité-n*var_consommation");
    selection = {

            true,   // Biais

            false,  // Annee
            false,   // Mois
            true,  // Mois 2
            false,   // Jour
            true,  // Jour Semaine
            false,   // Heure
            true,  // Heure_2
            false,  // Minutes


            1,  // nb_temperature
            1,  // nb_humidite
            1,  // nb_nebulosite
            0,  // nb_variation_temperature
            0,  // nb_variation_humidite
            0,  // nb_nebulosite

            0,  // nb_conso
            n,  // nb_variation_conso


    };
    erreurs.push_back(resultats_regression_lineaire(selection));

    // Ecriture dans le .txt

    std::ofstream fichier("ResultatsTests/ml_tests_entrees.txt");

    for (int i = 0; i < erreurs.size(); i++)
    {
        fichier << noms_tests[i] << ": " << std::to_string(erreurs[i]) << "\n";
    }

    fichier.close();

    
    // Graphe en fonction du nb de termes antérieurs fournis en entrées

    std::vector<double> erreurs_temps_normal, erreurs_temps_modifies;

    for (int i = 0; i <= 100; i++)
    {
        selection = {

            true,   // Biais

            false,  // Annee
            true,   // Mois
            false,  // Mois 2
            true,   // Jour
            false,  // Jour Semaine
            true,   // Heure
            false,  // Heure_2
            false,  // Minutes


            0,  // nb_temperature
            0,  // nb_humidite
            0,  // nb_nebulosite
            0,  // nb_variation_temperature
            0,  // nb_variation_humidite
            0,  // nb_nebulosite

            0,  // nb_conso
            i,  // nb_variation_conso


        };
        erreurs_temps_normal.push_back(resultats_regression_lineaire(selection));
        selection = {

            true,   // Biais

            false,  // Annee
            false,   // Mois
            true,  // Mois 2
            false,   // Jour
            true,  // Jour Semaine
            false,   // Heure
            true,  // Heure_2
            false,  // Minutes


            0,  // nb_temperature
            0,  // nb_humidite
            0,  // nb_nebulosite
            0,  // nb_variation_temperature
            0,  // nb_variation_humidite
            0,  // nb_nebulosite

            0,  // nb_conso
            i,  // nb_variation_conso


        };
        erreurs_temps_modifies.push_back(resultats_regression_lineaire(selection));
    }

    print_vecteurs("ResultatsTests/ml_tests_nb_termes_hist(0-100)_temps_normal_et_temps_modifies.csv", { erreurs_temps_normal, erreurs_temps_modifies });

}


void reseaux_test_dim_couche_cachee() {

    int nb = 7*8+1;
    selection_donnees_t selection{

            false,   // Biais

            false,  // Annee
            false,   // Mois
            true,  // Mois 2
            false,   // Jour
            true,  // Jour Semaine
            false,   // Heure
            true,  // Heure_2
            false,  // Minutes


            nb,  // nb_temperature
            nb,  // nb_humidite
            nb,  // nb_nebulosite
            0,  // nb_variation_temperature
            0,  // nb_variation_humidite
            0,  // nb_nebulosite

            0,  // nb_conso
            nb,  // nb_variation_conso


    };
    std::vector<std::vector<double>> errors_curves;

    std::vector<int> taille_cachee_a_tester;
    taille_cachee_a_tester.push_back(7);
    taille_cachee_a_tester.push_back(20);
    taille_cachee_a_tester.push_back(50);
    taille_cachee_a_tester.push_back(100);

    std::string tmp = "";
    for (int i = 0; i < taille_cachee_a_tester.size(); i++)
    {
        std::vector<double> train_error;
        std::vector<double> test_error;
        MLP mlp(selection, taille_cachee_a_tester[i]);
        mlp.init_parameters();
        mlp.entrainement(0.001, 10000, &train_error, &test_error, "SauvegardesReseaux/reseaux_test_dim_couche_cachee(taille_cachee " + std::to_string(taille_cachee_a_tester[i]) + ")_");
        errors_curves.push_back(train_error);
        errors_curves.push_back(test_error);
        tmp += "(" + std::to_string(taille_cachee_a_tester[i]) + ")";
    }

    // Rescaling (Pour avoir tout en MW) car les données étaient normalisées
    double scale = TEST.consommation_ecart_type();
    scale = 16.0 * scale * scale;
    for (int i = 0; i < errors_curves.size(); i++)
    {
        for (int j = 0; j < errors_curves[i].size(); j++)
        {
            errors_curves[i][j] = errors_curves[i][j] * scale;
        }
    }

    // Ecriture dans un fichier csv
    print_vecteurs("ResultatsTests/reseaux_test_dim_couche_cachee"+tmp+".csv", errors_curves);

}



int main() {

    srand((unsigned int)time(nullptr));

    init_donnees();

    //ml_tests_entrees();
    reseaux_test_dim_couche_cachee();
    


    return 0;
}