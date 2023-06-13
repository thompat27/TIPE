#pragma once

// STD
#include <assert.h>
#include <string>
#include <vector>
#include <fstream>
#include <stdio.h>
#include <ctime>

// INTERNALS
#include "sqlite3.h"
#include "matrix.cuh"
#include "utilities.cuh"




// CONSTANTS /////////////////////////////////////////////////////////////////////////////////////////////////////

const std::string BDD_PATH = "C:/Users/Thomas Valentin/Desktop/TIPE/TIPE/Donnees/BDD/tipe-conso-elec.db";
const std::string LIEU_LYON = "grand_lyon";
const int ID_STATION_LYON = 7481;
const std::string NOM_JOUR_SEMAINE[] = { "Lundi","Mardi","Mercredi","Jeudi","Vendredi","Samedi","Dimanche" };


// OBJECTS /////////////////////////////////////////////////////////////////////////////////////////////////////

class BaseDeDonnees {

private:

    sqlite3* BDD;

public:

    BaseDeDonnees(std::string bdd_path) {

        int code_ret;

        code_ret = sqlite3_open(bdd_path.c_str(), &BDD);

        if (code_ret) {
            fprintf(stderr, "Impossible d'ouvrir la base de donnees : %s\n", sqlite3_errmsg(BDD));
            sqlite3_close(BDD);
            assert(0);
        }

    }
    ~BaseDeDonnees() {
        sqlite3_close(BDD);
    }

    sqlite3_stmt* creer_requete(std::string requete) {
        int code_ret;
        const char* pzTail = nullptr;
        sqlite3_stmt* stmt;
        code_ret = sqlite3_prepare_v2(BDD, requete.c_str(), requete.size(), &stmt, &pzTail);
        if (code_ret != SQLITE_OK) {
            std::cout << "Erreur de preparation : " << sqlite3_errstr(code_ret) << std::endl;
            assert(0);
        }
        return stmt;
    }

};

class Date {

private:

    int _annee;
    int _moi;
    int _jour;
    int _heure;
    int _minutes;

public:

    Date(int annee = 0, int moi = 0, int jour = 0, int heure = 0, int minutes = 0) {
        fixer(annee, moi, jour, heure, minutes);
    }

    Date(long long date_heure) {
        fixer(date_heure);
    }
    Date(const std::tm& date_heure) : _annee(date_heure.tm_year), _moi(date_heure.tm_mon), _jour(date_heure.tm_mday), _heure(date_heure.tm_hour), _minutes(date_heure.tm_min) {}

    void fixer(int annee = 0, int moi = 0, int jour = 0, int heure = 0, int minutes = 0) {
        _annee = annee;
        _moi = moi;
        _jour = jour;
        _heure = heure;
        _minutes = minutes;
    }

    void fixer(long long date_heure) {
        _annee = date_heure / 100000000;
        _moi = (date_heure / 1000000) % 100;
        _jour = (date_heure / 10000) % 100;
        _heure = (date_heure / 100) % 100;
        _minutes = date_heure % 100;
    }
    long long vers_code() const {
        long long annee = _annee;
        long long moi = _moi;
        long long jour = _jour;
        long long heure = _heure;
        long long minutes = _minutes;
        return (((((((annee * 100LL) + moi) * 100LL) + jour) * 100LL) + heure) * 100LL) + minutes;
    }

    int annee() const {
        return _annee;
    }
    int moi() const {
        return _moi;
    }
    int jour() const {
        return _jour;
    }
    int heure() const {
        return _heure;
    }
    int minutes() const {
        return _minutes;
    }
    int jour_semaine() const {
        struct tm tm = { 0, 0, 12, _jour, _moi - 1, _annee - 1900, 0, 0, 0 };
        time_t t = mktime(&tm);
        int tmp = t < 0 ? -1 : localtime(&t)->tm_wday;
        if (tmp == -1) {
            std::cout << "Echec du calcul du jour de la semaine !\n";
            assert(0);
        }
        else {
            return (tmp + 6) % 7;
        }

    }

    std::tm get_tm() {
        return std::tm{ 0, minutes(), heure(), jour(), moi(), annee(), 0, 0, 0 };
    }

    Date add(Date date) {
        time_t time = std::mktime(&get_tm());
        time += std::mktime(&date.get_tm());
        std::time(&time);
        std::tm new_time = *std::localtime(&time);
        return Date(new_time.tm_year+1900, new_time.tm_mon+1, new_time.tm_mday, new_time.tm_hour, new_time.tm_min);
    }

};

std::string __my_to_string__(int i) {
    if (i < 10) {
        return "0" + std::to_string(i);
    }
    return std::to_string(i);
}
std::ostream& operator<<(std::ostream& os, Date d) {
    os << __my_to_string__(d.jour()) << "/" << __my_to_string__(d.moi()) << "/" << __my_to_string__(d.annee()) << " " << __my_to_string__(d.heure()) << ":" << __my_to_string__(d.minutes()) << ":00";
    return os;
}


struct context_donnees_t {

    std::string lieu_conso{ "" };
    int id_station{ 0 };
    Date debut{ Date(0) };
    Date fin{ Date(0) };

};


struct OrdreException : public std::exception {
    const char* what() const throw() {
        return "Exception : Ordre invalide !\n";
    }
};
template<typename T> T get_ordre(std::vector<T>& vec, int i, int ordre) {
    if (ordre == 0) {
        if (0 <= i && i < vec.size()) { return vec[i]; }
        throw OrdreException();
    }
    else {
        try
        {
            return get_ordre<T>(vec, i, ordre - 1) - get_ordre<T>(vec, i - 1, ordre - 1);
        }
        catch (OrdreException&)
        {
            return 0;
        }
    }
}

struct selection_donnees_t {

    bool biais{ false };

    // Données temporelles

    bool annee{ false };

    bool moi{ false };

    bool jour{ false };

    bool heure{ false };

    bool minutes{ false };

    bool jour_semaine{ false };

    // Données météo

    bool derivation_temperature{ false };
    bool temperature{ false };

    bool derivation_humidite{ false };
    bool humidite{ false };

    bool derivation_nebulosite{ false };
    bool nebulosite{ false };

    int derivation_pluviometrie{ false };
    bool pluviometrie{ false };

    // Données de consommation

    int ordre_derivation_conso{ 0 };

    int nb_derniere_conso{ 0 }; // Le nombre des dernière consommations prises en compte en entree


    int nb_donnees() {

        int n = 0;

        if (biais) n++;
        if (annee) n++;
        if (moi) n++;
        if (jour) n++;
        if (heure) n++;
        if (minutes) n++;
        if (jour_semaine) n++;

        n += nb_derniere_conso;

        if (temperature) n++;
        if (humidite) n++;
        if (nebulosite) n++;
        if (pluviometrie) n++;

        if (derivation_temperature) n++;
        if (derivation_humidite) n++;
        if (derivation_nebulosite) n++;
        if (derivation_pluviometrie) n++;

        return n;

    }

};

class Donnees {

private:

    BaseDeDonnees* bdd_;
    context_donnees_t* context_;

    std::string requete_;
    std::vector<Date> date_heure_;
    std::vector<double> consommation_;
    std::vector<double> temperature_;
    std::vector<double> humidite_;
    std::vector<double> nebulosite_;
    std::vector<double> pluviometrie_;


public:

    Donnees(BaseDeDonnees* bdd, context_donnees_t* context) : bdd_(bdd), context_(context)
    {}

    void creer_requete() {

        std::string lieu_conso = context_->lieu_conso;
        int id_station = context_->id_station;
        Date debut = context_->debut;
        Date fin = context_->fin;

        std::string begin = std::to_string(debut.vers_code());
        std::string end = std::to_string(fin.vers_code());

        requete_  = "SELECT consommation.date_heure, consommation.conso, meteo.temperature, meteo.humidite, meteo.nebulosite, meteo.pluviometrie ";
        requete_ += "FROM consommation JOIN meteo ON meteo.date_heure >= consommation.date_heure - 130 AND meteo.date_heure < consommation.date_heure + 170 ";
        requete_ += "WHERE consommation.date_heure >= " + begin + " AND consommation.date_heure <= " + end + " ";
        requete_ += "AND consommation.lieu = \"" + lieu_conso + "\" AND meteo.id_station=" + std::to_string(id_station);// On enlève les absences de données
        requete_ += " AND consommation.conso IS NOT NULL AND meteo.temperature IS NOT NULL AND meteo.humidite IS NOT NULL AND meteo.nebulosite IS NOT NULL AND meteo.pluviometrie IS NOT NULL";
        requete_ += ";";

    }
    void recuperer_donnees(int nb_quart_heure_interval = 4) {

        int code_ret;
        auto rq = bdd_->creer_requete(requete_);

        for (int i = 0; SQLITE_ROW == (code_ret = sqlite3_step(rq)); i++)
        {
            
            if (i % nb_quart_heure_interval == 0) {

                date_heure_.push_back(Date((long long)sqlite3_column_int64(rq, 0)));
                consommation_.push_back(sqlite3_column_double(rq, 1));
                temperature_.push_back(sqlite3_column_double(rq, 2));
                humidite_.push_back(sqlite3_column_double(rq, 3));
                nebulosite_.push_back(sqlite3_column_double(rq, 4));
                pluviometrie_.push_back(sqlite3_column_double(rq, 5));


            }
        }

        if (code_ret != SQLITE_DONE) {
            std::cout << "Erreur de d'execution : " << sqlite3_errstr(code_ret) << std::endl;
            assert(0);
        }

        sqlite3_finalize(rq);

    }
    void vider() {

        date_heure_.clear();
        temperature_.clear();
        humidite_.clear();
        nebulosite_.clear();
        pluviometrie_.clear();

    }

    context_donnees_t* context() {
        return context_;
    }
    int taille() {
        return date_heure_.size();
    }

    Date time(int i) {
        return date_heure_[i];
    }
    double consommation(int i) {
        return consommation_[i];
    }
    double temperature(int i) {
        return temperature_[i];
    }
    double humidite(int i) {
        return humidite_[i];
    }
    double nebulosite(int i) {
        return nebulosite_[i];
    }
    double pluviometrie(int i) {
        return pluviometrie_[i];
    }

    std::vector<Date> times(int deb = -1, int fin = -1) {

        if (deb == -1) { deb = 0; }
        if (fin == -1) { fin = date_heure_.size(); }

        std::vector<Date> res;
        for (int i = deb; i < fin; i++) {
            res.push_back(time(i));
        }

        return res;
    }
    std::vector<double> consommations(int deb = -1, int fin = -1) {

        if (deb == -1) { deb = 0; }
        if (fin == -1) { fin = date_heure_.size(); }

        std::vector<double> res;
        for (int i = deb; i < fin; i++) {
            res.push_back(consommation(i));
        }

        return res;
    }
    std::vector<double> temperatures(int deb = -1, int fin = -1) {

        if (deb == -1) { deb = 0; }
        if (fin == -1) { fin = date_heure_.size(); }

        std::vector<double> res;
        for (int i = deb; i < fin; i++) {
            res.push_back(temperature(i));
        }

        return res;
    }
    std::vector<double> humidites(int deb = -1, int fin = -1) {

        if (deb == -1) { deb = 0; }
        if (fin == -1) { fin = date_heure_.size(); }

        std::vector<double> res;
        for (int i = deb; i < fin; i++) {
            res.push_back(humidite(i));
        }

        return res;
    }
    std::vector<double> nebulosites(int deb = -1, int fin = -1) {

        if (deb == -1) { deb = 0; }
        if (fin == -1) { fin = date_heure_.size(); }

        std::vector<double> res;
        for (int i = deb; i < fin; i++) {
            res.push_back(nebulosite(i));
        }

        return res;
    }
    std::vector<double> pluviometries(int deb = -1, int fin = -1) {

        if (deb == -1) { deb = 0; }
        if (fin == -1) { fin = date_heure_.size(); }

        std::vector<double> res;
        for (int i = deb; i < fin; i++) {
            res.push_back(pluviometrie(i));
        }

        return res;
    }   

    void recup_annee(Cpu::Matrix<double>& X, int ligne_X, int deb_X, int deb, int number) {
        for (int i = 0; i < number; i++) {
            int j = i + deb;
            int k = i + deb_X;
            X.set(ligne_X, k, (double)time(j).annee());
        }
    }
    void recup_moi(Cpu::Matrix<double>& X, int ligne_X, int deb_X, int deb, int number) {
        for (int i = 0; i < number; i++) {
            int j = i + deb;
            int k = i + deb_X;
            X.set(ligne_X, k, (double)time(j).moi());
        }
    }
    void recup_jour(Cpu::Matrix<double>& X, int ligne_X, int deb_X, int deb, int number) {
        for (int i = 0; i < number; i++) {
            int j = i + deb;
            int k = i + deb_X;
            X.set(ligne_X, k, (double)time(j).jour());
        }
    }
    void recup_heure(Cpu::Matrix<double>& X, int ligne_X, int deb_X, int deb, int number) {
        for (int i = 0; i < number; i++) {
            int j = i + deb;
            int k = i + deb_X;
            X.set(ligne_X, k, (double)time(j).heure());
        }
    }
    void recup_minutes(Cpu::Matrix<double>& X, int ligne_X, int deb_X, int deb, int number) {
        for (int i = 0; i < number; i++) {
            int j = i + deb;
            int k = i + deb_X;
            X.set(ligne_X, k, (double)time(j).minutes());
        }
    }
    void recup_jour_semaine(Cpu::Matrix<double>& X, int ligne_X, int deb_X, int deb, int number) {
        for (int i = 0; i < number; i++) {
            int j = i + deb;
            int k = i + deb_X;
            X.set(ligne_X, k, (double)time(j).jour_semaine());
        }
    }

    void recup_temperature(Cpu::Matrix<double>& X, int ligne_X, int deb_X, int deb, int number, int ordre_derivation = 0) {

        for (int i = 0; i < number; i++) {
            int j = i + deb;
            int k = i + deb_X;
            X.set(ligne_X, k, get_ordre<double>(temperature_, j, ordre_derivation));
        }

    }
    void recup_humidite(Cpu::Matrix<double>& X, int ligne_X, int deb_X, int deb, int number, int ordre_derivation = 0) {

        for (int i = 0; i < number; i++) {
            int j = i + deb;
            int k = i + deb_X;
            X.set(ligne_X, k, get_ordre<double>(humidite_, j, ordre_derivation));
        }

    }
    void recup_nebulosite(Cpu::Matrix<double>& X, int ligne_X, int deb_X, int deb, int number, int ordre_derivation = 0) {

        for (int i = 0; i < number; i++) {
            int j = i + deb;
            int k = i + deb_X;
            X.set(ligne_X, k, get_ordre<double>(nebulosite_, j, ordre_derivation));
        }

    }
    void recup_pluviometrie(Cpu::Matrix<double>& X, int ligne_X, int deb_X, int deb, int number, int ordre_derivation = 0) {

        for (int i = 0; i < number; i++) {
            int j = i + deb;
            int k = i + deb_X;
            X.set(ligne_X, k, get_ordre<double>(pluviometrie_, j, ordre_derivation));
        }

    }
    void recup_consommation(Cpu::Matrix<double>& X, int ligne_X, int deb_X, int deb, int number, int ordre_derivation = 0) {

        for (int i = 0; i < number; i++) {
            int j = i + deb;
            int k = i + deb_X;
            X.set(ligne_X, k, get_ordre<double>(consommation_, j, ordre_derivation));
        }

    }

    Cpu::MatrixPair<double> dataset(selection_donnees_t* selec) {

        int size = selec->nb_donnees();
        int nb_derniere_conso = selec->nb_derniere_conso;
        int nb_sample = taille() - nb_derniere_conso;

        // Matrices temporaires
        Cpu::Matrix<double> X(size, nb_sample);
        Cpu::Matrix<double> Y(1, nb_sample);


        // Remplissage Y et vecteur_date_heure

        recup_consommation(Y, 0, 0, nb_derniere_conso, nb_sample, selec->ordre_derivation_conso);

        // Remplissage X

        int c = 0;

        if (selec->biais) {
            for (int i = 0; i < nb_sample; i++)
            {
                X.set(c, i, (double)1.f);
            }
            c++;
        }

        if (selec->annee) {
            recup_annee(X, c, 0, nb_derniere_conso, nb_sample);
            c++;
        }
        if (selec->moi) {
            recup_moi(X, c, 0, nb_derniere_conso, nb_sample);
            c++;
        }
        if (selec->jour) {
            recup_jour(X, c, 0, nb_derniere_conso, nb_sample);
            c++;
        }
        if (selec->heure) {
            recup_heure(X, c, 0, nb_derniere_conso, nb_sample);
            c++;
        }
        if (selec->minutes) {
            recup_minutes(X, c, 0, nb_derniere_conso, nb_sample);
            c++;
        }
        if (selec->jour_semaine) {
            recup_jour_semaine(X, c, 0, nb_derniere_conso, nb_sample);
            c++;
        }

        if (selec->temperature) {
            recup_temperature(X, c, 0, nb_derniere_conso, nb_sample, 0);
            c++;
        }
        if (selec->derivation_temperature) {
            recup_temperature(X, c, 0, nb_derniere_conso, nb_sample, 1);
            c++;
        }
        if (selec->humidite) {
            recup_humidite(X, c, 0, nb_derniere_conso, nb_sample, 0);
            c++;
        }
        if (selec->derivation_humidite) {
            recup_humidite(X, c, 0, nb_derniere_conso, nb_sample, 1);
            c++;
        }
        if (selec->nebulosite) {
            recup_nebulosite(X, c, 0, nb_derniere_conso, nb_sample, 0);
            c++;
        }
        if (selec->derivation_nebulosite) {
            recup_nebulosite(X, c, 0, nb_derniere_conso, nb_sample, 1);
            c++;
        }
        if (selec->pluviometrie) {
            recup_pluviometrie(X, c, 0, nb_derniere_conso, nb_sample, 0);
            c++;
        }
        if (selec->derivation_pluviometrie) {
            recup_pluviometrie(X, c, 0, nb_derniere_conso, nb_sample, 1);
            c++;
        }

        for (int k = 0; k < nb_derniere_conso; k++)
        {
            recup_consommation(X, c, 0, k, nb_sample, selec->ordre_derivation_conso);
            c++;
        }

        return { X, Y };

    }

    /*dataset_t dataset(selection_donnees_t* selec) {

        int size = selec->nb_donnees();
        int nb_derniere_conso = selec->nb_derniere_conso;
        int nb_sample = taille() - nb_derniere_conso;

        // Matrices temporaires
        Cpu::Matrix<double> X(size, nb_sample);
        Cpu::Matrix<double> Y(1, nb_sample);


        // Remplissage Y et vecteur_date_heure

        for (int i = 0; i < nb_sample; i++) {
            int j = i + nb_derniere_conso;
            Y.set(0, i, (double)consommation(j));
        }

        // Remplissage X

        int c = 0;

        if (selec->biais) {
            for (int i = 0; i < nb_sample; i++)
            {
                X.set(c, i, (double)1.f);
            }
            c++;
        }

        if (selec->annee) {
            for (int i = 0; i < nb_sample; i++)
            {
                int j = i + nb_derniere_conso;
                X.set(c, i, (double)time(j).annee());
            }
            c++;
        }
        if (selec->moi) {
            for (int i = 0; i < nb_sample; i++)
            {
                int j = i + nb_derniere_conso;
                X.set(c, i, (double)time(j).moi());
            }
            c++;
        }
        if (selec->jour) {
            for (int i = 0; i < nb_sample; i++)
            {
                int j = i + nb_derniere_conso;
                X.set(c, i, (double)time(j).jour());
            }
            c++;
        }
        if (selec->heure) {
            for (int i = 0; i < nb_sample; i++)
            {
                int j = i + nb_derniere_conso;
                X.set(c, i, (double)time(j).heure());
            }
            c++;
        }
        if (selec->minutes) {
            for (int i = 0; i < nb_sample; i++)
            {
                int j = i + nb_derniere_conso;
                X.set(c, i, (double)time(j).minutes());
            }
            c++;
        }
        if (selec->jour_semaine) {
            for (int i = 0; i < nb_sample; i++)
            {
                int j = i + nb_derniere_conso;
                X.set(c, i, (double)time(j).jour_semaine());
            }
            c++;
        }

        for (int k = 1; k <= nb_derniere_conso; k++)
        {
            for (int i = 0; i < nb_sample; i++)
            {
                int j = i + nb_derniere_conso;
                X.set(c, i, (double)consommation(j - k));
            }
            c++;
        }

        if (selec->temperature) {
            for (int i = 0; i < nb_sample; i++)
            {
                int j = i + nb_derniere_conso;
                X.set(c, i, (double)temperature(j));
            }
            c++;
        }
        if (selec->humidite) {
            for (int i = 0; i < nb_sample; i++)
            {
                int j = i + nb_derniere_conso;
                X.set(c, i, (double)humidite(j));
            }
            c++;
        }
        if (selec->nebulosite) {
            for (int i = 0; i < nb_sample; i++)
            {
                int j = i + nb_derniere_conso;
                X.set(c, i, (double)nebulosite(j));
            }
            c++;
        }
        if (selec->pluviometrie) {
            for (int i = 0; i < nb_sample; i++)
            {
                int j = i + nb_derniere_conso;
                X.set(c, i, (double)pluviometrie(j));
            }
            c++;
        }

        dataset_t d(X, Y);
        return d;

    }*/
};







struct selection_donnees_2_t /* Version de selection_donnees_t avec améliorations (utilisé par Donnees_2) */ {

    bool biais{ false };

    // Données temporelles

    bool annee{ false };
    bool mois{ false };
    bool mois_2{ false }; // 12 paramètres : tous fixés à 0 sauf celui du mois mis à 1
    bool jour{ false };
    bool heure{ false };
    bool minutes{ false };
    bool jour_semaine{ false }; // 7 paramètres : tous fixés à 0 sauf celui du jour de la semaine mis à 1

    // Données météo
    
    int nb_temperature{ 0 };
    int nb_humidite{ 0 };
    int nb_nebulosite{ 0 };

    int nb_variation_temperature{ 0 };
    int nb_variation_humidite{ 0 };
    int nb_variation_nebulosite{ 0 };

    // Données de consommation

    
    int nb_conso{ 0 }; // Le nombre des dernière consommations prises en compte en entree
    int nb_variation_conso{ 0 };


    // Methodes

    int nb_donnees_entrees() {

        int n = 0;

        if (biais) n++;
        if (annee) n++;
        if (mois) n++;
        if (mois_2) n+=12;
        if (jour) n++;
        if (heure) n++;
        if (minutes) n++;
        if (jour_semaine) n+=7;

        n += nb_temperature;
        n += nb_humidite;
        n += nb_nebulosite;

        n += nb_variation_temperature;
        n += nb_variation_humidite;
        n += nb_variation_nebulosite;

        n += nb_conso;
        n += nb_variation_conso;

        return n;

    }

    int max_nb_par_var() {

        int res = -1;

        if (nb_temperature > res) res = nb_temperature;
        if (nb_humidite > res) res = nb_humidite;
        if (nb_nebulosite > res) res = nb_nebulosite;
        if (nb_variation_temperature > res) res = nb_variation_temperature;
        if (nb_variation_humidite > res) res = nb_variation_humidite;
        if (nb_variation_nebulosite > res) res = nb_variation_nebulosite;
        if (nb_conso > res) res = nb_conso;
        if (nb_variation_conso > res) res = nb_variation_conso;

        return res;
    }

};

class Donnees_2 /* Version de Donnees mais avec des donnees normaliser et quelques modifications */ {

private:

    BaseDeDonnees* bdd_;
    context_donnees_t* context_;

    std::string requete_;
    std::vector<Date> date_heure_;
    std::vector<double> consommation_;
    std::vector<double> temperature_;
    std::vector<double> humidite_;
    std::vector<double> nebulosite_;
    double consommation_moyenne_{ 0. };
    double consommation_ecart_type_{ 0. };


public:

    Donnees_2(BaseDeDonnees* bdd, context_donnees_t* context) : bdd_(bdd), context_(context)
    {}

    void creer_requete() {

        std::string ville;
        if (context_->lieu_conso == "grand_lyon") { ville = "lyon"; }
        else { ville = "paris"; }

        std::string table = "normalise_" + ville;

        Date debut = context_->debut;
        Date fin = context_->fin;
        std::string begin = std::to_string(debut.vers_code());
        std::string end = std::to_string(fin.vers_code());

        requete_ = "";
        requete_ += "SELECT t.date_heure, t.conso, t.temperature, t.humidite, t.nebulosite ";
        requete_ += "FROM " + table + " AS t ";
        requete_ += "WHERE t.date_heure >= " + begin + " AND t.date_heure <= " + end + ";";

        // Récupération de la moyenne et de l'ecart type pour retrouver la consommation reelle en MW
        int code_ret;
        auto rq = bdd_->creer_requete("select avg_conso from moyennes_" + ville + ";");
        if (SQLITE_ROW == (code_ret = sqlite3_step(rq))) consommation_moyenne_ = sqlite3_column_double(rq, 0);
        code_ret = sqlite3_step(rq);
        if (code_ret != SQLITE_DONE) {
            std::cout << "Erreur de d'execution : " << sqlite3_errstr(code_ret) << std::endl;
            assert(0);
        }

        auto rq1 = bdd_->creer_requete("select sig_conso from ecarts_types_" + ville + ";");
        if (SQLITE_ROW == (code_ret = sqlite3_step(rq1))) consommation_ecart_type_ = sqlite3_column_double(rq1, 0);
        code_ret = sqlite3_step(rq1);
        if (code_ret != SQLITE_DONE) {
            std::cout << "Erreur de d'execution : " << sqlite3_errstr(code_ret) << std::endl;
            assert(0);
        }

        sqlite3_finalize(rq);
        sqlite3_finalize(rq1);

    }
    void recuperer_donnees() {

        int code_ret;
        auto rq = bdd_->creer_requete(requete_);

        while (SQLITE_ROW == (code_ret = sqlite3_step(rq))) {
            date_heure_.push_back(Date((long long)sqlite3_column_int64(rq, 0)));
            consommation_.push_back(sqlite3_column_double(rq, 1));
            temperature_.push_back(sqlite3_column_double(rq, 2));
            humidite_.push_back(sqlite3_column_double(rq, 3));
            nebulosite_.push_back(sqlite3_column_double(rq, 4));
        }

        if (code_ret != SQLITE_DONE) {
            std::cout << "Erreur de d'execution : " << sqlite3_errstr(code_ret) << std::endl;
            assert(0);
        }

        sqlite3_finalize(rq);

    }

    void vider() {

        date_heure_.clear();
        temperature_.clear();
        humidite_.clear();
        nebulosite_.clear();

    }

    context_donnees_t* context() {
        return context_;
    }
    int taille() {
        return date_heure_.size();
    }

    double consommation_moyenne() const {
        return consommation_moyenne_;
    }
    double consommation_ecart_type() const {
        return consommation_ecart_type_;
    }
    double convert_conso_to_MW(double conso) {
        return conso * consommation_ecart_type_ + consommation_moyenne_;
    }

    Date time(int i) {
        return date_heure_[i];
    }
    double consommation(int i) {
        return consommation_[i];
    }
    double temperature(int i) {
        return temperature_[i];
    }
    double humidite(int i) {
        return humidite_[i];
    }
    double nebulosite(int i) {
        return nebulosite_[i];
    }


    std::vector<Date> times(int deb = -1, int fin = -1) {

        if (deb == -1) { deb = 0; }
        if (fin == -1) { fin = date_heure_.size(); }

        std::vector<Date> res;
        for (int i = deb; i < fin; i++) {
            res.push_back(time(i));
        }

        return res;
    }
    std::vector<double> consommations(int deb = -1, int fin = -1) {

        if (deb == -1) { deb = 0; }
        if (fin == -1) { fin = date_heure_.size(); }

        std::vector<double> res;
        for (int i = deb; i < fin; i++) {
            res.push_back(consommation(i));
        }

        return res;
    }
    std::vector<double> temperatures(int deb = -1, int fin = -1) {

        if (deb == -1) { deb = 0; }
        if (fin == -1) { fin = date_heure_.size(); }

        std::vector<double> res;
        for (int i = deb; i < fin; i++) {
            res.push_back(temperature(i));
        }

        return res;
    }
    std::vector<double> humidites(int deb = -1, int fin = -1) {

        if (deb == -1) { deb = 0; }
        if (fin == -1) { fin = date_heure_.size(); }

        std::vector<double> res;
        for (int i = deb; i < fin; i++) {
            res.push_back(humidite(i));
        }

        return res;
    }
    std::vector<double> nebulosites(int deb = -1, int fin = -1) {

        if (deb == -1) { deb = 0; }
        if (fin == -1) { fin = date_heure_.size(); }

        std::vector<double> res;
        for (int i = deb; i < fin; i++) {
            res.push_back(nebulosite(i));
        }

        return res;
    }

    void recup_annee(Cpu::Matrix<double>& X, int ligne_X, int deb_X, int deb, int number) {
        for (int i = 0; i < number; i++) {
            int j = i + deb;
            int k = i + deb_X;
            X.set(ligne_X, k, (double)time(j).annee());
        }
    }
    void recup_moi(Cpu::Matrix<double>& X, int ligne_X, int deb_X, int deb, int number) {
        for (int i = 0; i < number; i++) {
            int j = i + deb;
            int k = i + deb_X;
            X.set(ligne_X, k, (double)time(j).moi());
        }
    }
    void recup_jour(Cpu::Matrix<double>& X, int ligne_X, int deb_X, int deb, int number) {
        for (int i = 0; i < number; i++) {
            int j = i + deb;
            int k = i + deb_X;
            X.set(ligne_X, k, (double)time(j).jour());
        }
    }
    void recup_heure(Cpu::Matrix<double>& X, int ligne_X, int deb_X, int deb, int number) {
        for (int i = 0; i < number; i++) {
            int j = i + deb;
            int k = i + deb_X;
            X.set(ligne_X, k, (double)time(j).heure()/24.0);
        }
    }
    void recup_minutes(Cpu::Matrix<double>& X, int ligne_X, int deb_X, int deb, int number) {
        for (int i = 0; i < number; i++) {
            int j = i + deb;
            int k = i + deb_X;
            X.set(ligne_X, k, (double)time(j).minutes());
        }
    }
    void recup_jour_semaine(Cpu::Matrix<double>& X, int ligne_X, int deb_X, int deb, int number) {
        for (int i = 0; i < number; i++) {
            int j = i + deb;
            int k = i + deb_X;
            for (int w = 0;  w < 7;  w++)
            {
                X.set(ligne_X + w, k, 0.0);
            }
            X.set(ligne_X + time(j).jour_semaine(), k, 0.1);
        }
    }
    void recup_moi_2(Cpu::Matrix<double>& X, int ligne_X, int deb_X, int deb, int number) {
        for (int i = 0; i < number; i++) {
            int j = i + deb;
            int k = i + deb_X;
            for (int w = 0; w < 12; w++)
            {
                X.set(ligne_X + w, k, 0.0);
            }
            X.set(ligne_X + time(j).moi()-1, k, 0.1);
        }
    }

    void recup_temperature(Cpu::Matrix<double>& X, int ligne_X, int deb_X, int deb, int number, int ordre_derivation = 0) {

        for (int i = 0; i < number; i++) {
            int j = i + deb;
            int k = i + deb_X;
            X.set(ligne_X, k, get_ordre<double>(temperature_, j, ordre_derivation));
        }

    }
    void recup_humidite(Cpu::Matrix<double>& X, int ligne_X, int deb_X, int deb, int number, int ordre_derivation = 0) {

        for (int i = 0; i < number; i++) {
            int j = i + deb;
            int k = i + deb_X;
            X.set(ligne_X, k, get_ordre<double>(humidite_, j, ordre_derivation));
        }

    }
    void recup_nebulosite(Cpu::Matrix<double>& X, int ligne_X, int deb_X, int deb, int number, int ordre_derivation = 0) {

        for (int i = 0; i < number; i++) {
            int j = i + deb;
            int k = i + deb_X;
            X.set(ligne_X, k, get_ordre<double>(nebulosite_, j, ordre_derivation));
        }

    }
    void recup_consommation(Cpu::Matrix<double>& X, int ligne_X, int deb_X, int deb, int number, int ordre_derivation = 0) {

        for (int i = 0; i < number; i++) {
            int j = i + deb;
            int k = i + deb_X;
            X.set(ligne_X, k, get_ordre<double>(consommation_, j, ordre_derivation));
        }

    }

    Cpu::MatrixPair<double> dataset(selection_donnees_2_t* selec) {

        int size = selec->nb_donnees_entrees();
        int max = selec->max_nb_par_var();
        int N = taille() - max;

        // Matrices temporaires
        Cpu::Matrix<double> X(size, N);
        Cpu::Matrix<double> Y(1, N);


        // Remplissage Y et vecteur_date_heure

        // On travaille toujours à trouver la dérivée de la consommation
        recup_consommation(Y, 0, 0, max, N, 1);

        // Remplissage X

        int c = 0;

        if (selec->biais) {
            for (int i = 0; i < N; i++)
            {
                X.set(c, i, (double)1.f);
            }
            c++;
        }


        if (selec->annee) {
            recup_annee(X, c, 0, max, N);
            c++;
        }
        if (selec->mois) {
            recup_moi(X, c, 0, max, N);
            c++;
        }
        if (selec->mois_2) {
            recup_moi_2(X, c, 0, max, N);
            c+=12;
        }
        if (selec->jour) {
            recup_jour(X, c, 0, max, N);
            c++;
        }
        if (selec->heure) {
            recup_heure(X, c, 0, max, N);
            c++;
        }
        if (selec->minutes) {
            recup_minutes(X, c, 0, max, N);
            c++;
        }
        if (selec->jour_semaine) {
            recup_jour_semaine(X, c, 0, max, N);
            c+=7;
        }



        for (int k = max-selec->nb_temperature; k < max; k++)
        {
            recup_temperature(X, c, 0, k, N, 0);
            c++;
        }
        for (int k = max - selec->nb_humidite; k < max; k++)
        {
            recup_humidite(X, c, 0, k, N, 0);
            c++;
        }
        for (int k = max - selec->nb_nebulosite; k < max; k++)
        {
            recup_nebulosite(X, c, 0, k, N, 0);
            c++;
        }



        for (int k = max - selec->nb_variation_temperature; k < max; k++)
        {
            recup_temperature(X, c, 0, k, N, 1);
            c++;
        }
        for (int k = max - selec->nb_variation_humidite; k < max; k++)
        {
            recup_humidite(X, c, 0, k, N, 1);
            c++;
        }
        for (int k = max - selec->nb_variation_nebulosite; k < max; k++)
        {
            recup_nebulosite(X, c, 0, k, N, 1);
            c++;
        }


        
        for (int k = max - selec->nb_conso; k < max; k++)
        {
            recup_consommation(X, c, 0, k, N, 0);
            c++;
        }
        for (int k = max - selec->nb_variation_conso; k < max; k++)
        {
            recup_consommation(X, c, 0, k, N, 1);
            c++;
        }

        return { X, Y };

    }

};

