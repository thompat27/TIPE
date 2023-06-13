#include "data.cuh"


// BaseDeDonnees /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

BaseDeDonnees::BaseDeDonnees(std::string bdd_path) {

    int code_ret;

    code_ret = sqlite3_open(bdd_path.c_str(), &BDD);

    if (code_ret) {
        fprintf(stderr, "Impossible d'ouvrir la base de donnees : %s\n", sqlite3_errmsg(BDD));
        sqlite3_close(BDD);
        assert(0);
    }

}

BaseDeDonnees:: ~BaseDeDonnees() {
    sqlite3_close(BDD);
}

sqlite3_stmt* BaseDeDonnees::creer_requete(std::string requete) {
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

// Date //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

Date::Date(int annee, int moi, int jour, int heure, int minutes) {
    fixer(annee, moi, jour, heure, minutes);
}

Date::Date(long long date_heure) {
    fixer(date_heure);
}

void Date::fixer(int annee, int moi, int jour, int heure, int minutes) {
    _annee = annee;
    _moi = moi;
    _jour = jour;
    _heure = heure;
    _minutes = minutes;
}

void Date::fixer(long long date_heure) {
    _annee = date_heure / 100000000;
    _moi = (date_heure / 1000000) % 100;
    _jour = (date_heure / 10000) % 100;
    _heure = (date_heure / 100) % 100;
    _minutes = date_heure % 100;
}

long long Date::vers_code() const {
    long long annee = _annee;
    long long moi = _moi;
    long long jour = _jour;
    long long heure = _heure;
    long long minutes = _minutes;
    return (((((((annee * 100LL) + moi) * 100LL) + jour) * 100LL) + heure) * 100LL) + minutes;
}

int Date::annee() const {
    return _annee;
}
int Date::moi() const {
    return _moi;
}
int Date::jour() const {
    return _jour;
}
int Date::heure() const {
    return _heure;
}
int Date::minutes() const {
    return _minutes;
}
int Date::jour_semaine() const {
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

std::ostream& operator<<(std::ostream& os, Date d)
{
    os << d.jour() << "/" << d.moi() << "/" << d.annee() << " " << d.heure() << ":" << d.minutes();
    return os;
}

// Donnees //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

context_donnees_t* Donnees::context() { return context_; }
selection_donnees_t* Donnees::selection() { return selection_; }

Donnees::Donnees(BaseDeDonnees* bdd, context_donnees_t* context, selection_donnees_t* selection, pretraiteur_donnees_t* pretraiteur) : bdd_(bdd), context_(context), selection_(selection), pretraiteur_(pretraiteur) {}

void Donnees::vider() {
    date_heure_.clear();
    temperature_.clear();
    humidite_.clear();
    nebulosite_.clear();
    pluviometrie_.clear();
}

void Donnees::creer_requete() {

    std::string lieu_conso = context_->lieu_conso;
    int id_station = context_->id_station;
    Date debut = context_->debut;
    Date fin = context_->fin;

    bool temperature = selection_->temperature;
    bool humidite = selection_->humidite;
    bool nebulosite = selection_->nebulosite;
    bool pluviometrie = selection_->pluviometrie;

    requete_ = "SELECT consommation.date_heure, consommation.conso";
    if (temperature)
    {
        requete_ += ", meteo.temperature";
    }
    if (humidite) {
        requete_ += ", meteo.humidite";
    }
    if (nebulosite) {
        requete_ += ", meteo.nebulosite";
    }
    if (pluviometrie) {
        requete_ += ", meteo.pluviometrie";
    }

    requete_ += " FROM consommation JOIN meteo ON meteo.date_heure >= consommation.date_heure - 130 AND meteo.date_heure < consommation.date_heure + 170 WHERE ";
    //_requete += " FROM consommation JOIN meteo ON meteo.date_heure = consommation.date_heure WHERE ";

    requete_ += "consommation.date_heure >= " + std::to_string(debut.vers_code()) + " AND consommation.date_heure <= " + std::to_string(fin.vers_code()) + " ";
    requete_ += "AND consommation.lieu = \"" + lieu_conso + "\" AND meteo.id_station=" + std::to_string(id_station) + " AND consommation.conso IS NOT NULL ";
    if (temperature) {
        requete_ += "AND meteo.temperature IS NOT NULL ";
    }
    if (humidite) {
        requete_ += "AND meteo.humidite IS NOT NULL ";
    }
    if (nebulosite) {
        requete_ += "AND meteo.nebulosite IS NOT NULL ";
    }
    if (pluviometrie) {
        requete_ += "AND meteo.pluviometrie IS NOT NULL ";
    }
    requete_ += ";";

}
void Donnees::recuperer_donnees() {

    int code_ret;

    auto rq = bdd_->creer_requete(requete_);

    for (int i = 0; SQLITE_ROW == (code_ret = sqlite3_step(rq)); i++)
    {
        if (i % selection_->nb_quart_heure_intervalle == 0) {
            date_heure_.push_back(Date((long long)sqlite3_column_int64(rq, 0)));

            consommation_.push_back(sqlite3_column_double(rq, 1));

            if (selection_->temperature) {
                temperature_.push_back(sqlite3_column_double(rq, 2));
            }

            if (selection_->humidite) {
                humidite_.push_back(sqlite3_column_double(rq, 3));
            }

            if (selection_->nebulosite) {
                nebulosite_.push_back(sqlite3_column_double(rq, 4));
            }

            if (selection_->pluviometrie) {
                pluviometrie_.push_back(sqlite3_column_double(rq, 5));
            }
        }
    }

    if (code_ret != SQLITE_DONE) {
        std::cout << "Erreur de d'execution : " << sqlite3_errstr(code_ret) << std::endl;
        assert(0);
    }

    sqlite3_finalize(rq);
}

void Donnees::ecrire() {
    for (int i = 0; i < date_heure_.size(); i++)
    {
        std::cout << date_heure_[i].annee() << "/" << date_heure_[i].moi() << "/" << date_heure_[i].jour() << " " << date_heure_[i].heure() << ":" << date_heure_[i].minutes() << " -- " << consommation_[i];
        if (selection_->temperature)
        {
            std::cout << " -- " << temperature_[i];
        }
        if (selection_->humidite)
        {
            std::cout << " -- " << humidite_[i];
        }
        if (selection_->nebulosite)
        {
            std::cout << " -- " << nebulosite_[i];
        }
        if (selection_->pluviometrie)
        {
            std::cout << " -- " << pluviometrie_[i];
        }
        std::cout << std::endl;
    }
}
void Donnees::ecrire_dans_fichier() {

    std::ofstream fichier("affichage/graphe.csv");

    for (int i = 0; i < date_heure_.size(); i++)
    {
        std::cout << date_heure_[i].jour() << "/" << date_heure_[i].moi() << "/" << date_heure_[i].annee() << " " << date_heure_[i].heure() << ":" << date_heure_[i].minutes() << ";" << date_heure_[i].annee() << ";" << date_heure_[i].moi() << ";" << date_heure_[i].jour() << ";" << date_heure_[i].heure() << ";" << date_heure_[i].minutes() << ";" << consommation_[i];
        if (selection_->temperature)
        {
            std::cout << ";" << temperature_[i];
        }
        if (selection_->humidite)
        {
            std::cout << ";" << humidite_[i];
        }
        if (selection_->nebulosite)
        {
            std::cout << ";" << nebulosite_[i];
        }
        if (selection_->pluviometrie)
        {
            std::cout << ";" << pluviometrie_[i];
        }
        std::cout << std::endl;
    }

    fichier.close();

}

Date Donnees::date_heure(int i) {
    return date_heure_[i];
}

double Donnees::consommation(int i) {
    return consommation_[i];
}
double Donnees::temperature(int i) {
    return temperature_[i];
}
double Donnees::humidite(int i) {
    return humidite_[i];
}
double Donnees::nebulosite(int i) {
    return nebulosite_[i];
}
double Donnees::pluviometrie(int i) {
    return pluviometrie_[i];
}
int Donnees::taille() {
    return date_heure_.size();
}

dataset_t Donnees::creer_matrice() {

    int size = selection()->nb_donnees();
    int nb_derniere_conso = selection()->nb_derniere_conso;
    int nb_sample = taille() - nb_derniere_conso;

    // Matrices temporaires
    Cpu::Matrix<double> X(size, nb_sample);
    Cpu::Matrix<double> Y(1, nb_sample);


    // Remplissage Y et vecteur_date_heure

    for (int i = 0; i < nb_sample; i++) {
        int j = i + nb_derniere_conso;
        Y.set(0, i, (double)consommation(j) / pretraiteur_->conso);
    }

    // Remplissage X

    int c = 0;

    if (selection()->biais) {
        for (int i = 0; i < nb_sample; i++)
        {
            X.set(c, i, (double)1.f / pretraiteur_->biais);
        }
        c++;
    }

    if (selection()->annee) {
        for (int i = 0; i < nb_sample; i++)
        {
            int j = i + nb_derniere_conso;
            X.set(c, i, (double)date_heure(j).annee() / pretraiteur_->annee);
        }
        c++;
    }
    if (selection()->moi) {
        for (int i = 0; i < nb_sample; i++)
        {
            int j = i + nb_derniere_conso;
            X.set(c, i, (double)date_heure(j).moi() / pretraiteur_->moi);
        }
        c++;
    }
    if (selection()->jour) {
        for (int i = 0; i < nb_sample; i++)
        {
            int j = i + nb_derniere_conso;
            X.set(c, i, (double)date_heure(j).jour() / pretraiteur_->jour);
        }
        c++;
    }
    if (selection()->heure) {
        for (int i = 0; i < nb_sample; i++)
        {
            int j = i + nb_derniere_conso;
            X.set(c, i, (double)date_heure(j).heure() / pretraiteur_->heure);
        }
        c++;
    }
    if (selection()->minutes) {
        for (int i = 0; i < nb_sample; i++)
        {
            int j = i + nb_derniere_conso;
            X.set(c, i, (double)date_heure(j).minutes() / pretraiteur_->minutes);
        }
        c++;
    }
    if (selection()->jour_semaine) {
        for (int i = 0; i < nb_sample; i++)
        {
            int j = i + nb_derniere_conso;
            X.set(c, i, (double)date_heure(j).jour_semaine() / pretraiteur_->jour_semaine);
        }
        c++;
    }

    for (int k = 1; k <= nb_derniere_conso; k++)
    {
        for (int i = 0; i < nb_sample; i++)
        {
            int j = i + nb_derniere_conso;
            X.set(c, i, (double)consommation(j - k) / pretraiteur_->conso);
        }
        c++;
    }

    if (selection()->temperature) {
        for (int i = 0; i < nb_sample; i++)
        {
            int j = i + nb_derniere_conso;
            X.set(c, i, (double)temperature(j) / pretraiteur_->temperature);
        }
        c++;
    }
    if (selection()->humidite) {
        for (int i = 0; i < nb_sample; i++)
        {
            int j = i + nb_derniere_conso;
            X.set(c, i, (double)humidite(j) / pretraiteur_->humidite);
        }
        c++;
    }
    if (selection()->nebulosite) {
        for (int i = 0; i < nb_sample; i++)
        {
            int j = i + nb_derniere_conso;
            X.set(c, i, (double)nebulosite(j) / pretraiteur_->nebulosite);
        }
        c++;
    }
    if (selection()->pluviometrie) {
        for (int i = 0; i < nb_sample; i++)
        {
            int j = i + nb_derniere_conso;
            X.set(c, i, (double)pluviometrie(j) / pretraiteur_->pluviometrie);
        }
        c++;
    }

    dataset_t d(X, Y);
    return d;
}

