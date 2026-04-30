# Sujet : Prédiction du nombre de vélos qui passent quotidiennement dans Paris

Exemple de problème business qu'on pourrait résoudre :

Où construire de nouvelles pistes cyclables dans Paris ?
En prédisant le trafic vélo futur, on peut identifier les zones à forte croissance et prioriser les investissements


1. Problématique
Décrivez votre projet simplement :
Que cherchez-vous à prédire ou analyser ?
Quel est l’objectif métier ou réel ?
*Ici:*

Prédire le nombre de vélos passant à un point donné à Paris

2. Formulation ML

Type de problème : classification / régression / autre
Variable cible :
Variables explicatives (idée initiale) :
Objectif d’évaluation (métrique ou critère) :

*Ici:*

**Problème de régression**
Variable cible : bike_count --> pour plus de précision, on va transformer la variable cible en log(bike_count)

**Features**
* température
* pluie
* jour de la semaine
* autre features à définir

**Objectif**
Minimiser l’erreur de prédiction (MAE / RMSE)

# Choix des données

1. Dataset principal
Indiquez :
Nom du dataset :
Source (URL) :
Type (tabulaire, time series, etc.) :
Description des variables principales :
Variable cible :

Types de problèmes éventuellement rencontrés:
i. sur le dataset
* NaNs, outliers, bruit
* Distribution imbalance, drift
ii. sur les features
* mauvais encodage, corrélation
* Temps leakage, non-stationnarité
iii. lors de la Collecte
* biais, données manquantes

*Ici:*

Nom : Comptage vélo Paris
Source : Open Data Paris - [Kaggle](https://www.kaggle.com/competitions/mdsb-2023/overview), n'existe malheureusement plus

Type : séries temporelles

Variables :
date / heure
identifiant du compteur
nombre de vélos

Variable cible : log(nombre de vélos par jour)

Qualité des données:

* compteur cassé → NaN
* pluie rare → imbalance
* COVID → data drift
* météo mal alignée → mauvais merge

2. Sources de données complémentaires
Vous devez inclure au moins une source externe !
Exemples :
météo
transports / grèves
événements
jours fériés
Pour chaque source il faut :
* Nom de la source :
* URL :
* Type (API / dataset / scraping) :
Variables envisagées :
Pourquoi cette source est pertinente :

*Ici:*

Source : météo (Open-Meteo)
URL : [lien url](https://open-meteo.com/en/docs/meteofrance-api)
Type : API
Variables :
* température
* précipitations
Pertinence :
On pense que la météo influence directement l’usage du vélo (si il pleut, on fait moins de vélo)

# Stratégie de collecte

1. Approche choisie
Pour chaque source :
Source	Méthode	Outil

*Ici:*

Dataset de départ : dataset bikecounter, obtenu par téléchargement directement sur le github indiqué
Dataset météo : historique Météo, obtenu sur le site météofrance avec la librairie	aiohttp
Dataset Grèves : scraping avec les libraries bs4 / selenium

2. Justification
Expliquez vos choix :
Pourquoi cette source ?
Pourquoi cette méthode (API vs scraping vs dataset) ?
Quels sont les avantages / limites ?

*Ici:*
L'API météo permet de recueillir des données propres et structurées
Son utilisation est facile à automatiser, reproductible, et permet d'éviter un scraping moins robuste, dépendant d'un html changeant

3. Contexte technique (statique vs dynamique)
Si vous utilisez du scraping :
La page est-elle statique ou dynamique ?
Quel outil est adapté ?

*Ici:*

Utiliser l'API météo ne relève pas d'une stratégie de scraping

Si on avait voulu recueillir des données sur des grèves, à partir du site internet lemonde par exemple, on aurait utiliser selenium/playwright

# Première collecte de données

1. Implémentation
Vous devez :
créer un script ou notebook qui :
charge le dataset principal
récupère au moins une source externe

Structure recommandée :
data/
  raw/
  processed/

scripts/
  data_collection.py

notebooks/
  data_exploration.ipynb

2. Exigences minimales
Votre code doit :
fonctionner correctement
afficher un aperçu (head)
sauvegarder les données en local (csv)
être reproductible

*Ici:*
```py
import requests
import pandas as pd

url = "https://archive-api.open-meteo.com/v1/archive"

params = {
    "latitude": 48.8566,
    "longitude": 2.3522,
    "start_date": "2022-01-01",
    "end_date": "2022-01-31",
    "daily": "temperature_2m_mean,precipitation_sum",
    "timezone": "Europe/Paris"
}

response = requests.get(url, params=params)
data = response.json()

weather_df = pd.DataFrame({
    "date": data["daily"]["time"],
    "temperature_mean": data["daily"]["temperature_2m_mean"],
    "precipitation_sum": data["daily"]["precipitation_sum"]
})
```

# Exploration des données

Présentez :
nombre de lignes / colonnes
noms des colonnes
types de variables
premières lignes

Exemple ici dans le notebook data_exploration :
* données journalières
* variables météo + comptage vélo
* Quelques visualisations simples

# Reproductibilité et éthique

1. Provenance
Pour chaque dataset :
d’où viennent les données ?
sont-elles accessibles publiquement ?
2. Reproductibilité
Expliquez :
comment relancer la collecte
quelles librairies sont utilisées
3. Éthique et utilisation
Vérifiez :
conditions d’utilisation
limites d’API
légalité du scraping

# Intégration dans le repository
Votre repository doit contenir :
un dossier data/ avec les données (A ne pas push sur github !!)
un script ou notebook de collecte
un fichier requirements.txt à jour
Pour le mettre à jour :
```py
pip freeze > requirements.txt
```