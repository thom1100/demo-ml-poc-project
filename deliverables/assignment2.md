# Assignment 2

## Data Cleaning

### Gestion des valeurs manquantes
Plusieurs colonnes du dataset contenaient des valeurs manquantes, notamment dans les données météorologiques.

**Méthodes utilisées :**
Suppression des lignes avec trop de valeurs manquantes

Choix d'Imputation :
Variables numériques manquantess remplacées par la moyenne ou la médiane selon les cas

Justification :
On veut éviter la perte excessive de données, le tout en préservant la distribution globale des variables

Alternatives testées :
Suppression complète des colonnes → rejetée pour éviter la perte d’information
Imputation par modèle → trop complexe pour ce projet, gain potentiellement faible par rapport à l'effort que cela demande

**Gestion des outliers**
Des valeurs extrêmes ont été observées sur certaines variables (ex : température, comptage vélo).

Détection via :
visualisation (boxplots)
seuils statistiques (IQR)

Traitement :

suppression des cas aberrants extrêmes --> par exemple les stations qui fermaient sur de longues périodes sont traitées de manière séparée

Justification :
Les modèles sensibles (ex : régression) peuvent être fortement impactés
Alternatives testées :
Aucun traitement → rejeté (dégradation des performances attendue)

**Types de données**

Conversion des colonnes en types appropriés :
dates → datetime
variables catégorielles → category

Justification :
Facilite le feature engineering
Évite les erreurs dans les transformations

## Feature Engineering
### Features temporelles
À partir de la variable de date, création de nouvelles features :
* heure
* jour de la semaine
* mois
* week-end vs semaine

Justification :
Le comportement (ex : trafic vélo) dépend fortement de la période considérée et de l'horaire de la journée

### Features météorologiques
À partir des données météo on a:
* normalisé la température
* créé un indicateur de pluie (binaire)

Justification :
La météo est un facteur clé explicatif

### Features dérivées
Création d'une moyenne glissante sur une semaine, qui ne prend pas en compte les jours suivants pour éviter le data leakage.
Utilisation des lags

Justification :
Capturer les effets temporels et la dynamique
Alternatives testées :
Transformée de Frourier → non retenues (hors scope POC)

## Transformations appliquées
### Scaling
StandardScaler sur variables numériques
Justification :
Nécessaire pour certains modèles (ex : régression, SVM)
Permet une convergence plus stable

### Encoding
One-hot encoding pour les variables catégorielles
Pourquoi :
Compatible avec la majorité des modèles
Évite l’ordre artificiel des catégories

Alternatives testées et rejetées
Target encoding du fait du risque de leakage
Ordinal encoding car One-Hot Encoding plus adapté

### Normalisation des distributions
On a utilisé le log pour transformer notre variable target

Pourquoi :
Réduire l’asymétrie
Améliorer la performance des modèles linéaires

### Choix et justification globale
Les transformations ont été choisies selon leur compatibilité avec plusieurs modèles ainsi que leur simplicité (POC) et interprétabilité

### Alternatives testées et rejetées
Technique envisagée et rejetée : Target encoding du fait d'un risque de leakage

### Impact attendu sur les modèles

Les transformations devraient permettre une meilleure convergence des modèles et une réduction du bruit

Impact par type de modèle :
Notre premier modèle linéaire voit ses performances fortement améliorés par scaling + log transform

### Organisation du code (data.py)
Le fichier src/data.py a été modifié pour :

charger les données brutes
appliquer les transformations
retourner un dataset prêt pour le ML

Pipeline typique :
load data
clean data
feature engineering
encoding / scaling
return X, y

## Datasets et notebooks associés à cet assignment
### Origine des datasets
Les datasets transformés ont été obtenus à partir de :
dataset brut (stocké dans data/raw/)
transformations réalisées dans :
meteorological_data_transformation.ipynb
feature_engineering.ipynb
first_model.ipynb

### Stockage
données brutes → data/raw/
données transformées → data/processed/

### Utilisation
Pour charger les données :
from src.data import load_data

X, y = load_data()