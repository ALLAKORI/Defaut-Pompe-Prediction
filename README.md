# 🔧 Prédiction de Défauts de Pompe Électrique - Machine Learning

Ce projet est une application de maintenance prédictive utilisant le machine learning pour détecter les défauts potentiels dans une pompe électrique industrielle. L'interface graphique est développée avec `ttkbootstrap` pour une esthétique moderne et sombre, et le modèle de prédiction est basé sur des données techniques de capteurs.

---

## 📌 Fonctionnalités

- Interface utilisateur intuitive avec thème sombre (`solar`)
- Saisie manuelle ou aléatoire des données capteurs
- Prédiction du type de défaut (ex : surcharge mécanique, cavitation, etc.)
- Affichage du niveau de risque, du conseil technique et de la confiance du modèle
- Utilisation de modèles supervisés (classification)

---

## 🖥️ Technologies utilisées

- **Python 3**
- **ttkbootstrap** (interface graphique)
- **pandas** (gestion des données)
- **scikit-learn** (modèle de machine learning)
- **Tkinter** (base GUI)

---

## 📊 Données d'entrée

Les paramètres techniques pris en compte incluent :

- Voltmetre (V)
- Puissance (kW)
- Vitesse (tr/min)
- Températures des roulements et bobines (PT100)
- Position de l’arbre (mm)
- Pression de refoulement (bar)
- Débit (m³/h)

---

## 🚀 Lancer l'application

```bash
python app.py
```
Assurez-vous d’avoir installé les dépendances :
```bash
pip install ttkbootstrap pandas scikit-learn
```
## 📁 Structure du projet
```bash
├── app.py                  # Code principal de l'application
├── model.pkl               # Modèle entraîné (à ajouter)
├── README.md               # Ce fichier
├── Projet Machine Learning.pdf  # Rapport du projet
```
---
## 📦 À venir

- Intégration d’un vrai modèle entraîné (model.pkl)
- Ajout de la validation croisée et des métriques ROC/PR
- Export des résultats en PDF ou CSV
- Version cloud via Streamlit ou Flask
---

## 🧠 Auteur

ALLADO KOSSI RICHARD Étudiant ingénieur en cybersécurité et intelligence artificielle ENSA Beni Mellal – 2025
