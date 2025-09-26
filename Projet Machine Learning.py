#!/usr/bin/env python
# coding: utf-8

# # PROJET DE PREDICTION DES DEFAUTS ELECTRIQUES  DES POMPES A HAUTES PRESSION

# # Importations des bibliothèques nécessaires

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import tkinter as tk
from tkinter import ttk
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
import random


# # Chargement des données

# In[2]:


df=pd.read_csv(r"C:\Users\Richard Kossi ALLADO\Desktop\data_capteurs_pompe_HP.csv",sep=";",header=0)


# # Informations génerales sur le dataframe 

# In[3]:


df.info()
print("Nombre de ligne:",len(df))


# # Affichage des cinqs premiers élements

# In[4]:


df.head()


# # Nombre d'occurence de defauts 

# In[5]:


defauts = df['Type_Défaut'].unique()
print(defauts,"\n le nombre de defauts egale a :",len(defauts))


# # Nombre d'occurence de chaque defauts 

# In[6]:


df['Type_Défaut'].value_counts()


# # Statistique descriptive 
# 

# In[7]:


df.describe()


# # Matrice de correlation des rapports entre les colonnes du dataframe

# In[8]:


df_cor=df.drop('Type_Défaut', axis=1)
correlations=df_cor.corr()

plt.figure(figsize=(20,10))
sns.heatmap(correlations,annot=True,cmap='coolwarm',square=True)
plt.title('Matrice de correlation')
plt.show()


# # gestion des valeurs manquantes

# In[9]:


df_1 = df.dropna(axis=0)
df_1.info() 
defauts=df_1['Type_Défaut'].unique()
print("\n les defauts sont:\n",defauts,"\n le nombre de defauts egale a:\n ",len(defauts))


# # Séparations des  features (X) de la partie  cible (y)

# In[10]:


X = df_1.drop('Type_Défaut', axis=1)
y = df_1['Type_Défaut']
X.columns


# # Encodage de  la variable cible

# In[11]:


# L'encodage permet a renommer les defauts selon des nombres 
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Création d'un dictionnaire de correspondance
mapping = {index: label for index, label in enumerate(le.classes_)}

print("y encodé :", y_encoded)
print("Correspondance :", mapping)


# # Séparation des data en  train/test

# In[12]:


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# # Entrainement de mon modele 

# In[13]:


model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


# # Prediction du modèle 

# In[14]:


y_pred = model.predict(X_test)


# # Evaluation du modèle 

# In[15]:


print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nRapport de classification :")
print(classification_report(y_test, y_pred, target_names=le.classes_))


# # Matrice de confusion 

# In[16]:


cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_,
            yticklabels=le.classes_)
plt.title("Matrice de confusion")
plt.xlabel("Prédictions")
plt.ylabel("Réel")
plt.show()


# # Importance des variables 

# In[17]:


importances = model.feature_importances_
features = X.columns
df_importance = pd.DataFrame({'Feature': features, 'Importance': importances})
df_importance = df_importance.sort_values(by='Importance', ascending=True)

plt.figure(figsize=(8,6))
plt.barh(df_importance['Feature'], df_importance['Importance'], color="teal")
plt.title("Importance des variables (Random Forest)")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.show()


# # Representation de la courbe ROC et de la courbe Precision - Recall 

# In[18]:


# Obtenir les probabilités de chaque classe
y_score = model.predict_proba(X_test)

# Liste des classes
classes = np.unique(y_test)
n_classes = len(classes)

# Binariser les labels
y_test_bin = label_binarize(y_test, classes=classes)

# ===============================
# ROC pour toutes les classes
# ===============================
plt.figure(figsize=(10,7))
for i in range(n_classes):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f'Classe {classes[i]} (AUC={roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=1.5)
plt.xlabel("Taux de faux positifs (FPR)")
plt.ylabel("Taux de vrais positifs (TPR)")
plt.title("Courbes ROC - Toutes les classes")
plt.legend(loc="lower right", fontsize=8)
plt.grid(True, linestyle=":")
plt.show()

# ===============================
# Précision-Rappel pour toutes les classes
# ===============================
plt.figure(figsize=(10,7))
for i in range(n_classes):
    precision, recall, _ = precision_recall_curve(y_test_bin[:, i], y_score[:, i])
    ap = average_precision_score(y_test_bin[:, i], y_score[:, i])
    plt.plot(recall, precision, lw=2, label=f'Classe {classes[i]} (AP={ap:.2f})')

plt.xlabel("Rappel")
plt.ylabel("Précision")
plt.title("Courbes Précision-Rappel - Toutes les classes")
plt.legend(loc="lower right", fontsize=8)
plt.grid(True, linestyle=":")
plt.show()


# # Testeur en entrant ses propres donnés

# In[19]:


# Simulation du modèle (à remplacer par ton vrai modèle)
# model = ...
# le = ...

champs = [
    "Voltmètre (V)", "Puissance (kW)", "Vitesse (tr/min)",
    "PT100_Roulement_Moteur_Avant (°C)", "PT100_Roulement_Moteur_Arrière (°C)",
    "PT100_Bobine_1 (°C)", "PT100_Bobine_2 (°C)", "PT100_Bobine_3 (°C)",
    "PT100_Roulement_Pompe_Avant (°C)", "PT100_Roulement_Pompe_Arrière (°C)",
    "Capteurs_Position_Arbre (mm)", "Pression_Refoulement (bar)", "Débit (m3/h)"
]

plages_valeurs = {
    "Voltmètre (V)": (340.0, 800.0),
    "Puissance (kW)": (220.0, 1000.0),
    "Vitesse (tr/min)": (2797.0, 3213.0),
    "PT100_Roulement_Moteur_Avant (°C)": (20.0, 130.0),
    "PT100_Roulement_Moteur_Arrière (°C)": (20.0, 130.0),
    "PT100_Bobine_1 (°C)": (20.0, 150.0),
    "PT100_Bobine_2 (°C)": (20.0, 150.0),
    "PT100_Bobine_3 (°C)": (20.0, 150.0),
    "PT100_Roulement_Pompe_Avant (°C)": (20.0, 130.0),
    "PT100_Roulement_Pompe_Arrière (°C)": (20.0, 130.0),
    "Capteurs_Position_Arbre (mm)": (0.05, 0.5),
    "Pression_Refoulement (bar)": (5.0, 25.0),
    "Débit (m3/h)": (200.0, 530.0)
}

conseils_defauts = {
    'Normal': "✅ Tout est normal. Continuez le suivi régulier.",
    'Surchauffe roulement pompe': "🛠 Vérifiez les roulements de la pompe et la lubrification.",
    'Surcharge mécanique': "⚙️ Vérifiez les charges mécaniques et l'alignement.",
    'Débit insuffisant': "💧 Vérifiez les filtres, les vannes et le débit de la pompe.",
    'Anomalie électrique - tension': "🔌 Vérifiez la tension d'alimentation et les connexions électriques.",
    'Vibrations excessives': "🔧 Inspectez les supports, les roulements et l'alignement de l'arbre.",
    'Surchauffe bobine': "🌡️ Vérifiez le refroidissement et la température des bobines.",
    'Surchauffe roulement moteur': "🧰 Vérifiez la lubrification et l'état des roulements moteur.",
    'Cavitation possible': "⚠️ Vérifiez la pression d'aspiration et l'alimentation en liquide."
}

niveau_risque = {
    'Normal': "🟢 Faible",
    'Surchauffe roulement pompe': "🔴 Élevé",
    'Surcharge mécanique': "🔴 Élevé",
    'Débit insuffisant': "🟠 Moyen",
    'Anomalie électrique - tension': "🔴 Élevé",
    'Vibrations excessives': "🟠 Moyen",
    'Surchauffe bobine': "🔴 Élevé",
    'Surchauffe roulement moteur': "🔴 Élevé",
    'Cavitation possible': "🔴 Élevé"
}

entrees = {}

def remplir_aleatoirement():
    for champ, entree in entrees.items():
        if champ in plages_valeurs:
            min_val, max_val = plages_valeurs[champ]
            valeur = round(random.uniform(min_val, max_val), 2)
            entree.delete(0, 'end')
            entree.insert(0, str(valeur))

def predire_defaut():
    mesures = {}
    try:
        for champ, entree in entrees.items():
            valeur_str = entree.get().strip().replace(",", ".")
            if valeur_str == "":
                valeur_str = "0.0"
            valeur_str = ''.join(c for c in valeur_str if c.isdigit() or c == '.')
            mesures[champ] = float(valeur_str)

        df_mesures = pd.DataFrame([mesures])[model.feature_names_in_]

        prediction = model.predict(df_mesures)
        type_defaut = prediction[0]

        try:
            proba = model.predict_proba(df_mesures)
            indice = list(model.classes_).index(type_defaut)
            confiance = round(proba[0][indice]*100, 2)
            texte_confiance = f"{confiance}%"
        except:
            texte_confiance = "N/A"

        texte_resultat = (
            f"🔍 Type de défaut prédit :\n{type_defaut}\n\n"
            f"💡 Conseil :\n{conseils_defauts.get(type_defaut,'Aucun conseil')}\n\n"
            f"⚠️ Niveau de risque :\n{niveau_risque.get(type_defaut,'N/A')}\n\n"
            f"📊 Confiance du modèle :\n{texte_confiance}"
        )
        label_resultat.config(text=texte_resultat, foreground="#f39c12", justify='left', anchor='nw')

    except Exception as e:
        label_resultat.config(
            text=f"❌ Erreur : {str(e)}",
            foreground="red",
            justify='center',
            anchor='center'
        )

# Fenêtre principale avec thème solar
fenetre = ttk.Window(themename="solar", size=(1000, 750))
fenetre.title("🔧 Prédiction des Défauts - Machine Learning")

# Titre
titre = ttk.Label(
    fenetre,
    text="🛠 Interface de prédiction de défauts de pompe électrique",
    font=("Arial", 18, "bold"),
    bootstyle="warning"
)
titre.pack(pady=20)

# Cadre principal
cadre_principal = ttk.Frame(fenetre)
cadre_principal.pack(fill='both', expand=True, padx=15, pady=10)
cadre_principal.columnconfigure(0, weight=1)
cadre_principal.columnconfigure(1, weight=1)
cadre_principal.rowconfigure(0, weight=1)

# Partie gauche : saisie
cadre_haut = ttk.LabelFrame(cadre_principal, text="🔢 Données d'entrée", padding=15, bootstyle="info")
cadre_haut.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
cadre_haut.columnconfigure(1, weight=1)

for i, champ in enumerate(champs):
    label = ttk.Label(cadre_haut, text=champ + " :", font=("Arial", 10, "bold"))
    label.grid(row=i, column=0, sticky="w", pady=5, padx=5)
    entree = ttk.Entry(cadre_haut)
    entree.grid(row=i, column=1, sticky="ew", pady=5, padx=5)
    entrees[champ] = entree

# Ligne des boutons (orange à gauche)
ligne_boutons = ttk.Frame(cadre_haut)
ligne_boutons.grid(row=len(champs), column=0, columnspan=2, pady=10)

btn_random = ttk.Button(
    ligne_boutons, text="🎲 Remplir aléatoirement", command=remplir_aleatoirement,
    bootstyle="warning"
)
btn_random.pack(side="left", padx=10)

btn_predire = ttk.Button(
    ligne_boutons, text="🔍 Lancer la prédiction", command=predire_defaut,
    bootstyle="success"
)
btn_predire.pack(side="left", padx=10)

# Partie droite : résultat
cadre_bas = ttk.LabelFrame(cadre_principal, text="📋 Résultat de la prédiction", padding=15, bootstyle="secondary")
cadre_bas.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
cadre_bas.columnconfigure(0, weight=1)
cadre_bas.rowconfigure(0, weight=1)

label_resultat = ttk.Label(
    cadre_bas,
    text="🕒 En attente de prédiction...",
    font=("Arial", 12),
    wraplength=400,
    justify='left',
    anchor='nw'
)
label_resultat.grid(row=0, column=0, sticky="nsew", padx=10, pady=10) 
fenetre.mainloop()


# In[ ]:




