# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 15:19:52 2020

@author: Alexis
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

dataSNCF = pd.read_csv(".\\regularite-mensuelle-tgv-aqst.csv",sep=";")



def GraphBase():    
    
    ################################################ Retard moyen en minute
    
    sns.set_theme(style="darkgrid", palette="dark", context="paper", )
    fig = sns.lineplot(x="Année", y="Retard moyen de tous les trains à l'arrivée (min)", data=dataSNCF)
    fig.set_xticks(range(2015,2021)) #On règle l'axe des abscisses 
    plt.ylabel("Retard moyen de tous les trains (mn)")
    plt.title("Evolution du retard moyen en min (Période 2015-2020)")
    plt.show()
    
    
    ############################################### causes en % 2015-2020
    
    sns.set_theme(style="darkgrid", palette="dark")
    sns.lineplot(x="Année", y="% trains en retard pour causes externes (météo, obstacles, colis suspects, malveillance, mouvements sociaux, etc.)", data=dataSNCF)
    sns.lineplot(x="Année", y="% trains en retard à cause infrastructure ferroviaire (maintenance, travaux)", data=dataSNCF)
    sns.lineplot(x="Année", y="% trains en retard à cause gestion trafic (circulation sur ligne ferroviaire, interactions réseaux)", data=dataSNCF)
    sns.lineplot(x="Année", y="% trains en retard à cause matériel roulant", data=dataSNCF)
    sns.lineplot(x="Année", y="% trains en retard à cause gestion en gare et réutilisation de matériel",data=dataSNCF)
    sns.lineplot(x="Année", y="% trains en retard à cause prise en compte voyageurs (affluence, gestions PSH, correspondances)",data=dataSNCF)
    plt.ylabel("% Retards")
    plt.ylim(0, 0.4) #On limite l'axe des y à des valeurs qui nous intéressent
    plt.locator_params(integer=True) 
    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown'] #On règle les couleurs du graphique puis la légende
    lines = [Line2D([0], [0], color=c, linewidth=1.5, linestyle='--') for c in colors]
    labels = ["Externe*", "Infrastructure**", "Trafic", "Matériel roulant***", "Gestion de gare", "Affluence"]
    plt.legend(lines, labels, bbox_to_anchor=(1.01, 1),borderaxespad=0)
    plt.title("Evolution de la part des différentes causes dans le retard des trains") # Et enfin le titre
    
    plt.show()
    
    
     ############################################### Taux retard
    
    dataSNCF["NbTrainT"] = dataSNCF["Nombre de circulations prévues"] # Nombre de trains théorique
    dataSNCF["NbTrainE"] = dataSNCF["NbTrainT"] - dataSNCF["Nombre de trains annulés"] # Nombre effectif
    dataSNCF["NbEnRetard"] = dataSNCF["Nombre de trains en retard à l'arrivée"] # Nombre de retards
    dataSNCF["TauxRetard"] = dataSNCF["NbEnRetard"]/dataSNCF["NbTrainE"] #Taux de retard
    
    
    sns.set_theme(style="darkgrid")
    fig = sns.lineplot(x="Année", y="TauxRetard", data=dataSNCF)
    fig.set_xticks(range(2015,2021))# On règle l'axe des abscisses
    plt.title("Taux de retards (Période 2015-2020)") #Et le titre
    plt.title("Evolution du taux de retard effectif")
    plt.show()
    
    print("Légendes approfondie des causes :", "\n", "* Présence d'obstacles, intempéries etc.", "\n",
         "** Maintenances", "\n", "*** Véhicules ferroviaires obstruant les voies", "\n" )
          

def BarPiePlot():
    ##Pour faire le barplot, on a besoin de prendre la moyenne de chaque cause de retard
     
    sns.set_theme(style="white", context="notebook", palette="dark")

    Ext=dataSNCF["% trains en retard pour causes externes (météo, obstacles, colis suspects, malveillance, mouvements sociaux, etc.)"].describe()["mean"]
    print("Part des retards pour causes externes :",Ext,)
    Infra=dataSNCF["% trains en retard à cause infrastructure ferroviaire (maintenance, travaux)"].describe()["mean"]
    print("Part des retards pour cause d'infrastructure :", Infra)
    Traf=dataSNCF["% trains en retard à cause gestion trafic (circulation sur ligne ferroviaire, interactions réseaux)"].describe()["mean"]
    print("Part des retards pour cause de gestion du trafic :", Traf)
    Mat=dataSNCF["% trains en retard à cause matériel roulant"].describe()["mean"]
    print("Part des retards pour cause de matériel :", Mat)
    Gest=dataSNCF["% trains en retard à cause gestion en gare et réutilisation de matériel"].describe()["mean"]
    print("Part des retards pour cause de gestion en gare :", Gest)
    Afflu=dataSNCF["% trains en retard à cause prise en compte voyageurs (affluence, gestions PSH, correspondances)"].describe()["mean"]
    print("Part des retards pour cause d'affluence voyageurs :" , Afflu)

    Retard_par_cause=[Ext, Infra, Traf, Mat, Gest, Afflu] # On les compile dans une liste
    bars=['Ext', 'Infra', 'Traf', 'Mat', 'Gest', 'Afflu'] # On nomme nos barres
    
    sns.barplot(x=bars, y=Retard_par_cause)
    plt.ylabel("Part du retard expliqué")
    plt.title("Part moyenne du retard des trains expliquée par une des causes (Période 2015-2020)")

    somme = Ext + Infra + Traf + Mat + Gest + Afflu #On vérifie que la somme fait bien 1
    print("Somme :",somme, "\n") 
    plt.show()
    
    if somme >= 0.95: #Boucle qui n'affiche le diagramme que si la somme est supérieure à un seuil résiduel que l'on a arbitrairement situé à 0.05.
        print("La somme des parts fait effectivement ~ 1, le résidu est négligeable et il est pertinent de faire un diagramme circulaire.", "\n")
        plt.figure()
        plt.pie(x=Retard_par_cause, labels=bars, autopct='%.2f')
        plt.title("Répartition moyenne des causes de retard (en %)")
        plt.show()
    else:
        None

def Graphiques():
    GraphBase()
    BarPiePlot()
    