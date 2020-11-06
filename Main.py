# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 09:55:36 2020

@author: gaeta
"""

import pandas as pd
import SNCFToolbox as ST

dataSNCF = pd.read_csv(".\\regularite-mensuelle-tgv-aqst.csv",sep=";") # Importation

print(dataSNCF.columns) # Affichage des colonnes

dataSNCF = dataSNCF.drop(["Commentaire (facultatif) annulations","Commentaire (facultatif) retards au départ",
               "Commentaire (facultatif) retards à l'arrivée"],axis=1) # Retrait de colonnes jugées inutiles. Axis indique la colonne.

