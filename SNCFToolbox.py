# -*- coding: utf-8 -*-
"""
Module
"""

# Import des modules/packages

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def CreateIndicator(InputStr):
    """
    La fonction CreateIndicator permet de transformer une variable texte en une série d'indicatrices.
    Ex.: ["Pierre","Paul","Jacques","Pierre"] devient [1,2,3,1].
    
    INPUT : un vecteur str.
    OUTPUT : un vecteur int.
    """
    
def HeatMap(DF):
    """
    La procédure HeatMap permet de créer une matrice de corrélation visuelle via le package seaborn.
    
    INPUT : un Dataframe.
    OUTPUT : Aucun, dessin d'une heatmap.
    """
    
def Numerical(Col):
    '''
    La fonction Numerical permet d'extraire une variable de type int qui est intégrée dans une variable str.
    ex.: "110 minutes" devient "110".
    
    INPUT : une colonne de Dataframe.
    OUTPUT : une nouvelle colonne de Sataframe qui n'écrase pas l'ancienne.
    '''
    
def NumPercent(Col):
    """
    La fonction NumPercent permet de transformer un string de type "99%" en float de type "0.99"
    
    INPUT : une colonne de Dataframe.
    OUTPUT : Une nouvelle colonne de Dataframe qui n'écrase pas l'ancienne.
    """
    

