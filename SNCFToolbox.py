# -*- coding: utf-8 -*-
"""
Module
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Fonctions

def CreateIndicator(df,Col,rem=0):
    """
    La fonction CreateIndicator permet de transformer une variable texte en une variable numérique de type integer.
    Ex.: ["Pierre","Paul","Jacques","Pierre"] devient [1,2,3,1]. S'applique sur un dataframe.
    
    INPUT : un dataframe, la colonne d'intérêt, le souhait de retirer l'ancienne colonne ou non.
    OUTPUT : un nouveau dataframe.
    """
    DFList = df[Col].to_list() # Nous transformons la colonne en liste.
    Modalities = [] # Nous allons regrouper toutes les modalités de la variable.
    for element in DFList:
        if element not in Modalities:
            Modalities.append(element)
    Modalities.sort() # Pour nous assurer que la gare de départ et d'arrivée gardent le même indicatif (à priori)
    print(Modalities)
    NewCol = [] # Cette liste va devenir notre nouvelle colonne.
    for element in DFList: # La modalité est l'index dans la liste précédemment créée
        NewCol.append(int(Modalities.index(element))+1) # (+1 pour éviter d'avoir une modalité à 0)
    df[Col+" mod"] = NewCol
    
    if rem == 1:
        df = df.drop(Col,axis=1)
        print("Colonnes supprimées.\n")
    print("\nLa fonction CreateIndicator a été exécutée jusqu'au bout.")
    return df

            
    
def CreateDummy(df,VarName,NewName,OrigValues,NewValues=[0,1]):
    """
    La fonction CreateDummy crée une dummy en supprimant la colonne d'origine.
    
    INPUT : Le dataframe, le nom de la colonne à traiter, son nouveau nom, ses modalités (liste) et les valeurs à prendre 
    (liste, par défaut 0 ou 1)
    OUTPUT : Le dataframe retraité
    """
    # Nous procédons à un contrôle de cohérence : Les modalités présentes sont-elles celles renseignées ?
    Liste = df[VarName].to_list()
    Modalities = []
    for element in Liste:
        if element not in Modalities:
            Modalities.append(element)
    InterVar = OrigValues.sort() # Cette variable contient les valeurs triées, nous ne modifions pas la liste renseignée par l'user.
    Modalities = Modalities.sort()
    if Modalities != InterVar:
        Alert = input("Attention, les modalités initiales sont mal renseignées et il risque d'y avoir une erreur. Souhaitez-vous poursuivre ? (O/N)\n")
        if Alert not in "Oo":
            print("\nLa fonction CreateDummy s'est arrêtée, aucun changement n'a été appliqué.\n")
            return # Fin du bloc de contrôle.
    NewList = [] # Contient les valeurs de la dummy.
    for element in Liste: # Pour chaque élément de la colonne du df initial
        index = int(OrigValues.index(element)-1) # Nous prenons l'index de l'élément dans la liste renseignée
        NewList.append(NewValues[index]) # Et nous intégrons la nouvelle valeur d'index similaire.
    df = df.drop([VarName],axis=1) # Nous supprimons l'ancienne colonne
    df[NewName] = NewList # Et nous intégrons la nouvelle
    print("La fonction CreateDummy a été exécutée avec succès.")
    return df

def ToLog(df,VarName,rem=0):
    """
    La fonction ToLog va prendre une ou plusieurs colonnes d'un dataframe et appliquer une transformation
    logarithmique.
    
    INPUT : Le dataframe, les noms des colonnes (sous forme de liste)
    OUTPUT : Le dataframe retraité
    """
    for col in VarName: # Pour chaque colonne indiquée
        data = df[col].to_list() # Nous récupérons les valeurs
        Liste = []
        for obs in data: # Pour chaque observation de la liste
            if obs != 0: # Si obs = 0, nous laisserons à 0
                obs = np.log(obs)
            Liste.append(obs)
        df["ln( "+str(col)+" )"] = Liste # Nous intégrons la nouvelle colonne dans le df
        if rem == 1:
            df.drop(col,axis=1) # Si demandé, nous supprimons la colonne pré-existante.
    return df

def ExportDB(df):
    """
    La procédure ExportDB envoie le dataframe sélectionné dans le dossier "Exports".
    
    INPUT : un dataframe
    """
    try:
        os.mkdir(".//Exports")
    except:
        pass
    df.to_csv(path_or_buf=".//Exports//ExportedCSV.csv")
    
    
    
    