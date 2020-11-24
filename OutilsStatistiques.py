# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt
import datetime
import pandas as pd
import seaborn as sns

def HeatMap(df):
    """
    La procédure Heatmap crée... une heatmap. Elle est basée sur la matrice de corrélation.
    Seules les variables numériques doivent être prises en compte.
    INPUT : un dataframe
    RESULT : Dessin d'une heatmap
    """
    NewDF = pd.DataFrame()
    i=1 # L'ID de la colonne dans le nouveau dataframe
    ColList = [] # La liste des colonnes incluses dans le nouveau DF
    for column in df.columns:
        Col = df[column].to_list() # Nous allons analyser la colonne sous forme de liste
        Check = True # Cette variable prend la valeur True par défaut et sera Fausse si la colonne n'est pas du bon type.
        for element in Col:
            if (isinstance(element, int) or isinstance(element,float)) == False:
                Check = False
        if Check == True:
            NewDF[str(i)] = df[column]
            i += 1
            ColList.append(column)
    print("Colonnes conservées : \n",ColList)
    corr = NewDF.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    f, ax = plt.subplots(figsize=(11, 9))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    graph = sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

class LROLS():
    """
    La class 'Linear Regression - Ordinary Least Squares' (ou LROLS) nous permet de calculer les 
    éléments propres à cette technique d'estimation. Il convient toutefois d'utiliser sklearn ou 
    statmodels pour un résultat optimal.
    """
    
    # Attributs
    def __init__(self,X,Y,Label="OLS"):
        self.X = X
        self.Y = Y # Nous voulons récupérer deux dataframes X et Y
        self.Label = Label
    
    # Méthodes
    
    def VarGraph(self):
        """
        La procédure VarGraph va dessiner l'histogramme et le graphique d'une variable dont le nom
        est indiqué dans la variable var. L'objectif est de constater visuellement la constance des 
        deux premiers moments et la potentielle présence d'outliers.
        """
        fig = plt.plot(self.Y)
        plt.title("Variable dépendante")
        plt.show()
        i=0
        for col in self.X.columns:
            fig = plt.plot(self.X[col])
            plt.title(self.X.columns[i])
            i+=1
            plt.show()
    
    def CorrGraph(self):
        """
        La procédure CorrGraph va dessiner des scatter plots pour déctecter une tendance linéaire.
        """
        y = self.Y.to_list()
        Nex = int(len(self.X.columns)) # Nombre de variables explicatives
        if Nex%2 == 0: # Nous allons mettre plusieurs graphiques en même temps. Nous cherchons la
            nrows = int(Nex/2) # configuration optimale pour le plcement des scatter plots.
            ncols = int(2)
            fig,axes = plt.subplots(nrows,ncols)
            fig.tight_layout() # Sert à espacer les graphiques.
            ind=0
            for row in range(0,nrows,1): # Pour chaque ligne ...
                for column in range(0,ncols,1): # ... pour chaque colonne ...
                    axes[row,column].scatter(self.X[self.X.columns[ind]],y,s=0.05) # ... on crée un scatter.
                    axes[row,column].set_title(str(self.X.columns[ind]))
                    ind +=1
            plt.show()
        elif Nex == 2 or Nex == 3:
            fig,axes = plt.subplots(Nex,1)
            fig.tight_layout() # Sert à espacer les graphiques.
            for i in range(0,Nex):
                axes[i].scatter(self.X[self.X.columns[i]],y,s=0.05)
                axes[i].set_title(str(self.X.columns[i]))
            plt.show()
        elif Nex%3 == 0:
            ncols = int(3)
            nrows = int(Nex/3)
            fig,axes = plt.subplots(nrows,ncols)
            fig.tight_layout() # Sert à espacer les graphiques.
            ind=0
            for row in range(0,nrows): # Pour chaque ligne ...
                for column in range(0,ncols): # ... pour chaque colonne ...
                    axes[row,column].scatter(self.X[self.X.columns[ind]],y,s=0.05) # ... on crée un scatter.
                    axes[row,column].set_title(str(self.X.columns[ind]))
                    ind +=1
            plt.show() # On montre le premier graphique, le subplot avec les n-1 variables
        else:
            ncols = int(2)
            nrows = int(Nex/2-0.5) # On édite les n-1 premiers graphiques
            fig,axes = plt.subplots(nrows,ncols)
            fig.tight_layout() # Sert à espacer les graphiques.
            ind=0
            for row in range(0,nrows): # Pour chaque ligne ...
                for column in range(0,ncols): # ... pour chaque colonne ...
                    axes[row,column].scatter(self.X[self.X.columns[ind]],y,s=0.05) # ... on crée un scatter.
                    axes[row,column].set_title(str(self.X.columns[ind]))
                    ind +=1
            plt.show() # On montre le premier graphique, le subplot avec les n-1 variables
            fig = plt.scatter(self.X[-1],y,s=0.05)
            fig.set_title(str(self.X.columns[-1]))    
        
    def GetCoeffs(self):
        """
        La fonction GetCoeffs nous donne les coefficients calculés par les moindre carrés ordinaires.
        """
        # La première étape est de transformer le dataframe en une matrice gérée par Numpy.
        y = np.transpose(self.Y.to_numpy())
        nlines = self.X.shape[0] # Nombre de lignes du DF
        cteline = np.ones((nlines,1)) # Un vecteur de 1 qui materialisera la constante
        semX = self.X.to_numpy()
        X = np.concatenate((cteline,semX),axis=1)
        Inter1 = np.linalg.inv(np.dot(np.transpose(X),X))
        Inter2 = np.dot(np.transpose(X),y)
        Beta = np.dot(Inter1,Inter2)# Beta = (X'X)^{-1}X'Y
        return Beta.tolist()
    
    def GetResiduals(self):
        """
        La fonction GetResiduals renvoie les rédidus.
        """
        Beta = self.GetCoeffs()
        nlines = self.X.shape[0] # Nombre de lignes du DF
        cteline = np.ones((nlines,1)) # Un vecteur de 1 qui materialisera la constante
        semX = self.X.to_numpy()
        X = np.concatenate((cteline,semX),axis=1)
        Yest = [] # Cette liste contiendra les prédictions pour y
        for row in X: # Pour chaque observation
            Yest.append(np.dot(X,Beta)) # On ajoute XB la prédiction
        y = np.transpose(self.Y.to_numpy())
        res = y - Yest
        res = res.tolist()[0] # On met un index car ça nous renvoie une liste contenant une autre liste.
        return res
        
        
    def GetSD(self):
        """
        La fonction GetSD nous renvoie les écarts-types estimés des coefficients.
        """
        nlines = self.X.shape[0] # Nombre de lignes du DF
        cteline = np.ones((nlines,1)) # Un vecteur de 1 qui materialisera la constante
        semX = self.X.to_numpy()
        X = np.concatenate((cteline,semX),axis=1)
        Res = self.GetResiduals() # Nous allons calculer SSR
        SSR = 0
        for element in Res:
            SSR += element**2
        VarEs = SSR/(X.shape[0]-len(self.GetCoeffs())) # Estimation de la variance des aléas
        Mat = np.linalg.inv(np.dot(np.transpose(X),X)) # Mat est la matrice (X'X)^{-1}
        VCov = np.dot(VarEs,Mat) # Estimation de la matrice de variance-covariance des coefficients.
        Liste = []
        for el in range(0,len(self.GetCoeffs())): # Nous nous attendons à avoir une liste de même longueur 
            sd = np.sqrt(VCov[el,el]) # que celle des coefficients.
            Liste.append(sd)
        return Liste
        
        
    def GetTStats(self):
        """
        La fonction GetTStats nous renvoie les t-statistiques des coefficients.
        """
        Coeffs = self.GetCoeffs()
        Sd = self.GetSD()
        Liste = []
        for el in range(0,len(Sd)):
            T = Coeffs[el]/Sd[el]
            Liste.append(T)
        return Liste
    
    def PlotResiduals(self):
        """
        La procédure PlotResiduals() trace le graphique des résidus.
        """
        Res = self.GetResiduals()
        fig = plt.plot(Res)
        plt.title("Graphique des résidus, modèle "+str(self.Label))
        plt.show()
        
    def GetR2(self):
        """
        La fonction GetR2 nous renvoie le coefficient de détermination de la régression et le coefficient
        de détermination ajusté.
        """
        ymean = self.Y.describe()["mean"] # moyenne de la variable dépendante
        ylist = self.Y.to_list()
        SCT = 0
        for i in range(0,len(self.Y)): # Nous réalisons la somme des carrés totaux
            SCT += (ylist[i] - ymean)**2
        res = self.GetResiduals()
        SCR = 0
        for i in range(0,len(res)): # Et la somme du carré des résidus
            SCR += res[i]**2
        R2 = 1-SCR/SCT
        N = self.X.shape[0] # Nombre d'observations
        K = self.X.shape[1] + 1 # Nombre de régresseurs, p+1
        R2adj = 1-(SCR/(N-K))/(SCT/(N-1))
        print(R2adj)
        return (R2,R2adj)
    
    def WhiteS(self):
        """
        La fonction WhiteReduced permet de faire un test de White simplifié. Le carré des résidus du
        modèle est expliqué par la prédiction et son carré, les coefficients étant estimés par les MCO.
        La statistique de test NR^{2} suit asymptoptiquement un khi-2 à 2 degrés de liberté.
        
        Le test de White permet d'indiquer si les résidus sont dans une configuration homoscédastique ou
        hétéroscédastique. Dans le second cas, les MCO ne sont pas optimaux et le modèle précedemment
        construit est invalidé.
        """
        nlines = self.X.shape[0] # Nombre de lignes du DF
        cteline = np.ones((nlines,1)) # Un vecteur de 1 qui materialisera la constante
        semX = self.X.to_numpy()
        X = np.concatenate((cteline,semX),axis=1)
        Beta = self.GetCoeffs()
        Liste=[]
        for i in range(0,len(Beta)):
            Liste.append([Beta[i]])
        Beta = np.array(Liste) # Beta est une liste à l'origine, nous le transformons en vecteur
        Ypred = np.dot(X,Beta) # Nous avons les prédictions pour y
        Ypred2 = []
        for i in range(0,len(Ypred)): # Et nous créons leurs carrés
            Ypred2.append(Ypred[i]**2)
        Ypred2 = np.array(Ypred2) # Ypred2 devient un vecteur vertical
        one = np.ones((Ypred.shape[0],1),dtype=int)
        X2 = np.concatenate((one,Ypred),axis=1)
        X2 = np.concatenate((X2,Ypred2),axis=1) # Nous créons notre nouvelle matrice X
        res = self.GetResiduals() # Nous avons les résidus ...
        res2 = [] #... que nous élevons au carré.
        for i in range(0,len(res)):
            el = res[i]**2
            res2.append([el]) # el est dans une liste pour que numpy comprenne qu'il faut créer un vecteur
        res2 = np.array(res2) # vertical
        Beta2 = np.dot(np.linalg.inv(np.dot(np.transpose(X2),X2)),np.dot(np.transpose(X2),
                        res2)) # Estimation MCO des beta de la régression auxiliaire
        SCT = 0
        mean = np.mean(res2) # Nous calculons SCT
        for i in range(0,len(res2)):
            SCT += (res2[i]-mean)**2
        SCT = float(SCT[0]) # SCT a été stocké dans une liste, nous le retirons.
        res2pred = np.dot(X2,Beta2) # Les prédictions du modèle auxiliaire.
        Sresiduals = ((res2 - res2pred)**2).tolist()  # Contient les résidus de la régression auxiliaires.
        SCR = 0 # Calculons SCR
        for i in range(0,len(Sresiduals)):
            SCR += Sresiduals[i][0] # Chaque obs. étant elle-même dans une sous-liste.
        R2 = 1 - (SCR/SCT) # Nous avons désormais de quoi créer la stat de White.
        StatCal = X2.shape[0]*R2
        CriticalValue = 5.99 # Observée à 5% et 2 ddl
        if StatCal >= CriticalValue:
            print("La statistique calculée "+str(round(StatCal,3))+" est supérieure à la valeur critique",
                  str(CriticalValue)+" observée au seuil de 5% et à deux degrés de liberté. Au",
                  "risque de première espèce de 5%, il y a hétéroscédasticité.", "\n")
            return True # Il y a hétéroscédasticité.
        elif StatCal < CriticalValue:
            print("La statistique calculée "+str(round(StatCal,3))+" est inférieure à la valeur critique",
                  str(CriticalValue)+" observée au seuil de 5% et à deux degrés de liberté. Au",
                  "risque de première espèce de 5%, il y a homoscédasticité.", "\n")
            return False # Il n'y en a pas
        else:
            print("Une erreur imprévue est survenue.", "\n")
            return 
        
    def DurbinWatson(self):
        """
        La méthode DurbinWatson réalise le test de Durbin Watson pour vérifier si les résidus de la
        régression suivent un processus AR(1). Attention, ce test présente quelques défauts et doit
        être complété avec plusieurs autres. Les valeurs des statistiques sont pour 3 explicatives.
        """
        Residuals = self.GetResiduals() # On récupère les résidus
        SSR = 0
        for i in range(0,len(Residuals)): # Calcul de SSR
            SSR += Residuals[i]**2
        SSD = 0
        for i in range(1,len(Residuals)): # Calcul du numérateur
            SSD += (Residuals[i] - Residuals[i-1])**2
        DW = SSD/SSR
        dL = 1.73
        dU = 1.80 # Stats de la table de Durbin-Watson
        if DW >= 0 and DW < dL:
            print("DW = "+str(round(DW,3))+", il y a une autocorrélation positive.", "\n")
            return 1
        elif (DW >= dL and DW < dU) or (DW > (4-dU) and DW <= (4-dL)):
            print("DW = "+str(round(DW,3))+", il y a un doute.", "\n")
            return 2
        elif DW >= dU and DW <= (4-dU):
            print("DW = "+str(round(DW,3))+", il n'y a pas d'autocorrélation.", "\n")
            return 3
        elif DW > (4-dL) and DW < 4:
            print("DW = "+str(round(DW,3))+", il y a une autocorrélation positive.", "\n")
            return 4
        else:
            print("DW = "+str(round(DW,3))+", une erreur inattendue est survenue.", "\n")
            return
            
    def ExportResults(self,filename="RésultatsOLS"):
        """
        La méthode ExportResults exporte les résultats de la régression dans un fichier texte.
        Plusieurs régressions peuvent être exportées.
        """
        try:
            os.mkdir(".//Exports")
        except:
            pass
        coeff = self.GetCoeffs()
        tstats = self.GetTStats()
        colList = self.X.columns.to_list()
        with open(".//Exports//OLS.txt","a+") as dataFile:
            dataFile.write("\n\n=============================================================\n\n")
            dataFile.write(" Rapport de la régression "+self.Label+"  \n")
            dataFile.write(" Date et heure : "+str(datetime.datetime.now())+"\n\n")
            dataFile.write("=============================================================\n\n")
            dataFile.write("Les variables explicatives sont : "+str(colList)+". \n\n")
            dataFile.write("Les coefficients MCO sont : "+str(coeff)+". \n\n")
            dataFile.write("Les t-stat estimés sont : "+str(tstats)+". \n\n")
            dataFile.write("Le R2 estimé est : "+str(round(self.GetR2()[0],3))+". \n\n")
            dataFile.write("Le R2 ajusté estimé est : "+str(round(self.GetR2()[1],3))+". \n\n")
            if self.WhiteS() == True:
                dataFile.write("Attention, de l'hétéroscédasticité a été détectée.\n")
            else:
                dataFile.write("Le test de White simplifié n'a pas décelé d'hétéroscédasticité.\n")
            if self.DurbinWatson() == 1:
                dataFile.write("Attention, les résidus seraient positivement autocorrélés.\n\n")
            elif self.DurbinWatson() == 2:
                dataFile.write("Attention, il se pourrait que les résidus soient autocorrélés.\n\n")
            elif self.DurbinWatson() == 3:
                dataFile.write("Le test de Durbin-Watson n'a détecté aucune autocorrélation des résidus.\n\n")
            elif self.DurbinWatson() == 4:
                dataFile.write("Attention, les résidus seraient négativement autocorrélés.\n\n")
            dataFile.write("--------------------------------------------------------------")
        print("Le fichier a été créé / mis à jour.")
            
            
            
        