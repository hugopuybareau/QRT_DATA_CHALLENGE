### Import packages ###

import numpy as np
import pandas as pd
from typing import List, Literal

### Machine Learning ###

# Pre-processing
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Modelisation
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    KFold,
    GridSearchCV,
)

from scipy import stats
from scipy.stats import pearsonr, spearmanr

# Metrics
from sklearn.metrics import mean_squared_error, confusion_matrix
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    auc,
    roc_curve,
)
import shap

# Graphics
import matplotlib.pyplot as plt
import seaborn as sns

### Pre-processing functions ###


def replacing(df : pd.DataFrame):  # Function that will group 'Null's and 'NaN's as 'NaN's

    df.replace("None", np.nan, inplace=True)
    df.replace("Null", np.nan, inplace=True)

    return df


def nan_colormap(df : pd.DataFrame, title : str):  # Function using seaborn's colormap to locate NaNs, use after 'replacing'

    plt.figure(figsize=(16,9))
    sns.heatmap(df.isna(), cbar=False)
    plt.title(title, fontdict={"fontsize": 18}, pad=12)


def spot_columns(df: pd.DataFrame, percent: float) -> None:

    # Calcul du pourcentage de NaNs pour chaque colonne
    nan_by_columns = df.isna().mean() * 100
    
    # Filtrer les colonnes qui ont plus de 'percent' % de NaNs
    columns_with_nans = nan_by_columns[nan_by_columns > percent]
    
    # Afficher les colonnes correspondantes
    for column, nan_percent in columns_with_nans.items():
        print(f"Column '{column}' has {nan_percent:.2f}% NaN values")
        

def replace_outliers_z1(df: pd.DataFrame, range : int):

    df_numeric = df.apply(pd.to_numeric, errors='coerce')
    
    # Calculer les Z-scores pour chaque valeur du DataFrame
    z_scores = np.abs(stats.zscore(df_numeric))
    outliers = np.where(z_scores > range) # Indices des valeurs abérrantes
    
    df_cleaned = df.copy()
    
    # Remplacer les valeurs aberrantes par la moyenne de la colonne correspondante
    for row, col in zip(outliers[0], outliers[1]):
        mean_value = df_numeric.iloc[:, col].median()
        df_cleaned.iat[row, col] = mean_value
    
    return df_cleaned

def replace_outliers_iqr(df: pd.DataFrame, column: str) -> pd.DataFrame:
    # Calculer le premier quartile (Q1) et le troisième quartile (Q3)
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    # Définir les bornes pour les valeurs aberrantes
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Calculer la moyenne de la colonne
    mean_value = df[column].mean()

    # Remplacer les valeurs aberrantes par la moyenne
    df[column] = np.where((df[column] < lower_bound) | (df[column] > upper_bound), mean_value, df[column])
    
    return df


def nan_imputer(df: pd.DataFrame, cols: List[str], method: Literal['mean', 'no_data', 'ffill', 'bfill', 'zeros']) -> pd.DataFrame:
    # Travailler sur une copie du DataFrame
    df_copy = df.copy()
    
    for col in cols:
        if method == 'mean':
            if df_copy[col].dtype in ["float64", "int64"]:  # Gérer les colonnes numériques
                df_copy[col] = df_copy[col].fillna(df_copy[col].mean())
            else:
                df_copy[col] = df_copy[col].fillna("no_data")  # Pour les colonnes non numériques
        elif method == 'ffill':
            df_copy[col] = df_copy[col].fillna(method='ffill')
        elif method == 'bfill':
            df_copy[col] = df_copy[col].fillna(method='bfill')
        elif method == 'zeros':
            df_copy[col] = df_copy[col].fillna(0)
        elif method == 'no_data':  # Nouveau bloc pour la méthode no_data
            df_copy[col] = df_copy[col].fillna("no_data")
        else:
            raise ValueError(f"Method '{method}' is not recognized. Use 'mean', 'no_data', 'ffill', 'bfill', or 'zeros'.")
    
    return df_copy


def frequencie_pie_chart(df : pd.DataFrame, col : str, title : str):  # Frequencie pie chart on a specific column
    freq = pd.crosstab(df[col], columns="count")
    freq = freq * 100 / freq.sum()

    color_map = sns.color_palette("BrBG")

    plt.figure(figsize=(10, 6))
    plt.pie(
        freq["count"],
        labels=freq.index,
        autopct="%1.1f%%",
        startangle=90,
        colors=color_map,
        wedgeprops={"linewidth": 2, "edgecolor": "black"},
    )
    plt.title(title, fontdict={"fontsize": 18}, pad=12)
    plt.show()


def correlation_matrix(df: pd.DataFrame, method: Literal['pearson', 'kendall', 'spearman'], title: str):  # Returns the correlation matrix colormap of a DataFrame

    # Calculer la matrice de corrélation
    tab = df.corr(method=method)

    # Création de la figure et de l'axe
    plt.figure(figsize=(10, 8))
    sns.heatmap(tab, annot=True, cmap="coolwarm", fmt=".2f", linewidths=.5, cbar_kws={"shrink": .8})

    # Rotation des étiquettes des axes x pour une meilleure lisibilité
    plt.xticks(rotation=90)

    # Ajouter un titre
    plt.title(title, fontdict={"fontsize": 18}, pad=12)

    # Afficher la figure
    plt.show()

def univariate_analysis(df : pd.DataFrame, num_cols : list, cat_cols : list):
    
    # Visualisation des distributions des variables numériques sans palette
    for col in num_cols:
        plt.figure(figsize=(12, 6))
        
        # Histogramme sans palette
        plt.subplot(1, 2, 1)
        sns.histplot(df[col], kde=True)
        plt.title(f'Histogramme de {col}')
        
        # Boxplot
        plt.subplot(1, 2, 2)
        sns.boxplot(x=df[col])
        plt.title(f'Boxplot de {col}')
        
        plt.tight_layout()
        plt.show()
    
    # Visualisation des fréquences des variables catégorielles
    for col in cat_cols:
        plt.figure(figsize=(12, 6))
        sns.countplot(y=df[col], order=df[col].value_counts().index)
        plt.title(f'Diagramme en barres de {col}')
        plt.show()

def bivariate_analysis(df : pd.DataFrame):
    # Séparer les variables numériques et catégorielles
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=[object, 'category']).columns
    
    # Analyse des relations entre variables numériques
    for col1 in numeric_cols:
        for col2 in numeric_cols:
            if col1 != col2:
                plt.figure(figsize=(12, 6))
                
                # Nuage de points avec ligne de régression
                plt.subplot(1, 2, 1)
                sns.regplot(x=df[col1], y=df[col2], scatter_kws={'s':10})
                plt.title(f'{col1} vs {col2}')
                
                # Afficher le coefficient de corrélation
                corr_pearson, _ = pearsonr(df[col1], df[col2])
                corr_spearman, _ = spearmanr(df[col1], df[col2])
                plt.subplot(1, 2, 2)
                sns.heatmap([[corr_pearson, corr_spearman]], annot=True, cmap='coolwarm', cbar=False,
                            xticklabels=['Pearson', 'Spearman'], yticklabels=['Correlation'])
                plt.title(f'Corrélation entre {col1} et {col2}')
                
                plt.tight_layout()
                plt.show()
    
    # Analyse des relations entre variables catégorielles
    for col1 in categorical_cols:
        for col2 in categorical_cols:
            if col1 != col2:
                plt.figure(figsize=(12, 6))
                sns.countplot(x=col1, hue=col2, data=df)
                plt.title(f'{col1} vs {col2}')
                plt.show()
    
    # Analyse des relations entre variables numériques et catégorielles
    for col1 in categorical_cols:
        for col2 in numeric_cols:
            plt.figure(figsize=(12, 6))
            sns.boxplot(x=col1, y=col2, data=df)
            plt.title(f'{col1} vs {col2}')
            plt.show()

def plot_time_series(df : pd.DataFrame, time_column : str, columns_to_plot : list) :
    
    # Définir la taille de la figure
    plt.figure(figsize=(14, 8))
    
    # Tracer chaque colonne
    for column in columns_to_plot:
        sns.lineplot(x=df[time_column], y=df[column], label=column)
    
    # Définir les labels et le titre
    plt.xlabel('Temps')
    plt.ylabel('Valeurs')
    plt.title('Évalutation temporelle des données')
    plt.legend()
    plt.grid(True)
    
    # Afficher le graphique
    plt.show()

def plot_time_series_with_nans(df : pd.DataFrame, time_column : str, columns_to_plot : list, meth=Literal['mean', 'ffill', 'bfill']):
    
    plt.figure(figsize=(14, 8))
    
    for column in columns_to_plot:

        original_series = df[column].copy()
        # Imputation des NaNs selon la méthode choisie
        if meth == 'mean':
            if df[column].dtype in ["float64", "int64"]:
                df[column] = df[column].fillna(df[column].mean())
            else:
                df[column] = df[column].fillna("no_data")
        else :
            df[column] = df[column].fillna(method=meth)
        
        # Tracer les données imputées
        sns.lineplot(x=df[time_column], y=df[column], label=column)
        
        # Mettre en évidence les NaNs dans les données originales
        nan_indices = original_series[original_series.isna()].index
        if not nan_indices.empty:
            plt.scatter(df.loc[nan_indices, time_column], [df[column].mean()] * len(nan_indices), color='red')
    
    # Définir les labels et le titre
    plt.xlabel('Temps')
    plt.ylabel('Valeurs')
    plt.title('Évolution temporelle des données')
    plt.legend()
    plt.grid(True)
    
    # Afficher le graphique
    plt.show()


### Modélisation ###


def cross_validation(model, x_train: pd.DataFrame, y_train: pd.DataFrame, n_fold: int, scoring_method: str):

    kf = KFold(n_splits=n_fold, shuffle=True, random_state=42)
    score = cross_val_score(model, x_train, y_train, cv=kf, scoring=scoring_method)

    print(
        "##### CROSS-VALIDATION SCORES #####",
        "\n\n",
        scoring_method,
        " scores from cross-validation ",
        score,
        "\n",
        scoring_method,
        " mean from cross-validation ",
        score.mean(),
        "\n",
        "Standard deviation ",
        score.std(),
        "\n\n",
        "###################################",
    )


def metrics(model, x_test : pd.DataFrame, y_test : pd.DataFrame, title : str, type : Literal['classifier', 'regressor']):

    y_pred = model.predict(x_test)  
    auc = 0
    gini = 0

    if type == "classifier":
        y_pred_proba = model.predict_proba(x_test)[:, 1]
        cm = confusion_matrix(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba, multi_class="ovr")
        gini = 2 * auc - 1
        plt.figure(figsize=(8, 6))
        sns.set(font_scale=1.2)
        sns.heatmap(cm, annot=True, cmap="coolwarm", cbar=True)
        plt.xlabel("Predicted")
        plt.ylabel("Originals")
        plt.title(title, fontdict={"fontsize": 18}, pad=12)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")

    print(
        "##### SCORES #####",
        "\n\n",
        "Accuracy = ",
        accuracy,
        "\n",
        "Precision = ",
        precision,
        "\n",
        "Recall = ",
        recall,
        "\n",
        "F1 - Score = ",
        f1,
        "\n",
        "GINI = ",
        gini,
        "\n\n",
        "##################",
    )
