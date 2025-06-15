import pandas as pd
from sklearn.impute import SimpleImputer

def impute_numeric(df, columns, strategy='median'):
    """
    Imputa valores nulos en columnas numéricas usando la mediana o media.
    
    Args:
        df (pd.DataFrame): Dataset original.
        columns (list): Lista de columnas numéricas.
        strategy (str): 'median' o 'mean'.
    
    Returns:
        pd.DataFrame: Dataset con valores imputados.
    """
    imputer = SimpleImputer(strategy=strategy)
    df[columns] = imputer.fit_transform(df[columns])
    return df

def impute_categorical(df, columns, strategy='most_frequent'):
    """
    Imputa valores nulos en columnas categóricas usando la moda.
    
    Args:
        df (pd.DataFrame): Dataset original.
        columns (list): Lista de columnas categóricas.
        strategy (str): 'most_frequent'.
    
    Returns:
        pd.DataFrame: Dataset con valores imputados.
    """
    imputer = SimpleImputer(strategy=strategy)
    df[columns] = imputer.fit_transform(df[columns])
    return df