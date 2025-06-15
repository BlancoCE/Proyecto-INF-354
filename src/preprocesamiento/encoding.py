import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

def onehot_encode(df, columns, drop_first=True):
    """
    Aplica One-Hot Encoding a columnas categóricas nominales.
    
    Args:
        df (pd.DataFrame): Dataset original.
        columns (list): Lista de columnas a codificar.
        drop_first (bool): Elimina la primera categoría para evitar multicolinealidad.
    
    Returns:
        pd.DataFrame: Dataset con columnas codificadas.
    """
    encoder = OneHotEncoder(drop='first' if drop_first else None, sparse=False)
    encoded_data = encoder.fit_transform(df[columns])
    encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(columns))
    return pd.concat([df.drop(columns, axis=1), encoded_df], axis=1)

def ordinal_encode(df, column, categories):
    """
    Aplica Ordinal Encoding a columnas categóricas ordinales.
    
    Args:
        df (pd.DataFrame): Dataset original.
        column (str): Columna a codificar.
        categories (list): Orden jerárquico (ej: ['leve', 'serio', 'mortal']).
    
    Returns:
        pd.DataFrame: Dataset con columna codificada.
    """
    encoder = OrdinalEncoder(categories=[categories])
    df[column] = encoder.fit_transform(df[[column]])
    return df