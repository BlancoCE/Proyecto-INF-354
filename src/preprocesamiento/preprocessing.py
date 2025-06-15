import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from imblearn.over_sampling import SMOTE

def preprocess_data(df):
    # 1. Conservar columnas originales para referencia
    original_columns = df.columns
    
    # 2. Imputación de nulos
    numeric_cols = ['Driver Age', 'Speed Limit (km/h)', 'Number of Vehicles Involved', 'Number of Casualties']
    numeric_cols = [col for col in numeric_cols if col in df.columns]  # Solo las que existan
    
    imputer_num = SimpleImputer(strategy='median')
    df[numeric_cols] = imputer_num.fit_transform(df[numeric_cols])
    
    # 3. Codificación categórica COMPLETA
    categorical_cols = ['Weather Conditions', 'Road Type', 'Driver Gender', 
                       'State Name', 'City Name', 'Day of Week', 'Time of Day',
                       'Vehicle Type Involved', 'Road Condition', 
                       'Lighting Conditions', 'Traffic Control Presence',
                       'Driver License Status', 'Alcohol Involvement']
    
    # Filtrar solo las columnas que existen
    categorical_cols = [col for col in categorical_cols if col in df.columns]
    
    # OneHotEncoder para TODAS las categóricas
    encoder_ohe = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
    encoded_data = encoder_ohe.fit_transform(df[categorical_cols])
    encoded_df = pd.DataFrame(
        encoded_data,
        columns=encoder_ohe.get_feature_names_out(categorical_cols)
    )
    
    # 4. Ordinal Encoding para severidad
    severity_order = [['Minor', 'Serious', 'Fatal']]
    encoder_ordinal = OrdinalEncoder(categories=severity_order)
    df['Accident Severity Encoded'] = encoder_ordinal.fit_transform(df[['Accident Severity']])
    
    # 5. Escalado de numéricas
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    
    # 6. Combinar todo (excluyendo columnas originales no necesarias)
    cols_to_drop = categorical_cols + ['Accident Severity', 'Accident Location Details']
    cols_to_drop = [col for col in cols_to_drop if col in df.columns]
    
    df_processed = pd.concat([
        df.drop(cols_to_drop, axis=1),
        encoded_df
    ], axis=1)
    
    # 7. Verificación FINAL de tipos
    non_numeric = df_processed.select_dtypes(exclude=['number']).columns
    if len(non_numeric) > 0:
        print("\n¡Atención! Columnas no numéricas remanentes:", non_numeric)
        df_processed = pd.get_dummies(df_processed, columns=non_numeric)
    
    # 8. Balanceo con SMOTE
    X = df_processed.drop('Accident Severity Encoded', axis=1)
    y = df_processed['Accident Severity Encoded']
    
    print("\nVerificación final:")
    print("Total columnas:", len(X.columns))
    print("Columnas no numéricas:", X.select_dtypes(exclude=['number']).columns.tolist())
    
    if y.nunique() > 1:
        smote = SMOTE(random_state=42)
        X_balanced, y_balanced = smote.fit_resample(X, y)
        df_processed = pd.concat([
            pd.DataFrame(X_balanced, columns=X.columns),
            pd.Series(y_balanced, name='Accident Severity Encoded')
        ], axis=1)
    
    return df_processed

if __name__ == "__main__":
    # Leer datos
    df = pd.read_csv('data/raw/accident_prediction_india.csv')
    
    # Procesar
    df_processed = preprocess_data(df)
    
    # Guardar
    df_processed.to_csv('data/processed/accidents_processed.csv', index=False)
    print("Preprocesamiento completado. Datos guardados en data/processed/accidents_processed.csv")