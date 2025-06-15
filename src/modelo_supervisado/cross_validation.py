import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# Cargar datos procesados
df = pd.read_csv('data/processed/accidents_processed.csv')
X = df.drop('Accident Severity Encoded', axis=1)
y = df['Accident Severity Encoded']

# Configurar modelo (mismos hiperparámetros que antes)
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    class_weight='balanced',
    random_state=42
)

# Validación cruzada con 100 splits
scores = cross_val_score(
    model, 
    X, 
    y, 
    cv=100,  # 100 particiones
    scoring='accuracy'
)

# Resultados
print(f"Mediana de accuracy en 100 splits: {np.median(scores):.2f}")
print(f"Rango de accuracy: [{np.min(scores):.2f}, {np.max(scores):.2f}]")