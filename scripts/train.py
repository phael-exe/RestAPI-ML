import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import fetch_california_housing
import joblib
from pathlib import Path

print("Iniciando script de treinamento...")

housing = fetch_california_housing()
X = pd.DataFrame(housing.data, columns=housing.feature_names)
y = pd.Series(housing.target, name='MedHouseVal')

print("Dados carregados com sucesso")

features_to_use = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population']
X = X[features_to_use]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(n_estimators=100, random_state=42)
print("Treinando o modelo RandomForestRegressor...")
model.fit(X_train, y_train)
print("Treinamento do modelo concluído.")

# Caminho para salvar o modelo
model_dir = Path(__file__).resolve().parent.parent / 'saved_models'
model_dir.mkdir(exist_ok=True)
model_filename = model_dir / 'california_housing_model.joblib'

joblib.dump(model, model_filename)

print(f"Modelo salvo em {model_filename}. Script de treinamento finalizado.")