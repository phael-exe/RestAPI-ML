import joblib
import numpy as np
from pathlib import Path

class ModelPredictor:
    def __init__(self, model_path):
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Modelo não encontrado em: {model_path}")
        
        self.model = joblib.load(model_path)
        self.model_path = model_path

    def predict(self, features):
        """
        Faz predição com as features fornecidas
        
        Args:
            features: array ou lista com as features
            
        Returns:
            numpy array com as predições
        """
        # Garante que features está no formato correto (2D array)
        features = np.array(features)
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        return self.model.predict(features)
        