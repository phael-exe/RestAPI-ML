import joblib
import pandas as pd
from typing import List
from src.config import settings
from src.utils.logger import setup_logger
from src.schemas.prediction_schema import PredictionInput, PredictionOutput
from src.utils.exceptions import ModelNotLoadedError

logger = setup_logger(__name__)

class PredictionService:
    def __init__(self):
        self.model = None
        self._load_model()
    
    def _load_model(self):
        try:
            logger.info(f"Loading model from {settings.MODEL_PATH}")
            self.model = joblib.load(settings.MODEL_PATH)
            logger.info("Model loaded successfully.")
        except FileNotFoundError:
            logger.error(f"Model file not found at {settings.MODEL_PATH}")
            self.model = None
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self.model = None

    def predict(self, input_data: PredictionInput) -> PredictionOutput:
        if not self.model:
            raise ModelNotLoadedError("Model is not loaded.")
        
        # Prepare data for prediction (scikit-learn expects 2D array)
        data = [[
            input_data.sepal_length,
            input_data.sepal_width,
            input_data.petal_length,
            input_data.petal_width
        ]]
        
        prediction = self.model.predict(data)[0]
        
        # Map prediction to class name (Iris dataset classes)
        target_names = ['setosa', 'versicolor', 'virginica']
        class_name = target_names[prediction] if 0 <= prediction < len(target_names) else "unknown"
        
        return PredictionOutput(class_id=int(prediction), class_name=class_name)

# Singleton instance
prediction_service = PredictionService()
