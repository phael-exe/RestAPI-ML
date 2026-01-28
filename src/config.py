from pydantic_settings import BaseSettings
import os

class Settings(BaseSettings):
    APP_NAME: str = "RestAPI-ML"
    VERSION: str = "1.0.0"
    DEBUG: bool = False
    MODEL_PATH: str = os.path.join("saved_models", "iris_model.joblib")
    
    class Config:
        env_file = ".env"

settings = Settings()
