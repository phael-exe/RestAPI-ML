from fastapi import APIRouter, HTTPException, status
from src.schemas.prediction_schema import PredictionInput, PredictionOutput
from src.services.prediction_service import prediction_service
from src.utils.exceptions import ModelNotLoadedError

router = APIRouter()

@router.get("/health", tags=["Saúde"])
async def health_check():
    """
    Endpoint de verificação de saúde.
    """
    if not prediction_service.model:
        return {"status": "unhealthy", "model_loaded": False}
    return {"status": "healthy", "model_loaded": True}

@router.post("/predict", response_model=PredictionOutput, tags=["Predição"])
async def predict(input_data: PredictionInput):
    """
    Realiza a predição da classe da flor Iris com base nas características.
    """
    try:
        result = prediction_service.predict(input_data)
        return result
    except ModelNotLoadedError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Falha na predição: {str(e)}"
        )
