from pydantic import BaseModel, Field

class PredictionInput(BaseModel):
    sepal_length: float = Field(..., gt=0, example=5.1)
    sepal_width: float = Field(..., gt=0, example=3.5)
    petal_length: float = Field(..., gt=0, example=1.4)
    petal_width: float = Field(..., gt=0, example=0.2)

class PredictionOutput(BaseModel):
    class_id: int
    class_name: str
