from pydantic import BaseModel, ConfigDict


class Settings(BaseModel):
    model_config = ConfigDict(validate_assignment=True)
    jwt_secret: str = "change-this-secret"
    jwt_algorithm: str = "HS256"
    token_exp_minutes: int = 120
    db_url: str = "sqlite:///transport.db"
    yolo_model_path: str = "yolo26m.pt"
    yolo_conf_threshold: float = 0.5
    # Reserved seat ROI as fractions of frame dimensions (0.0 - 1.0)
    roi_x1: float = 0.0
    roi_y1: float = 0.0
    roi_x2: float = 1.0
    roi_y2: float = 1.0
    # Violence classifier
    violence_model_path: str = "models/violence_classifier.pth"
    violence_threshold: float = 0.6


settings = Settings()
