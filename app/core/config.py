import logging
import os
import secrets

from pydantic import BaseModel, ConfigDict

logger = logging.getLogger("transport.config")


def _jwt_secret() -> str:
    value = os.getenv("JWT_SECRET")
    if value:
        return value
    # No shared default: a random per-process secret means tokens die on restart, which is safe.
    logger.warning("JWT_SECRET not set; using a random secret (tokens reset on restart).")
    return secrets.token_urlsafe(32)


class Settings(BaseModel):
    model_config = ConfigDict(validate_assignment=True)
    jwt_secret: str = _jwt_secret()
    jwt_algorithm: str = "HS256"
    token_exp_minutes: int = int(os.getenv("TOKEN_EXP_MINUTES", "120"))
    require_2fa: bool = os.getenv("REQUIRE_2FA", "true").lower() != "false"
    db_url: str = os.getenv("DB_URL", "sqlite:///transport.db")
    yolo_model_path: str = os.getenv("YOLO_MODEL", "yolo26m.pt")
    yolo_conf_threshold: float = 0.5
    # Reserved seat ROI as fractions of frame dimensions (0.0 - 1.0)
    roi_x1: float = 0.0
    roi_y1: float = 0.0
    roi_x2: float = 1.0
    roi_y2: float = 1.0
    # Violence classifier
    violence_model_path: str = os.getenv("VIOLENCE_MODEL", "models/temporal_pyramid_network.pth")
    violence_threshold: float = 0.6


settings = Settings()
