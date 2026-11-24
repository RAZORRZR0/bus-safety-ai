from datetime import datetime, timedelta, timezone
from typing import Optional

import jwt
import pyotp
from fastapi import HTTPException, status
from passlib.context import CryptContext

from app.core.config import settings

pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(password: str, hashed: str) -> bool:
    return pwd_context.verify(password, hashed)


PRE_2FA_TOKEN_MINUTES = 5
STREAM_TOKEN_SECONDS = 60


def create_access_token(subject: str, mfa: bool = True) -> str:
    """`mfa` marks whether the second factor was passed; only such tokens open the API.
    A pre-2FA token is only good for /auth/verify-2fa, so it expires quickly."""
    minutes = settings.token_exp_minutes if mfa else PRE_2FA_TOKEN_MINUTES
    expire = datetime.now(timezone.utc) + timedelta(minutes=minutes)
    payload = {"sub": subject, "exp": expire, "mfa": mfa, "scope": "api"}
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def decode_access_token(token: str) -> dict:
    try:
        payload = jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
        if not payload.get("sub"):
            raise ValueError("Missing sub claim.")
        return payload
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token."
        ) from exc


def create_stream_token(subject: str) -> str:
    """Short-lived token for the MJPEG stream, which an <img> tag can only pass in the URL."""
    expire = datetime.now(timezone.utc) + timedelta(seconds=STREAM_TOKEN_SECONDS)
    return jwt.encode({"sub": subject, "exp": expire, "scope": "stream"},
                      settings.jwt_secret, algorithm=settings.jwt_algorithm)


def generate_2fa_secret() -> str:
    return pyotp.random_base32()


def verify_2fa_code(secret: str, code: str, valid_window: int = 1) -> bool:
    totp = pyotp.TOTP(secret)
    return bool(totp.verify(code, valid_window=valid_window))


def provisioning_uri(secret: str, username: str, issuer: Optional[str] = None) -> str:
    label = issuer or "TransportPolicing"
    return pyotp.TOTP(secret).provisioning_uri(name=username, issuer_name=label)
