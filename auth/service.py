from datetime import datetime, timedelta
from typing import Optional, cast
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.orm import Session
from auth.models import User
from auth.schemas import TokenData
from dotenv import load_dotenv
import os

load_dotenv()

_secret_key = cast(str, os.getenv("SECRET_KEY"))
if _secret_key is None:
    raise RuntimeError("SECRET_KEY is required")

SECRET_KEY: str = _secret_key

ALGORITHM = os.getenv("ALGORITHM", "HS256")

ACCESS_TOKEN_EXPIRY_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRY_MINUTES", 30))

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(password: str) -> str:
    """Converts plain text password to bcrypt hash."""
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verifies a plain text password against a stored hash."""
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """
    Creates a JWT token containing the provided data.
    Token expires after ACCESS_TOKEN_EXPIRE_MINUTES by default.
    """
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRY_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def decode_token(token: str) -> Optional[TokenData]:
    """
    Decodes and validates a JWT token.
    Returns TokenData if valid, None if invalid or expired.
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email = payload.get("sub")

        if email is None:
            return None
        return TokenData(email=email)
    except JWTError:
        return None

def get_user_by_email(db: Session, email: str) -> Optional[User]:
    return db.query(User).filter(User.email == email).first()

def create_user(db: Session, email: str, password: str, full_name: str) -> User:
    """Creates a new user with hashed password."""
    if get_user_by_email(db, email):
        raise ValueError("Email already registered")

    hashed = hash_password(password)
    user = User(
        email=email,
        hashed_password=hashed,
        full_name=full_name
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user

def authenticate_user(db: Session, email: str, password: str) -> Optional[User]:
    """
    Verifies email and password combination.
    Returns user if valid, None if invalid.
    """
    user = get_user_by_email(db, email)
    if not user:
        return None
    if not verify_password(password, str(user.hashed_password)):
        return None
    return user