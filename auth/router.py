from fastapi import APIRouter, HTTPException, Depends, Request
from fastapi.security import OAuth2PasswordBearer, HTTPBearer
from sqlalchemy.orm import Session
from database import get_db
from auth.schemas import UserRegister, UserLogin, UserResponse, Token
from auth.service import (
    create_user, authenticate_user, create_access_token, decode_token, get_user_by_email
)
from auth.models import User
from slowapi import Limiter
from slowapi.util import get_remote_address

router = APIRouter(prefix="/auth", tags=["Authentication"])

limiter = Limiter(key_func=get_remote_address)

oauth2_scheme = HTTPBearer()

def get_current_user(
    credentials = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
) -> User:
    token = credentials.credentials
    
    credentials_exception = HTTPException(status_code=401, detail="Could not validate credentials", headers={"WWW-Authenticate": "Bearer"})

    token_data = decode_token(token)
    if token_data is None:
        raise credentials_exception
    
    if token_data.email is None:
        raise credentials_exception
    
    user = get_user_by_email(db, token_data.email)

    if user is None:
        raise credentials_exception

    if user.is_active is False:
        raise HTTPException(status_code=400, detail="Inactive user")

    return user
    
@router.post("/register", response_model=UserResponse, status_code=201)
@limiter.limit("5/minute")
def register(request: Request, user_data: UserRegister, db: Session = Depends(get_db)):
    try:
        user = create_user(
            db, 
            email=user_data.email,
            password=user_data.password,
            full_name=user_data.full_name
        )
        return user
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/login", response_model=Token)
@limiter.limit("5/minute")
def login(request: Request, user_data: UserLogin, db: Session = Depends(get_db)):
    user = authenticate_user(db, user_data.email, user_data.password)
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    access_token = create_access_token(data={"sub": user.email})
    return {"access_token": access_token, "token_type": "bearer"}

@router.get("/me", response_model=UserResponse)
def get_me(current_user: User = Depends(get_current_user)):
    return current_user
