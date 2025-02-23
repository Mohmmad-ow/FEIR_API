from datetime import timedelta, datetime, timezone

import jwt
import passlib.context
import os
from jwt.exceptions import InvalidTokenError
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sqlmodel import select

from database import Users

from database import Users, get_session

SECRET_KEY = os.environ.get('SECRET_KEY', 'secret')
ALGORITHM = "HS256"


class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: str | None = None

class User(BaseModel):
    username: str
    password: str
    email: str
    full_name: str | None = None
    isAdmin: bool = False

pwt_context = passlib.context.CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def verify_password(plain_password, hashed_password):
    print(f"plain password: {plain_password}")
    print(f"hashed password: {hashed_password}")
    return pwt_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwt_context.hash(password)

def authenticate_user(username: str, password: str, session: Session):
    # TODO -> find a way to get the user
    statement = select(Users).where(Users.username == username)
    users = session.execute(statement)
    user = users.first()[0]

    print(f"user found: {user}")
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user

def create_access_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=15)
    print(f"expire: {expire}, and so far so good.")
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(payload=to_encode, key=SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt
