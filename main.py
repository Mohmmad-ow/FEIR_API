
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from typing_extensions import Annotated
from pydantic import BaseModel


app = FastAPI()

authScheme = OAuth2PasswordBearer(tokenUrl="token")


fake_users_db = {
    "johndoe": {
        "username": "johndoe",
        "full_name": "John Doe",
        "email": "johndoe@example.com",
        "hashed_password": "fakehashedsecret",
        "disabled": False,
    },
    "alice": {
        "username": "alice",
        "full_name": "Alice Wonderson",
        "email": "alice@example.com",
        "hashed_password": "fakehashedsecret2",
        "disabled": True,
    },
}




class User(BaseModel):
    username: str
    email: str | None = None
    full_name: str | None = None
    disabled: bool | None = None

class UserInDB(User):
    hashed_password: str

def get_user(db: dict,username: str):
    dict_user = db.get(username)
    return UserInDB(**dict_user) if dict_user else None

def fake_decode_token(token: str):
    user = get_user(fake_users_db, token)
    return user

def fake_hash_password(password: str):
    return "fakehashed" + password

async def get_current_user(token: Annotated[str, Depends(authScheme)]):
    print(f"token : {token}")
    user = fake_decode_token(token)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"}
        )
    return user

async def get_current_active_user(current_user: Annotated[User, Depends(get_current_user)]):
    print(f"current user: {current_user}")
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="User Disabled")
    return current_user

@app.post("/token")
async def login(form_data: Annotated[OAuth2PasswordRequestForm, Depends()]):
    user_dict = fake_users_db.get(form_data.username)
    if not user_dict:
        raise HTTPException(
            status_code=400, detail="Incorrect username or password"
        )
    user = UserInDB(**user_dict)
    hashed_password = fake_hash_password(form_data.password)
    if not user.hashed_password == hashed_password:
        raise HTTPException(
            status_code=400, detail="Incorrect username or password"
        )
    return {"access_token": user.username, "token_type": "bearer"}

@app.get("/")
def read_root():
    return {"Hello": "World2.0"}

@app.get("/items/{item_id}")
def read_item(token: Annotated[str, Depends(authScheme)]):
    return {"token": token}

@app.get("/users/me")
async def read_users_me(current_user: Annotated[User, Depends(get_current_active_user)]):
    return current_user