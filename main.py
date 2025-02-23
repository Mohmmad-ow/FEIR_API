from contextlib import asynccontextmanager
from datetime import timedelta

from pydantic import BaseModel
from starlette.responses import RedirectResponse

from utils import add_image_to_fs
from fastapi import FastAPI, Depends, HTTPException, status, UploadFile
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy import select
from sqlalchemy.orm import Session
from typing_extensions import Annotated
from database import create_db_and_tables, engine, get_session, Users, Classes, Students
from encryption import oauth2_scheme, authenticate_user, create_access_token, User, get_password_hash
from middleware import get_user_by_token
# from types import ClassesScheme, StudentScheme
class ClassesScheme(BaseModel):
    name: str
    college: str = "information technology"
    department: str = "software engineering"
    year: int

class StudentScheme(BaseModel):
    name: str
    img: str | None = "storage/images/photo_2023-12-12_17-49-56.jpg"
    group: str
    college: str = "information technology"
    department: str = "software engineering"
    year: int


# random file to make a default


app = FastAPI()




@asynccontextmanager
async def lifespan(app: FastAPI):
    # Create Database, and get the session
    create_db_and_tables()

    db = get_session()

    # Clean up app when shutting down
    try:
        yield db
    finally:
        db.close()
        engine.dispose()

@app.post("/token")
def login(form_data: Annotated[OAuth2PasswordRequestForm, Depends()], db: Session = Depends(get_session)):
    print("you're trying (and probably falling) to login")
    isAuthenticated = authenticate_user(form_data.username, form_data.password, db)
    print(f"Is authenticated: {isAuthenticated}")
    if not isAuthenticated:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    else:
        print(f"User: {form_data.username} logged in successfully")

        access_token = create_access_token(data={"sub": form_data.username}, expires_delta=timedelta(minutes=15))
        return {"access_token": access_token, "token_type": "bearer"}


@app.get("/")
def root(db: Session = Depends(get_session)):
    users = db.query(Users).all()
    return {"Hello": "world", "users": users}

@app.post("/register")
async def register(user: User, db: Session = Depends(get_session)):
    print(f"Here is what you sent: {user}")
    hashed_password = get_password_hash(user.password)
    created_user = Users(username=user.username, hashed_password=hashed_password, email=user.email, isAdmin=user.isAdmin)
    db.add(created_user)
    db.commit()
    db.refresh(created_user)
    print(f"User: {created_user.username} created successfully")
    return created_user

@app.post("/login")
def login(form_data: Annotated[OAuth2PasswordRequestForm, Depends()], db: Session = Depends(get_session)):
    isAuthed = authenticate_user(form_data.username, form_data.password, db)
    print(f"Is authed: {isAuthed}")
    return {"username": form_data.username, "password": form_data.password}


@app.get("/users/me")
async def get_current_user(token: Annotated[str, Depends(oauth2_scheme)], db: Session = Depends(get_session)):
    return get_user_by_token(token, session=db)

# Create Classes (including class info)
@app.post("/classes/create")
def create_class(class_info: ClassesScheme, db: Session = Depends(get_session)):
    print(f"Here is what you sent: {class_info}")
    new_class = Classes(name=class_info.name, college=class_info.college, department=class_info.department,
                        year=class_info.year)
    db.add(new_class)
    db.commit()
    db.refresh(new_class)
    print(f"Class: {new_class.name} created successfully")
    # return RedirectResponse(url="/students/add?class_id={}".format(new_class.id))
    return {new_class.id: new_class.name, "status": "success"}

@app.get("/classes/all")
async def get_all_classes(db: Session = Depends(get_session)):
    classes = db.query(Classes).all()
    return classes

@app.get("/classes/{class_id}")
@app.get("/classes/{class_id}")
async def view_class(class_id: int, db: Session = Depends(get_session)):
    print(f"Class ID: {class_id}")

    statement = (
        select(Students, Classes)
        .join(Classes, Students.class_id == Classes.id)
        .where(Classes.id == class_id)
    )
    result = db.execute(statement).all()

    # Convert results into dictionaries
    data = [
        {
            "student": student.__dict__,
            "class": class_.__dict__
        }
        for student, class_ in result
    ]

    return data

# get students

@app.get("/students/all")
def get_all_students(db: Session = Depends(get_session)):
    return db.query(Students).all()


# create students
@app.post("/students/add/multiple")
def add_student(student_info: list[StudentScheme], db: Session = Depends(get_session), class_id: int = None):
    print(f"Here is what you sent: {student_info}")
    if len(student_info) > 80:
        raise HTTPException(status_code=400, detail="Too many students")
    else:
        for student in student_info:
            path = add_image_to_fs(student.img)
            new_student = Students(class_id=class_id, image_url=path, name=student.name,department=student.department,college=student.college,year=student.year)
            db.add(new_student)
            db.commit()
            db.refresh(new_student)
            print(f"Student: {new_student.username} created successfully")
        return RedirectResponse(url="/classes/{}".format(class_id))

@app.post("/students/add/single")
def add_student(student_info: StudentScheme, db: Session = Depends(get_session), class_id: int = None):
    print(f"Here is what you sent: {student_info}")
    if not student_info:
        raise HTTPException(status_code=400, detail="No Info Sent")
    else:
        path = student_info.img
        print(f"image is: {student_info.img}")
        if not student_info.img:
            path = "storage/images/photo_2023-12-12_17-49-56.jpg"
        # else:
        #     path = add_image_to_fs(student_info.img)
        new_student = Students( class_id=class_id, Image_URI=path, name=student_info.name,department=student_info.department,
                                college=student_info.college,year=student_info.year, group=student_info.group)
        db.add(new_student)
        db.commit()
        db.refresh(new_student)
        print(f"Student: {new_student.name} created successfully")
        return RedirectResponse(url="/classes/{}".format(class_id))

