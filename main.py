from contextlib import asynccontextmanager
from datetime import timedelta

from certifi import where
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware
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

# CROS settings

origins = [
    "http://127.0.0.1:3000",  # If your Electron app runs on a local server
    "http://localhost:5173",
    "http://127.0.0.1:8000",  # Allow API itself (if making internal requests)
    "file://",  # Electron apps might run from file://
]



app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Allow specific origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)



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
    print(users)
    return {"user": users}

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

# request form


class LoginRequest(BaseModel):
    username: str
    password: str

@app.post("/login")
def login(form_data: LoginRequest, db: Session = Depends(get_session)):
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

# delete classes
@app.delete("/classes/delete/{class_id}")
async def delete_class(class_id: int, db: Session = Depends(get_session)):
    # Fetch the class by ID
    target_class = db.query(Classes).filter(Classes.id == class_id).first()
    print(target_class)
    if not target_class:
        raise HTTPException(status_code=404, detail="Class not found")

    # Optional: delete students associated with this class
    db.query(Students).filter(Students.class_id == class_id).delete()

    # Delete the class
    db.delete(target_class)
    db.commit()

    return {"status": "success", "message": f"Class with ID {class_id} has been deleted"}

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

from fastapi import File, Form, UploadFile

@app.post("/students/upload")
async def upload_student(
    name: str = Form(...),
    group: str = Form(...),
    college: str = Form("information technology"),
    department: str = Form("software engineering"),
    year: int = Form(...),
    class_id: int = Form(...),
    image: UploadFile = File(None),
    db: Session = Depends(get_session)
):
    print(f"Received student: {name}")

    image_path = "storage/images/default.jpg"
    if image:
        image_path = add_image_to_fs(image)

    new_student = Students(
        name=name,
        group=group,
        college=college,
        department=department,
        year=year,
        class_id=class_id,
        Image_URI=image_path,
    )
    db.add(new_student)
    db.commit()
    db.refresh(new_student)

    return {"status": "success", "student_id": new_student.id}
