import jwt
from fastapi import UploadFile
from jwt.exceptions import ExpiredSignatureError, InvalidTokenError
from pydantic import BaseModel
from sqlalchemy.orm import Session
from database import Users, Classes, Students
from encryption import SECRET_KEY, ALGORITHM

class ClassesScheme(BaseModel):
    name: str
    college: str = "information technology"
    department: str = "software engineering"
    year: int

class StudentScheme(BaseModel):
    name: str
    img: UploadFile
    group: str
    college: str = "information technology"
    department: str = "software engineering"
    year: int

def decode_access_token(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload  # Contains the decoded data, e.g., {"sub": "username", "exp": 1234567890}
    except ExpiredSignatureError:
        raise Exception("Token has expired")
    except InvalidTokenError:
        raise Exception("Invalid token")

def get_user_by_token(token: str, session: Session):
    payload = decode_access_token(token)
    payload.get("sub")
    return session.query(Users).filter_by(username=payload.get("sub")).first()


# class info stuff


# create a class
def create_class_info(class_info: ClassesScheme, session: Session) -> int:
    new_class = Classes(name=class_info.name, college=class_info.college, department=class_info.department, year=class_info.year)
    session.add(new_class)
    session.commit()
    session.refresh(new_class)
    return new_class.id
# create students of a class
def upload_imagess(class_id: int, image_urls: list[str], session: Session):
    for image_url in image_urls:
        new_student = Students(class_id=class_id, image_url=image_url)
        session.add(new_student)
        session.commit()
        session.refresh(new_student)