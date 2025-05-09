from fastapi.params import Depends
from sqlalchemy import create_engine
from sqlmodel import Field, Relationship, SQLModel, Session
from datetime import datetime
from typing import List, Optional, Annotated


class Users(SQLModel, table=True):
    id: Optional[int] = Field(unique=True, primary_key=True, nullable=False)
    username: str = Field(unique=True, nullable=False)
    full_name: Optional[str] = Field(unique=False, nullable=True)
    isAdmin: bool = Field(default=False, nullable=False)
    hashed_password: str = Field(nullable=False)

    # Relationships
    records: List["Records"] = Relationship(back_populates="user")

from sqlmodel import SQLModel, Field, Relationship
from typing import List, Optional

class StudentClassLink(SQLModel, table=True):
    student_id: Optional[int] = Field(
        default=None, foreign_key="students.id", primary_key=True
    )
    class_id: Optional[int] = Field(
        default=None, foreign_key="classes.id", primary_key=True
    )
    group_name: Optional[str] = Field(default=None, alias="group", nullable=True)

    student: Optional["Students"] = Relationship(back_populates="student_links")
    class_: Optional["Classes"] = Relationship(back_populates="class_links")

# Updated Classes model
class Classes(SQLModel, table=True):
    id: Optional[int] = Field(unique=True, primary_key=True, nullable=False)
    name: str = Field(nullable=False)
    college: Optional[str] = Field(default="Information Technology", nullable=True)
    department: Optional[str] = Field(default="Software", nullable=True)
    year: Optional[int] = Field(default=1, nullable=False)

    students: List["Students"] = Relationship(
        back_populates="classes",
        link_model=StudentClassLink,
    )

    # Add this explicit link
    class_links: List[StudentClassLink] = Relationship(back_populates="class_", cascade_delete=True)

    records: List["Records"] = Relationship(back_populates="class_", cascade_delete=True)

# Updated Students model
class Students(SQLModel, table=True):
    id: Optional[int] = Field(unique=True, primary_key=True, nullable=False)
    name: str = Field(nullable=False)
    Image_URI: str = Field(nullable=False)

    classes: List["Classes"] = Relationship(
        back_populates="students",
        link_model=StudentClassLink,
    )

    # Add this explicit link
    student_links: List[StudentClassLink] = Relationship(back_populates="student", cascade_delete=True)

    attendances: List["Attendances"] = Relationship(back_populates="student", cascade_delete=True)

class Records(SQLModel, table=True):
    id: Optional[int] = Field(unique=True, primary_key=True, nullable=False)
    date_created: datetime = Field(default_factory=datetime.now, nullable=False)

    # Foreign Keys
    class_id: Optional[int] = Field(foreign_key="classes.id", nullable=False)
    user_id: Optional[int] = Field(foreign_key="users.id", nullable=False)

    # Relationships
    class_: Optional[Classes] = Relationship(back_populates="records")
    user: Optional[Users] = Relationship(back_populates="records")
    attendances: List["Attendances"] = Relationship(back_populates="record", cascade_delete=True)


class Attendances(SQLModel, table=True):
    id: Optional[int] = Field(unique=True, primary_key=True, nullable=False)
    isPresent: bool = Field(default=False, nullable=False)
    hours: Optional[int] = Field(default=0, nullable=True)

    # Foreign Keys
    student_id: Optional[int] = Field(foreign_key="students.id", nullable=False)
    record_id: Optional[int] = Field(foreign_key="records.id", nullable=False)

    # Relationships
    student: Optional[Students] = Relationship(back_populates="attendances")
    record: Optional[Records] = Relationship(back_populates="attendances")






# database engine

sqlite_file_name = "FEIR.db"
sqlite_url = f"sqlite:///{sqlite_file_name}"



engine = create_engine(sqlite_url, echo=True)


def create_db_and_tables():
    SQLModel.metadata.create_all(engine)


def get_session():
    with Session(engine) as session:
        yield session



SQLModel.metadata.create_all(engine)

Session_Dep = Annotated[Session, Depends(get_session)]
