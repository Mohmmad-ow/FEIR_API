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


class Classes(SQLModel, table=True):
    id: Optional[int] = Field(unique=True, primary_key=True, nullable=False)
    name: str = Field(nullable=False)
    college: Optional[str] = Field(default="Information Technology", nullable=True)
    department: Optional[str] = Field(default="Software", nullable=True)
    year: Optional[int] = Field(default=1, nullable=False)

    # Relationships
    students: List["Students"] = Relationship(back_populates="class_")
    records: List["Records"] = Relationship(back_populates="class_")


class Students(SQLModel, table=True):
    id: Optional[int] = Field(unique=True, primary_key=True, nullable=False)
    name: str = Field(nullable=False)
    group: Optional[str] = Field(default="A", nullable=False)
    college: Optional[str] = Field(default="Information Technology", nullable=True)
    department: Optional[str] = Field(default="Software", nullable=True)
    year: Optional[int] = Field(default=1, nullable=False)
    Image_URI: str = Field(nullable=False)

    # Foreign Key
    class_id: Optional[int] = Field(foreign_key="classes.id", nullable=False)

    # Relationships
    class_: Optional[Classes] = Relationship(back_populates="students")
    attendances: List["Attendances"] = Relationship(back_populates="student")


class Records(SQLModel, table=True):
    id: Optional[int] = Field(unique=True, primary_key=True, nullable=False)
    date_created: datetime = Field(default_factory=datetime.now, nullable=False)

    # Foreign Keys
    class_id: Optional[int] = Field(foreign_key="classes.id", nullable=False)
    user_id: Optional[int] = Field(foreign_key="users.id", nullable=False)

    # Relationships
    class_: Optional[Classes] = Relationship(back_populates="records")
    user: Optional[Users] = Relationship(back_populates="records")
    attendances: List["Attendances"] = Relationship(back_populates="record")


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
