from contextlib import asynccontextmanager
from datetime import timedelta
from typing import List, Any

from pydantic import BaseModel
from sqlalchemy.exc import SQLAlchemyError
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import RedirectResponse, JSONResponse

from utils import add_image_to_fs
from fastapi import FastAPI, HTTPException, status, Body
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy import select
from typing_extensions import Annotated
from database import create_db_and_tables, engine, get_session, Users, Classes, Students, Records, Attendances
from encryption import oauth2_scheme, authenticate_user, create_access_token, User, get_password_hash
from middleware import get_user_by_token
from fastapi.staticfiles import StaticFiles




# AI related imports
import cv2
from fastapi import Depends
from sqlalchemy.orm import Session, joinedload
from deepface import DeepFace
import numpy as np
import os
from scipy.spatial.distance import cosine

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



app.mount("/images", StaticFiles(directory="storage/images"), name="images")
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





@app.get("/classes-students/all")
async def view_class(db: Session = Depends(get_session)):
    statement = (
        select(Students, Classes)
        .join(Classes, Students.class_id == Classes.id)
    )
    results = db.execute(statement).all()

    # Group students by class
    class_map = {}
    for student, class_ in results:
        class_id = class_.id
        student_dict = {
            **student.__dict__,
            "image_url": f"http://localhost:8000/images/{os.path.basename(student.Image_URI)}"
        }

        if class_id not in class_map:
            class_map[class_id] = {
                "class": class_.__dict__,
                "students": []
            }

        class_map[class_id]["students"].append(student_dict)

    # Remove SQLAlchemy state from class dicts
    for entry in class_map.values():
        entry["class"].pop("_sa_instance_state", None)
        for student in entry["students"]:
            student.pop("_sa_instance_state", None)

    return list(class_map.values())



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
            "student": {
                **student.__dict__,
                "image_url": f"http://localhost:8000/images/{os.path.basename(student.Image_URI)}"
            },
            "class": class_.__dict__
        }
        for student, class_ in result
    ]

    return data


# delete classes
@app.delete("/classes/delete/{class_id}")
async def delete_class(class_id: int, db: Session = Depends(get_session)):
    IMAGE_DIRECTORY = "storage/images"
    # Fetch the class by ID
    target_class = db.query(Classes).filter(Classes.id == class_id).first()
    if not target_class:
        raise HTTPException(status_code=404, detail="Class not found")

    # Fetch and delete student images
    students = db.query(Students).filter(Students.class_id == class_id).all()
    for student in students:
        if student.Image_URI:  # Make sure your Student model has this attribute
            image_path = os.path.join(IMAGE_DIRECTORY, student.Image_URI)
            if os.path.exists(image_path):
                os.remove(image_path)

    # Delete student records
    db.query(Students).filter(Students.class_id == class_id).delete()

    # Delete the class
    db.delete(target_class)
    db.commit()

    return {"status": "success", "message": f"Class with ID {class_id} has been deleted"}

# edit class name
@app.put("/classes/edit/{class_id}")
def edit_class(class_info: ClassesScheme, class_id: int, db: Session = Depends(get_session)):
    # Fetch the class by ID
    target_class = db.query(Classes).filter(Classes.id == class_id).first()
    print(target_class)
    if not target_class:
        raise HTTPException(status_code=404, detail="Class not found")

    # ✅ Update fields
    target_class.name = class_info.name
    target_class.college = class_info.college
    target_class.department = class_info.department
    target_class.year = class_info.year

    # ✅ Commit changes
    db.commit()
    db.refresh(target_class)

    return {
        "status": "success",
        "message": f"Class with ID {class_id} updated successfully",
        "updated_class": {
            "id": target_class.id,
            "name": target_class.name,
            "college": target_class.college,
            "department": target_class.department,
            "year": target_class.year
        }
    }

# get students
app_host = "http://127.0.0.1:8000"  # or use environment variable/config

@app.get("/students/all")
def get_all_students(db: Session = Depends(get_session)):
    students = db.query(Students).all()
    results = []

    for student in students:
        print(f'student image uri: {student.Image_URI}')
        results.append({
            "id": student.id,
            "name": student.name,
            "class_id": student.class_id,
            "college": student.college,
            "department": student.department,
            "year": student.year,
            "group": student.group,
            "image_uri": f"{app_host}/images/{student.Image_URI}" if student.Image_URI else None
        })

    return JSONResponse(content=results)

# create students
import uuid
from fastapi import Form, File, UploadFile, Depends
from sqlalchemy.orm import Session

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
    image_path = "storage/images/default.svg"

    if image:
        try:
            # Step 1: Read uploaded image into NumPy array (BGR)
            img_bytes = await image.read()
            np_arr = np.frombuffer(img_bytes, np.uint8)
            bgr_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            # Step 2: Convert to RGB for DeepFace
            rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

            # Step 3: Extract faces using DeepFace
            faces = DeepFace.extract_faces(
                rgb_img,
                detector_backend="opencv",  # or 'opencv', 'mtcnn', etc.
                enforce_detection=True
            )

            if not faces:
                raise Exception("No face detected.")

            # Step 4: Convert and save the first detected face
            face_rgb = faces[0]["face"]

            # 🧠 Normalize to 0–255 if it's in 0–1 range
            if face_rgb.max() <= 1.0:
                face_rgb *= 255.0

            face_rgb_uint8 = face_rgb.astype(np.uint8)
            face_bgr = cv2.cvtColor(face_rgb_uint8, cv2.COLOR_RGB2BGR)

            os.makedirs("storage/images", exist_ok=True)
            filename = f"{uuid.uuid4().hex[:12]}.jpg"
            face_path = os.path.join("storage/images", filename)
            cv2.imwrite(face_path, face_rgb)

            image_path = filename

        except Exception as e:
            print(f"⚠️ Face extraction failed: {e}")
            image_path = "storage/images/default.svg"

    # Step 5: Save to DB
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

    return {
        "status": "success",
        "student_id": new_student.id,
        "face_image": image_path
    }


class DeleteStudentsRequest(BaseModel):
    students: List[int]

@app.delete("/students/delete")
async def delete_students(

    payload: DeleteStudentsRequest,
    db: Session = Depends(get_session)
):
    deleted_students = []

    for student_id in payload.students:
        student = db.query(Students).filter(
            Students.id == student_id
        ).first()
        print(f'found student: {student}')
        if not student:
            continue

        # Delete image if exists
        if student.Image_URI and os.path.exists(student.Image_URI):
            try:
                os.remove(student.Image_URI)
            except Exception as e:
                print(f"Error deleting image for student {student.name}: {e}")
        else:
            print(f"couldn't find this image: {student.Image_URI}")
        db.delete(student)
        deleted_students.append(student.name)

    db.commit()

    return {
        "message": f"{len(deleted_students)} student(s) deleted.",
        "deleted": deleted_students
    }

def embed_face_into_canvas(face, canvas_size=(160, 160)):
    # Make sure face is uint8
    if face.max() <= 1.0:
        face = (face * 255).astype(np.uint8)
    else:
        face = face.astype(np.uint8)

    canvas = np.zeros((canvas_size[0], canvas_size[1], 3), dtype=np.uint8)

    # Resize face to fit canvas
    face_resized = cv2.resize(face, (canvas_size[1], canvas_size[0]))

    # Place the resized face in canvas
    canvas[:canvas_size[0], :canvas_size[1]] = face_resized

    return canvas


@app.post("/attendance/recognize")
async def recognize_students(
    class_id: int = Form(...),
    classroom_image: UploadFile = File(...),
    db: Session = Depends(get_session)
):
    # Save uploaded classroom image
    os.makedirs("temp", exist_ok=True)
    classroom_img_path = f"temp/classroom_{class_id}.jpg"
    with open(classroom_img_path, "wb") as f:
        f.write(await classroom_image.read())

    # Fetch student images from DB
    students = db.query(Students).filter(Students.class_id == class_id).all()
    student_faces = []
    for s in students:
        image_path = s.Image_URI
        if not os.path.isabs(image_path):
            image_path = os.path.join("storage/images", image_path) if not image_path.startswith("storage/") else image_path
        if os.path.exists(image_path):
            student_faces.append({"id": s.id, "name": s.name, "img": image_path})

    if not student_faces:
        return {"error": "No valid student images found."}

    # Detect faces from classroom image
    try:
        detected_faces = DeepFace.extract_faces(
            classroom_img_path,
            detector_backend="opencv",
            enforce_detection=False
        )
    except Exception as e:
        return {"error": f"Failed to extract faces: {str(e)}"}

    if not detected_faces:
        return {"error": "No faces detected in classroom image."}

    print(f"🔍 Detected {len(detected_faces)} face(s) in classroom image.")

    present_students = set()
    threshold = 0.5  # Tune this

    for i, face_data in enumerate(detected_faces):
        face_img = face_data["face"]

        try:
            face_img = embed_face_into_canvas(face_img)
            embedding1 = DeepFace.represent(
                face_img,
                model_name="Facenet512",  # <- more accurate
                enforce_detection=False
            )[0]["embedding"]
        except Exception as e:
            print(f"❌ Could not get embedding for face #{i}: {e}")
            continue

        for student in student_faces:
            try:
                embedding2 = DeepFace.represent(
                    img_path=student["img"],
                    model_name="Facenet512",
                    enforce_detection=True
                )[0]["embedding"]
                distance = cosine(embedding1, embedding2)
                print(f"Distance to {student['name']}: {distance:.4f}")

                if distance < threshold:
                    print(f"✅ Matched with {student['name']} (ID: {student['id']})")
                    present_students.add(student["id"])
                    break

            except Exception as e:
                print(f"⚠️ Error with {student['name']}: {e}")
                continue

    # Split present and absent
    present_list = [s for s in student_faces if s["id"] in present_students]
    absent_list = [s for s in student_faces if s["id"] not in present_students]

    # Cleanup
    if os.path.exists(classroom_img_path):
        os.remove(classroom_img_path)

    results = {
        "present": present_list,
        "absent": absent_list,
        "total_detected_faces": len(detected_faces),
        "total_students": len(student_faces),
        "matched_count": len(present_list)
    }

    return results

@app.get("/records/all")
def get_all_records(db: Session = Depends(get_session)):
    records = db.query(Records).join(Attendances, Records.id == Attendances.record_id).all()

    return records
class AttendanceStudents(BaseModel):
    student_id: int
    isPresent: bool
    hours: float

class CreateRecordPayload(BaseModel):
    class_id: int
    student_list: List[AttendanceStudents]

@app.post("/records/create")
def create_record(
    token: Annotated[str, Depends(oauth2_scheme)],
    payload: CreateRecordPayload = Body(...),
    db: Session = Depends(get_session)
):
    try:
        user = get_user_by_token(token, session=db)
        if not user:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

        # Create the attendance record
        new_record = Records(class_id=payload.class_id, user_id=user.id)
        db.add(new_record)
        db.commit()
        db.refresh(new_record)

        # Create all attendance entries
        attendances = [
            Attendances(
                isPresent=entry.isPresent,
                student_id=entry.student_id,
                record_id=new_record.id
            )
            for entry in payload.student_list
        ]
        db.add_all(attendances)
        db.commit()

        return {
            "message": "Attendance record created successfully",
            "record_id": new_record.id,
            "attendances": [a.student_id for a in attendances]
        }

    except SQLAlchemyError as e:
        db.rollback()
        print("Database error:", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database error while creating record"
        )

    except Exception as e:
        db.rollback()
        print("Unexpected error:", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred"
        )



@app.get("/records/summary")
def get_records_summary(db: Session = Depends(get_session)):
    records = db.query(Records).options(
        joinedload(Records.class_),
        joinedload(Records.attendances).joinedload(Attendances.student)
    ).all()

    summary = []
    # print(f'These are the records: {records}')
    for record in records:
        total_students = len(record.attendances)
        print(f'class data: {record.class_}')
        attended_students = sum(1 for att in record.attendances if att.isPresent)
        attendance_percentage = (
            round((attended_students / total_students) * 100, 2)
            if total_students > 0 else 0
        )

        summary.append({
            "record_id": record.id,
            "date_created": record.date_created,
            "class": record.class_,
            "class_id": record.class_id,
            "user_id": record.user_id,
            "total_students": total_students,
            "attended_students": attended_students,
            "attendance_percentage": attendance_percentage,
            "students": [
                {
                    "id": att.student.id,
                    "name": att.student.name,
                    "isPresent": att.isPresent,
                    "hours": att.hours
                }
                for att in record.attendances
            ]
        })

    return summary
