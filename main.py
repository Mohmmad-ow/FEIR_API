from collections import defaultdict
from contextlib import asynccontextmanager
from datetime import timedelta, datetime
from typing import List, Optional, Dict

from moviepy import VideoFileClip
from pydantic import BaseModel, Field
from sqlalchemy.exc import SQLAlchemyError
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import  JSONResponse
import uuid
from fastapi import Form, File, UploadFile



from fastapi import FastAPI, status, Body
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy import select
from typing_extensions import Annotated
from database import create_db_and_tables, engine, get_session, Users, Classes, Students, Records, Attendances, \
    StudentClassLink
from encryption import oauth2_scheme, authenticate_user, create_access_token, User, get_password_hash
from middleware import get_user_by_token
from fastapi.staticfiles import StaticFiles




# AI related imports
import cv2
from fastapi import Depends
from sqlalchemy.orm import Session, joinedload, selectinload
from deepface import DeepFace
import numpy as np
from scipy.spatial.distance import cosine

class ClassesScheme(BaseModel):
    name: str
    college: str = "information technology"
    department: str = "software engineering"
    year: int

class StudentScheme(BaseModel):
    name: str
    img: str | None = "default.svg"
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

        access_token = create_access_token(data={"sub": form_data.username}, expires_delta=timedelta(hours=4))
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

@app.get("/classes/full-info")
def get_classes_with_students(session: Session = Depends(get_session)):
    all_classes = session.query(Classes).all()
    response = []

    for cls in all_classes:
        student_links = session.query(StudentClassLink).filter(StudentClassLink.class_id == cls.id).all()

        students_info = []
        group_counts: Dict[str, int] = defaultdict(int)

        for link in student_links:
            student = session.query(Students).filter(Students.id == link.student_id).first()
            group_name = link.group_name if link.group_name else "Ungrouped"
            group_counts[group_name] += 1

            students_info.append({
                "id": student.id,
                "name": student.name,
                "group": link.group_name,
                "image_uri":f"{app_host}/images/{student.Image_URI}" if student.Image_URI else None,
            })

        response.append({
            "class_id": cls.id,
            "class_name": cls.name,
            "total_students": len(students_info),
            "group_counts": group_counts,
            "students": students_info
        })

    return response


@app.get("/classes/{class_id}")
async def view_class(class_id: int, db: Session = Depends(get_session)):
    print(f"Class ID: {class_id}")

    # Fetch class first
    target_class = db.query(Classes).filter(Classes.id == class_id).first()
    if not target_class:
        raise HTTPException(status_code=404, detail="Class not found")

    # Now fetch students linked to that class
    statement = (
        select(StudentClassLink, Students)
        .join(Students, StudentClassLink.student_id == Students.id)
        .where(StudentClassLink.class_id == class_id)
    )
    results = db.execute(statement).all()

    students = [
        {
            "id": student.id,
            "name": student.name,
            "image_url": f"http://localhost:8000/images/{os.path.basename(student.Image_URI)}" if student.Image_URI else None,
            "group": link.group_name
        }
        for link, student in results
    ]

    return {
        "class": {
            "id": target_class.id,
            "name": target_class.name,
            "college": target_class.college,
            "department": target_class.department,
            "year": target_class.year
        },
        "students": students
    }




# delete classes
@app.delete("/classes/delete/{class_id}")
async def delete_class(class_id: int, db: Session = Depends(get_session)):
    # Fetch the class by ID
    target_class = db.query(Classes).filter(Classes.id == class_id).first()
    if not target_class:
        raise HTTPException(status_code=404, detail="Class not found")

    # Delete the links between students and the class (many-to-many relationship)
    db.query(StudentClassLink).filter(StudentClassLink.class_id == class_id).delete()

    # Optionally, delete associated images here (if needed)
    # You can delete images from disk as well if needed, e.g.:
    # for student in target_class.students:
    #     student_image_path = os.path.join(IMAGE_DIRECTORY, os.path.basename(student.Image_URI))
    #     if os.path.exists(student_image_path):
    #         os.remove(student_image_path)

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
    # Use outer joins to include students without classes
    students = db.query(Students).all()

    return students

@app.get("/students/all-info")
def get_all_students(db: Session = Depends(get_session)):
    # Use outer joins to include students without classes
    results = db.query(
        Students.id,
        Students.name,
        Students.Image_URI,
        Classes.id.label("class_id"),
        Classes.name.label("class_name"),
        Classes.college,
        Classes.department,
        Classes.year,
        StudentClassLink.group_name
    ).outerjoin(
        StudentClassLink, StudentClassLink.student_id == Students.id
    ).outerjoin(
        Classes, Classes.id == StudentClassLink.class_id
    ).all()

    # Group results by student
    student_map = {}
    for row in results:
        if row.id not in student_map:
            student_map[row.id] = {
                "id": row.id,
                "name": row.name,
                "image_uri": f"{app_host}/images/{row.Image_URI}" if row.Image_URI else None,
                "classes": []
            }

        # Only append class data if it's available (i.e., the student has been assigned to a class)
        if row.class_id:
            student_map[row.id]["classes"].append({
                "class_id": row.class_id,
                "class_name": row.class_name,
                "college": row.college,
                "department": row.department,
                "year": row.year,
                "group": row.group_name
            })

    return JSONResponse(content=list(student_map.values()))

@app.get("/students/assign-to-class")
def get_classes_and_students(session: Session = Depends(get_session)):
    stmt = (
        select(Classes)
        .options(
            joinedload(Classes.class_links).joinedload(StudentClassLink.student)
        )
    )
    classes = session.execute(stmt).unique().scalars().all()

    all_students = session.query(Students).all()

    result = []
    for cls in classes:
        students = []
        for link in cls.class_links:
            if link.student:
                students.append({
                    "id": link.student.id,
                    "name": link.student.name,
                    "image_uri": link.student.Image_URI,
                    "group_name": link.group_name
                })
        result.append({
            "id": cls.id,
            "name": cls.name,
            "college": cls.college,
            "department": cls.department,
            "year": cls.year,
            "students": students
        })

    return {
        "classes": result,
        "students": [
            {
                "id": student.id,
                "name": student.name,
                "image_uri": student.Image_URI
            }
            for student in all_students
        ]
    }
 # create students


class StudentGroupMapping(BaseModel):
    student_id: int
    group_name: Optional[str] = None

class AssignStudentsToClassRequest(BaseModel):
    class_id: int
    students: List[StudentGroupMapping]

@app.post("/students/assign-to-class")
def assign_students_to_class(data: AssignStudentsToClassRequest, session: Session = Depends(get_session)):
    # Check if the class exists
    target_class = session.query(Classes).filter(Classes.id == data.class_id).first()
    if not target_class:
        raise HTTPException(status_code=404, detail="Class not found")

    # Prepare tracking lists
    successfully_assigned = []
    successfully_removed = []
    failed_assignments = []

    # --- Get all currently assigned students for this class ---
    existing_links = session.query(StudentClassLink).filter(
        StudentClassLink.class_id == data.class_id
    ).all()

    existing_student_ids = {link.student_id for link in existing_links}
    incoming_student_ids = {item.student_id for item in data.students}

    # --- Remove students no longer in the incoming list ---
    students_to_remove = [link for link in existing_links if link.student_id not in incoming_student_ids]

    for link in students_to_remove:
        try:
            session.delete(link)
            successfully_removed.append({"student_id": link.student_id})
        except Exception as e:
            failed_assignments.append({"student_id": link.student_id, "error": f"Failed to remove: {str(e)}"})

    # --- Add new students or update group if needed ---
    for item in data.students:
        student = session.query(Students).filter(Students.id == item.student_id).first()
        if not student:
            failed_assignments.append({"student_id": item.student_id, "error": "Student not found"})
            continue

        existing_link = next((l for l in existing_links if l.student_id == item.student_id), None)
        if existing_link:
            # Optional: update group if changed
            if existing_link.group_name != item.group_name:
                existing_link.group_name = item.group_name
        else:
            # New link
            new_link = StudentClassLink(
                student_id=item.student_id,
                class_id=data.class_id,
                group_name=item.group_name if item.group_name else None
            )
            session.add(new_link)
            successfully_assigned.append({"student_id": item.student_id})

    # Commit changes
    try:
        session.commit()
    except Exception as e:
        session.rollback()
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

    return {
        "message": "Class assignments updated successfully",
        "successfully_assigned": successfully_assigned,
        "successfully_removed": successfully_removed,
        "failed_assignments": failed_assignments
    }

@app.post("/students/upload")
async def upload_student(
    name: str = Form(...),
    image: UploadFile = File(None),
    db: Session = Depends(get_session)
):
    print(f"Received student: {name}")
    image_path = "storage/images/default.svg"  # Default image if no image is uploaded

    if image:
        try:
            # Step 1: Read uploaded image into NumPy array (BGR)
            img_bytes = await image.read()

            # Ensure the file is an image by checking the content type
            if not image.content_type.startswith("image/"):
                raise HTTPException(status_code=400, detail="Uploaded file is not an image.")

            np_arr = np.frombuffer(img_bytes, np.uint8)
            bgr_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if bgr_img is None:
                raise HTTPException(status_code=400, detail="Invalid image format.")

            # Step 2: Convert to RGB for DeepFace
            rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

            # Step 3: Extract faces using DeepFace
            faces = DeepFace.extract_faces(
                rgb_img,
                detector_backend="opencv",  # You can change this to other backends like 'mtcnn'
                enforce_detection=True
            )

            if not faces:
                raise HTTPException(status_code=400, detail="No face detected in the image.")

            # Step 4: Convert and save the first detected face
            face_rgb = faces[0]["face"]

            # 🧠 Normalize to 0–255 if it's in 0–1 range
            if face_rgb.max() <= 1.0:
                face_rgb *= 255.0

            face_rgb_uint8 = face_rgb.astype(np.uint8)
            # face_bgr = cv2.cvtColor(face_rgb_uint8, cv2.COLOR_RGB2BGR)

            # Ensure the 'storage/images' directory exists
            os.makedirs("storage/images", exist_ok=True)

            # Generate a unique filename, keeping the original extension
            file_extension = os.path.splitext(image.filename)[1]
            filename = f"{uuid.uuid4().hex[:12]}{file_extension}"
            face_path = os.path.join("storage/images", filename)

            # Save the face image
            cv2.imwrite(face_path, face_rgb)

            image_path = filename

        except Exception as e:
            print(f"⚠️ Face extraction failed: {e}")
            # If face extraction fails, default image will be used
            image_path = "storage/images/default.svg"

    # Step 5: Save to DB
    new_student = Students(
        name=name,
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

@app.delete("/students/delete-multiple")
async def delete_students(
        payload: DeleteStudentsRequest,
        db: Session = Depends(get_session)
):
    deleted_students = []
    image_directory = "storage/images"

    # Fetch all students in one go
    students_to_delete = db.query(Students).filter(Students.id.in_(payload.students)).all()

    if not students_to_delete:
        raise HTTPException(status_code=404, detail="No students found.")

    for student in students_to_delete:
        print(f'Found student: {student.name}')

        # Delete image if it exists
        if student.Image_URI:
            image_path = os.path.join(image_directory, student.Image_URI)
            if os.path.exists(image_path):
                try:
                    os.remove(image_path)
                    print(f"Deleted image for student {student.name}: {image_path}")
                except Exception as e:
                    print(f"Error deleting image for student {student.name}: {e}")
            else:
                print(f"Image not found for student {student.name}: {image_path}")

        # Manually delete student-class links
        links = db.query(StudentClassLink).filter(StudentClassLink.student_id == student.id).all()
        for link in links:
            print(f"Deleting link between student {student.name} and class {link.class_id}")
            db.delete(link)

        # Delete student
        db.delete(student)
        deleted_students.append({"name": student.name, "id": student.id})

    db.commit()

    return {
        "message": f"{len(deleted_students)} student(s) deleted.",
        "deleted": deleted_students
    }


import os
from fastapi import HTTPException

def embed_face_into_canvas(face, canvas_size=(160, 160)):
    # Ensure face is uint8
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


# Recognize students in classroom (from images)
@app.post("/attendance/recognize")
async def recognize_students(
    class_id: int = Form(...),
    classroom_images: List[UploadFile] = File(...),
    db: Session = Depends(get_session)
):
    os.makedirs("temp", exist_ok=True)

    # Step 1: Load student info and their stored face images
    student_links = db.query(StudentClassLink).filter(StudentClassLink.class_id == class_id).all()
    student_ids = [link.student_id for link in student_links]
    if not student_ids:
        raise HTTPException(status_code=404, detail="No students linked to this class.")
    
    students = db.query(Students).filter(Students.id.in_(student_ids)).all()
    student_faces = []
    for s in students:
        image_path = os.path.join("storage/images", s.Image_URI) if not os.path.isabs(s.Image_URI) else s.Image_URI
        if os.path.exists(image_path):
            student_faces.append({"id": s.id, "name": s.name, "img": image_path})
    if not student_faces:
        raise HTTPException(status_code=404, detail="No valid student images found.")

    # Step 2: Process each uploaded classroom image
    present_students = set()
    threshold = 0.5
    total_faces_detected = 0

    for idx, image in enumerate(classroom_images):
        img_path = f"temp/frame_{idx}.jpg"
        with open(img_path, "wb") as f:
            f.write(await image.read())

        try:
            detected_faces = DeepFace.extract_faces(
                img_path, detector_backend="opencv", enforce_detection=False
            )
        except Exception as e:
            print(f"Error extracting faces from frame {idx}: {e}")
            continue

        print(f"📸 Frame {idx}: Detected {len(detected_faces)} face(s)")
        total_faces_detected += len(detected_faces)

        for i, face_data in enumerate(detected_faces):
            face_img = embed_face_into_canvas(face_data["face"])
            try:
                embedding1 = DeepFace.represent(face_img, model_name="Facenet512", enforce_detection=False)[0]["embedding"]
            except Exception as e:
                print(f"❌ Could not get embedding for face #{i} in frame {idx}: {e}")
                continue

            # Match against each student
            for student in student_faces:
                try:
                    embedding2 = DeepFace.represent(
                        img_path=student["img"],
                        model_name="Facenet512",
                        # detector_backend="opencv" <- this is the default but it can be changed to get more accurate results at the cost of speed
                        # e.g. mtcnn, dlib, opencv, ssd, retinaface, mediapipe
                        enforce_detection=True
                    )[0]["embedding"]
                    distance = cosine(embedding1, embedding2)
                    if distance < threshold:
                        if student["id"] not in present_students:
                            print(f"✅ Match found: {student['name']} (ID: {student['id']}) in frame {idx}")
                        present_students.add(student["id"])
                        break  # Avoid duplicate matches for same face
                except Exception as e:
                    print(f"⚠️ Error with student {student['name']}: {e}")

        os.remove(img_path)

    present_list = [s for s in student_faces if s["id"] in present_students]
    absent_list = [s for s in student_faces if s["id"] not in present_students]

    return {
        "present": present_list,
        "absent": absent_list,
        "total_frames": len(classroom_images),
        "total_faces_detected": total_faces_detected,
        "matched_count": len(present_list)
    }

# recognize students in classroom (from videos)
@app.post("/attendance/recognize_from_video")
async def recognize_from_video(
    class_id: int = Form(...),
    video_file: UploadFile = File(...),
    frame_interval_sec: int = 60,  # Changeable interval
    db: Session = Depends(get_session)
):
    os.makedirs("temp", exist_ok=True)

    # Save uploaded video temporarily
    video_filename = f"temp/{uuid.uuid4()}.mp4"
    with open(video_filename, "wb") as f:
        f.write(await video_file.read())

    # Extract frames every N seconds
    video_clip = VideoFileClip(video_filename)
    duration = int(video_clip.duration)
    frame_paths = []

    for t in range(0, duration, frame_interval_sec):
        frame = video_clip.get_frame(t)
        frame_path = f"temp/frame_{t}.jpg"
        cv2.imwrite(frame_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        frame_paths.append(frame_path)
    video_clip.close()

    # Load student images for the class
    student_links = db.query(StudentClassLink).filter(StudentClassLink.class_id == class_id).all()
    student_ids = [link.student_id for link in student_links]
    if not student_ids:
        raise HTTPException(status_code=404, detail="No students linked to this class.")

    students = db.query(Students).filter(Students.id.in_(student_ids)).all()
    student_faces = []
    for s in students:
        image_path = os.path.join("storage/images", s.Image_URI) if not os.path.isabs(s.Image_URI) else s.Image_URI
        if os.path.exists(image_path):
            student_faces.append({"id": s.id, "name": s.name, "img": image_path})

    if not student_faces:
        raise HTTPException(status_code=404, detail="No valid student images found.")

    # Detect faces in frames
    present_students = set()
    threshold = 0.5
    total_faces_detected = 0

    for idx, frame_path in enumerate(frame_paths):
        try:
            detected_faces = DeepFace.extract_faces(
                frame_path, detector_backend="opencv", enforce_detection=False
            )
        except Exception as e:
            print(f"Error extracting faces from frame {idx}: {e}")
            continue

        print(f"🎞️ Frame {idx}: Detected {len(detected_faces)} face(s)")
        total_faces_detected += len(detected_faces)

        for i, face_data in enumerate(detected_faces):
            face_img = embed_face_into_canvas(face_data["face"])
            try:
                embedding1 = DeepFace.represent(face_img, model_name="Facenet512", enforce_detection=False)[0]["embedding"]
            except Exception as e:
                print(f"❌ Could not get embedding for face #{i} in frame {idx}: {e}")
                continue

            for student in student_faces:
                try:
                    embedding2 = DeepFace.represent(
                        img_path=student["img"],
                        model_name="Facenet512",
                        enforce_detection=True
                    )[0]["embedding"]
                    distance = cosine(embedding1, embedding2)
                    print(f"Distance to {student['name']}: {distance}")
                    if distance < threshold:
                        if student["id"] not in present_students:
                            print(f"✅ Match found: {student['name']} (ID: {student['id']}) in frame {idx}")
                        present_students.add(student["id"])
                        break
                except Exception as e:
                    print(f"⚠️ Error comparing to {student['name']}: {e}")

        os.remove(frame_path)

    # Cleanup video
    os.remove(video_filename)

    present_list = [s for s in student_faces if s["id"] in present_students]
    absent_list = [s for s in student_faces if s["id"] not in present_students]

    return {
        "present": present_list,
        "absent": absent_list,
        "total_frames": len(frame_paths),
        "total_faces_detected": total_faces_detected,
        "matched_count": len(present_list)
    }


class DeleteStudentsRequest(BaseModel):
    student_ids: List[int] = Field(..., example=[1, 2, 3], description="List of student IDs to delete")

@app.delete("/students/delete")
def delete_multiple_students(
    student_ids: DeleteStudentsRequest,
    db: Session = Depends(get_session)
):
    if not student_ids:
        raise HTTPException(status_code=400, detail="No student IDs provided")

    deleted_students = []
    failed_students = []

    for student_id in student_ids:
        student = db.query(Students).filter(Students.id == student_id).first()

        if not student:
            failed_students.append({"id": student_id, "reason": "Student not found"})
            continue

        # Delete from student_classes
        db.execute(
            StudentClassLink.delete().where(StudentClassLink.c.student_id == student_id)
        )

        # Delete image file
        if student.image_path and os.path.exists(student.image_path):
            try:
                os.remove(student.image_path)
            except Exception as e:
                failed_students.append({"id": student_id, "reason": f"Image deletion failed: {str(e)}"})
                continue

        db.delete(student)
        deleted_students.append(student_id)

    db.commit()

    return {
        "message": "Student deletion process completed",
        "deleted_students": deleted_students,
        "failed_students": failed_students
    }



@app.get("/records/all")
def get_all_records(db: Session = Depends(get_session)):
    records = db.query(Records).join(Attendances, Records.id == Attendances.record_id).all()

    return records


class AttendanceStudent(BaseModel):
    student_id: int
    isPresent: bool
    hours: float = 2  # Optional: if you want to include hours per entry

class CreateRecordPayload(BaseModel):
    class_id: int
    student_list: List[AttendanceStudent]
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

        # Create the record for the given class and user
        new_record = Records(class_id=payload.class_id, user_id=user.id)
        db.add(new_record)
        db.commit()
        db.refresh(new_record)

        # Add all attendance entries
        attendances = []
        for entry in payload.student_list:
            if entry.student_id is None:
                raise HTTPException(status_code=400, detail="Missing student_id in payload")

            attendances.append(Attendances(
                student_id=entry.student_id,
                isPresent=entry.isPresent,
                record_id=new_record.id
            ))

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


class StudentSummary(BaseModel):
    id: int
    name: str
    isPresent: bool
    hours: int

class RecordSummary(BaseModel):
    record_id: int
    date_created: datetime
    class_id: int
    user_id: int
    attendance_percentage: float
    total_students: int
    attended_students: int
    students: List[StudentSummary]
    class_: dict  # Optional: include a submodel for the class

@app.get("/records/summary", response_model=List[RecordSummary])


@app.get("/records/summary")
def get_records_summary(db: Session = Depends(get_session)):
    records = db.query(Records).options(
        joinedload(Records.class_),
        joinedload(Records.attendances).joinedload(Attendances.student)
    ).all()

    summary = []
    for record in records:
        total_students = len(record.attendances)
        attended_students = sum(1 for att in record.attendances if att.isPresent)
        attendance_percentage = (
            round((attended_students / total_students) * 100, 2)
            if total_students > 0 else 0
        )

        summary.append({
            "record_id": record.id,
            "date_created": record.date_created,
            "class": {
                "id": record.class_.id,
                "name": record.class_.name
            },
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
