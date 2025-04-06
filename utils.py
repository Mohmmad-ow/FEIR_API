import os
import shutil

from fastapi import UploadFile

UPLOAD_DIR = "storage/images"

def add_image_to_fs(file: UploadFile) -> str:
    if not file:
        return "storage/images/default.jpg"

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    print(f"File saved at: {file_path}")
    return file_path
