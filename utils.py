import os
import shutil

from fastapi import UploadFile
UPLOAD_DIR = os.environ.get("UPLOAD_DIR", "images")

def add_image_to_fs(file: UploadFile):
    if not file:
        return False
    else:
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            print(f"file saved to {file_path}")
    return file_path