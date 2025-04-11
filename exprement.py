import cv2
import matplotlib.pyplot as plt
from deepface import DeepFace
from scipy.spatial.distance import cosine

# === Load two face images (change paths to your images) ===
img1_path = "storage/images/bfc52597dec5.jpg"
img2_path = "temp/moe_2.jpg"

# === Display the images side by side ===
img1 = cv2.imread(img1_path)
img2 = cv2.imread(img2_path)

# Convert from BGR to RGB for display
img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

face_detected = DeepFace.extract_faces(img_path=img2_path)
face = face_detected[0]["face"]
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(img1_rgb)
plt.title("Image 1")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(face)
plt.title("Image 2")
plt.axis('off')
plt.show()

# === Get Embeddings from DeepFace ===
embedding1 = DeepFace.represent(img1_path, model_name="Facenet", enforce_detection=True)[0]["embedding"]
embedding2 = DeepFace.represent(face, model_name="Facenet", enforce_detection=True)[0]["embedding"]

# === Compute cosine distance ===
distance = cosine(embedding1, embedding2)

# === Print result ===
print(f"🧠 Cosine distance between the faces: {distance:.4f}")
if distance < 0.4:
    print("✅ Likely the same person")
else:
    print("❌ Likely different people")
