
import os
import cv2
import torch
import pickle
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.metrics.pairwise import cosine_similarity

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(image_size=160, margin=0, device=device)
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Paths
base_path = r"D:\semesters\Bsce semester 6\Machine learning\testing"
register_dir = os.path.join(base_path, "Raw")
embedding_dir = os.path.join(base_path, "embeddings")
os.makedirs(embedding_dir, exist_ok=True)

# Extract face embedding
def extract_embedding_from_frame(frame):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face = mtcnn(img_rgb)
    if face is None:
        return None
    with torch.no_grad():
        embedding = model(face.unsqueeze(0).to(device))
    return embedding.cpu().numpy()[0]

# Register and save embeddings
def register_faces():
    for person in os.listdir(register_dir):
        person_folder = os.path.join(register_dir, person)
        if not os.path.isdir(person_folder):
            continue
        embeddings = []
        for file in os.listdir(person_folder):
            img_path = os.path.join(person_folder, file)
            img = cv2.imread(img_path)
            if img is None:
                continue
            emb = extract_embedding_from_frame(img)
            if emb is not None:
                embeddings.append(emb)
        if embeddings:
            mean_embedding = np.mean(embeddings, axis=0)
            with open(os.path.join(embedding_dir, f"{person}.pkl"), 'wb') as f:
                pickle.dump(mean_embedding, f)
            print(f"[+] Registered {person} ({len(embeddings)} images)")
        else:
            print(f"[!] No valid faces found for {person}")

# Load registered embeddings
def load_registered_embeddings():
    registered = {}
    for file in os.listdir(embedding_dir):
        if file.endswith('.pkl'):
            name = os.path.splitext(file)[0]
            with open(os.path.join(embedding_dir, file), 'rb') as f:
                registered[name] = pickle.load(f)
    return registered

# Real-time recognition
def recognize_faces_live(threshold=0.6, capture_count=5):
    registered = load_registered_embeddings()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[!] Cannot access webcam")
        return

    print("[*] Starting real-time face recognition. Press 'q' to quit.")

    unknown_buffer = []

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        emb = extract_embedding_from_frame(frame)
        label = "No Face Detected"

        if emb is not None:
            best_match = "Unknown"
            best_score = -1
            for name, reg_embed in registered.items():
                score = cosine_similarity([emb], [reg_embed])[0][0]
                if score > best_score:
                    best_score = score
                    best_match = name

            if best_score >= threshold:
                label = f"{best_match} ({best_score:.2f})"
                unknown_buffer.clear()  # Reset buffer if a known person is detected
            else:
                label = "Unknown"
                unknown_buffer.append(frame.copy())

                if len(unknown_buffer) >= capture_count:
                    print("[?] Unknown face detected multiple times.")
                    response = input(">>> Do you want to register this person? (y/n): ").strip().lower()
                    if response == 'y':
                        name = input(">>> Enter name for new person: ").strip()
                        person_dir = os.path.join(register_dir, name)
                        os.makedirs(person_dir, exist_ok=True)

                        for i, img in enumerate(unknown_buffer):
                            path = os.path.join(person_dir, f"{name}_{i}.jpg")
                            cv2.imwrite(path, img)

                        print("[*] Saved images. Registering new person...")
                        register_faces()
                        registered = load_registered_embeddings()  # Refresh
                        print(f"[+] {name} has been registered.")
                    else:
                        print("[*] Face not registered.")
                    unknown_buffer.clear()

        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0) if label != "Unknown" else (0, 0, 255), 2)

        cv2.imshow("Real-Time Face Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# Run pipeline
if __name__ == "__main__":
    register_faces()
    recognize_faces_live()
