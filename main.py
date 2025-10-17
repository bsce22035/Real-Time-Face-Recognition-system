import os
import cv2
import torch
import pickle
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.metrics.pairwise import cosine_similarity
from torchvision import transforms

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(image_size=160, margin=0, device=device)
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Paths
base_path = r"E:\OneDrive - Higher Education Commission\University Work\Semester 6\Machine Learning\Theory\Project\testing"
register_dir = os.path.join(base_path, "Raw")
embedding_dir = os.path.join(base_path, "embeddings")
os.makedirs(embedding_dir, exist_ok=True)

# Define preprocessing manually
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

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

        embedding_path = os.path.join(embedding_dir, f"{person}.pkl")
        # Skip if embedding already exists
        if os.path.exists(embedding_path):
            print(f"[=] Embedding already exists for {person}, skipping.")
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
            with open(embedding_path, 'wb') as f:
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
def recognize_faces_live_Single(threshold=0.6):
    registered = load_registered_embeddings()
    cap = cv2.VideoCapture(0)  # 0 is default webcam

    if not cap.isOpened():
        print("[!] Cannot access webcam")
        return

    print("[*] Starting real-time face recognition. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        emb = extract_embedding_from_frame(frame)
        if emb is not None:
            best_match = "Unknown"
            best_score = -1
            for name, reg_embed in registered.items():
                score = cosine_similarity([emb], [reg_embed])[0][0]
                if score > best_score:
                    best_score = score
                    best_match = name
            label = f"{best_match} ({best_score:.2f})" if best_score >= threshold else "Unknown"
            cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No Face Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Real-Time Face Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    
def recognize_faces_live_Multi(threshold=0.6):
    registered = load_registered_embeddings()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[!] Cannot access webcam")
        return

    print("[*] Starting real-time face recognition. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, _ = mtcnn.detect(img_rgb)

        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                face = frame[y1:y2, x1:x2]

                if face.size == 0:
                    continue

                try:
                    face_tensor = preprocess(face).unsqueeze(0).to(device)
                except Exception as e:
                    continue  # Skip invalid crops

                with torch.no_grad():
                    embedding = model(face_tensor).cpu().numpy()[0]

                best_match = "Unknown"
                best_score = -1
                for name, reg_embed in registered.items():
                    score = cosine_similarity([embedding], [reg_embed])[0][0]
                    if score > best_score:
                        best_score = score
                        best_match = name

                label = f"{best_match} ({best_score:.2f})" if best_score >= threshold else "Unknown"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        else:
            cv2.putText(frame, "No Face Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Real-Time Face Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# Run pipeline
if __name__ == "__main__":
    register_faces()
    # recognize_faces_live_Single()
    recognize_faces_live_Multi()
