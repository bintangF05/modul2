import cv2
import numpy as np
import pickle
import os
from pathlib import Path
from insightface.app import FaceAnalysis

# ============================================================
# 1. KONFIGURASI (Disesuaikan untuk Acer Aspire Lite 15)
# ============================================================
KNOWN_FACES_DIR = Path("known_faces")
DB_PATH = "face_db.pkl"
THRESHOLD = 0.4       # Ambang batas kemiripan [cite: 991-992]
MODEL_NAME = "buffalo_s" # PAKAI VERSI 'S' AGAR RINGAN 

# ============================================================
# 2. INISIALISASI MODEL (PAKSA CPU)
# ============================================================
# InsightFace otomatis pakai RetinaFace (detector) + ArcFace (embedder) [cite: 1026]
app = FaceAnalysis(name=MODEL_NAME, providers=["CPUExecutionProvider"])
app.prepare(ctx_id=-1, det_size=(320, 320)) # ctx_id=-1 artinya pakai CPU [cite: 1029-1031]

# ============================================================
# 3. FUNGSI MEMBANGUN DATABASE (ENROLLMENT)
# ============================================================
def build_database():
    """Scan folder known_faces dan hitung mean embedding per orang [cite: 1046-1050]"""
    db = {}
    for person_dir in sorted(KNOWN_FACES_DIR.iterdir()):
        if not person_dir.is_dir(): continue
        
        person_name = person_dir.name
        embeddings = []
        
        for img_path in sorted(person_dir.glob("*.jpg")):
            img = cv2.imread(str(img_path))
            if img is None: continue
            
            faces = app.get(img) # Ekstrak fitur wajah [cite: 1085]
            if not faces: continue
            
            # Ambil wajah paling besar/utama [cite: 1089-1090]
            face = max(faces, key=lambda f: f.bbox[2] * f.bbox[3])
            embeddings.append(face.embedding)
            
        if embeddings:
            # Rata-ratakan embedding dan normalisasi L2 [cite: 1093-1094]
            mean_emb = np.stack(embeddings).mean(axis=0)
            db[person_name] = mean_emb / np.linalg.norm(mean_emb)
            print(f"Terdaftar: {person_name} ({len(embeddings)} foto)")
            
    with open(DB_PATH, "wb") as f:
        pickle.dump(db, f)
    return db

# ============================================================
# 4. FUNGSI PENGENALAN LIVE WEBCAM
# ============================================================
def live_recognition(db):
    """Jalankan face recognition pada frame webcam [cite: 1225-1231]"""
    cap = cv2.VideoCapture(0)
    print("Webcam dimulai. Tekan 'q' untuk keluar.")
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        # Tambahkan ini: flip secara horizontal (1) agar tidak mirror
        frame = cv2.flip(frame, 1)
        
        faces = app.get(frame)
        for face in faces:
            # Koordinat Bounding Box [cite: 1169]
            x1, y1, x2, y2 = [int(v) for v in face.bbox]
            
            # Hitung Similarity (Cosine Similarity) [cite: 1171-1176]
            query_emb = face.embedding / np.linalg.norm(face.embedding)
            best_name, best_sim = "Unknown", -1.0
            
            for name, ref_emb in db.items():
                sim = float(np.dot(query_emb, ref_emb))
                if sim > best_sim:
                    best_sim, best_name = sim, name
            
            # Cek terhadap Threshold [cite: 1181, 1384]
            if best_sim < THRESHOLD:
                label, color = f"Unknown ({best_sim:.2f})", (0, 0, 220)
            else:
                label, color = f"{best_name} ({best_sim:.2f})", (0, 200, 0)
            
            # Gambar Box dan Nama [cite: 1199-1206]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
        cv2.imshow("Face Recognition - Tekan Q untuk Keluar", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        
    cap.release()
    cv2.destroyAllWindows()

# ============================================================
# 5. MAIN EXECUTION
# ============================================================
if __name__ == "__main__":
    # Cek apakah database sudah ada [cite: 1323]
    if os.path.exists(DB_PATH):
        with open(DB_PATH, "rb") as f:
            database = pickle.load(f)
        print(f"Database dimuat: {list(database.keys())}")
    else:
        print("Membangun database baru...")
        database = build_database()
        
    live_recognition(database)