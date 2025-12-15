import os

# --- נתיבים ---
# תיקיית הנתונים הראשית
DATA_PATH = os.path.join("data")

# המיקום שבו יישמר מסד הנתונים הווקטורי
VECTOR_STORE_PATH = os.path.join("vector_store")

# --- מודלים ---
# מודל ה-Embeddings (זה שהופך טקסט למספרים)
EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_DEVICE = "cpu"  # אם יש לכם כרטיס גרפי של NVIDIA, אפשר לשנות ל-'cuda'

# --- הגדרות חיתוך טקסט ---
CHUNK_SIZE = 350
CHUNK_OVERLAP = 75