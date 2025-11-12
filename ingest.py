import os
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

# הגדרות
DATA_PATH = "data/"
VECTOR_STORE_PATH = "./vector_store"

def main():
    print("מתחיל תהליך הכנה (Ingestion)...")

    # 1. טעינת הנתונים (Load)
    #    טוען את כל קבצי ה-PDF מהתיקייה 'data'
    print(f"טוען קבצים מתיקיית {DATA_PATH}...")
    loader = DirectoryLoader(DATA_PATH, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    if not documents:
        print("לא נמצאו קבצי PDF בתיקייה 'data'.")
        return
    print(f"נטענו {len(documents)} מסמכים.")

    # 2. חיתוך הנתונים (Split/Chunk)
    print("חותך נתונים...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=350, chunk_overlap=75)
    splits = text_splitter.split_documents(documents)

    # 3. יצירת Embeddings ואחסון (Store)
    print("טוען מודל Embeddings מקומי (זה לוקח רגע בפעם הראשונה)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={'device': 'cpu'}
    )
    print("מודל Embeddings טעון!")

    print(f"יוצר ושומר מסד נתונים וקטורי ב- {VECTOR_STORE_PATH}...")
    # כאן אנחנו יוצרים את מסד הנתונים ושומרים אותו בדיסק
    vectorstore = Chroma.from_documents(
        documents=splits, 
        embedding=embeddings,
        persist_directory=VECTOR_STORE_PATH
    )

    print("---")
    print("תהליך ההכנה הסתיים! מסד הנתונים מוכן.")

# הפקודה שגורמת לסקריפט לרוץ כשקוראים לו
if __name__ == "__main__":
    main()