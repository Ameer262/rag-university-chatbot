import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader, DirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

# הגדרות נתיבים
DATA_PATH = "data/"
VECTOR_STORE_PATH = "./vector_store"

# --- פונקציה חכמה שטוענת תיקייה ומדביקה תגיות (Metadata) ---
def load_and_tag_files(folder_path, dept_tag):
    print(f"📂 סורק את תיקיית: {folder_path} (תגית: {dept_tag})...")
    documents = []
    
    # 1. טעינת PDF
    pdf_loader = DirectoryLoader(folder_path, glob="**/*.pdf", loader_cls=PyPDFLoader, recursive=True)
    try:
        docs = pdf_loader.load()
        documents.extend(docs)
    except Exception: pass

    # 2. טעינת Word
    docx_loader = DirectoryLoader(folder_path, glob="**/*.docx", loader_cls=Docx2txtLoader, recursive=True)
    try:
        docs = docx_loader.load()
        documents.extend(docs)
    except Exception: pass

    # 3. טעינת TXT (עם תיקון עברית)
    txt_loader = DirectoryLoader(folder_path, glob="**/*.txt", loader_cls=TextLoader, recursive=True, loader_kwargs={'encoding': 'utf-8'})
    try:
        docs = txt_loader.load()
        documents.extend(docs)
    except Exception: pass

    # --- שלב התיוג (הדבקת המדבקה) ---
    # אנחנו עוברים על כל מסמך שמצאנו בתיקייה הזו ומוסיפים לו שדה 'department'
    for doc in documents:
        doc.metadata['department'] = dept_tag
        
    print(f"   ✅ נמצאו {len(documents)} מסמכים בתיקייה זו.")
    return documents

def main():
    print("--- מתחיל תהליך טעינה עם תיוג חכם ---")
    
    all_documents = []
    
    # 1. טעינת מדעי המחשב (CS)
    # שימו לב: אנחנו שולחים את הנתיב הספציפי ואת התגית 'cs'
    cs_docs = load_and_tag_files(os.path.join(DATA_PATH, "cs"), "cs")
    all_documents.extend(cs_docs)

    # 2. טעינת מערכות מידע (IS)
    is_docs = load_and_tag_files(os.path.join(DATA_PATH, "is"), "is")
    all_documents.extend(is_docs)

    if not all_documents:
        print("❌ שגיאה: לא נמצאו קבצים.")
        return

    print(f"\n📚 סה'כ מסמכים לעיבוד: {len(all_documents)}")

    # 3. חיתוך הנתונים
    print("✂️ חותך את המידע לחתיכות קטנות...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=350, chunk_overlap=75)
    splits = text_splitter.split_documents(all_documents)
    
    # הדפסת ביקורת - נראה דוגמה למדבקה
    if len(splits) > 0:
        print(f"   🔍 בדיקה: המטא-דאטה של החתיכה הראשונה: {splits[0].metadata}")

    # 4. יצירת Embeddings ושמירה
    print("🧠 טוען את מודל ה-Embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name="paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={'device': 'cpu'}
    )

    print(f"💾 שומר את המידע המתויג ב- {VECTOR_STORE_PATH}...")
    vectorstore = Chroma.from_documents(
        documents=splits, 
        embedding=embeddings,
        persist_directory=VECTOR_STORE_PATH
    )
    
    print("\n✨ תהליך ההכנה הסתיים! המאגר מוכן עם תגיות.")

if __name__ == "__main__":
    main()