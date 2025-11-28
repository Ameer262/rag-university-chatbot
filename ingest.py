import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader, DirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

# הגדרות נתיבים
DATA_PATH = "data/"
VECTOR_STORE_PATH = "./vector_store"

# פונקציה חכמה לטעינה ותיוג
def load_and_tag_files(path_to_folder, dept_tag, category_tag):
    # נרמול הנתיב כדי שיעבוד טוב גם בווינדוס
    normalized_path = os.path.normpath(path_to_folder)
    
    print(f"\n📂 בודק את התיקייה: {normalized_path}")
    
    if not os.path.exists(normalized_path):
        print(f"   ❌ שגיאה: התיקייה לא קיימת! בדוק את השם שלה.")
        return []

    documents = []
    
    # טעינת PDF
    pdf_loader = DirectoryLoader(normalized_path, glob="**/*.pdf", loader_cls=PyPDFLoader)
    try:
        docs = pdf_loader.load()
        if docs: print(f"   found {len(docs)} PDFs")
        documents.extend(docs)
    except Exception as e: pass

    # טעינת Word
    docx_loader = DirectoryLoader(normalized_path, glob="**/*.docx", loader_cls=Docx2txtLoader)
    try:
        docs = docx_loader.load()
        if docs: print(f"   found {len(docs)} Word docs")
        documents.extend(docs)
    except Exception as e: pass

    # טעינת TXT (עם תיקון עברית)
    txt_loader = DirectoryLoader(normalized_path, glob="**/*.txt", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'})
    try:
        docs = txt_loader.load()
        if docs: print(f"   found {len(docs)} Text files")
        documents.extend(docs)
    except Exception as e: pass

    # תיוג (הדבקת המדבקות)
    if documents:
        for doc in documents:
            doc.metadata['department'] = dept_tag
            doc.metadata['category'] = category_tag
            
            # הדפסה לביקורת
            source = os.path.basename(doc.metadata.get('source', ''))
            print(f"   ✅ נטען ותויג: {source} -> [חוג: {dept_tag}, סוג: {category_tag}]")
    else:
        print("   ⚠️ התיקייה ריקה או שלא נמצאו קבצים נתמכים.")

    return documents

def main():
    print("--- מתחיל תהליך בניית המאגר ---")
    
    all_documents = []
    
    # 1. מדעי המחשב - סילבוסים
    all_documents.extend(load_and_tag_files("data/cs/syll", "cs", "syllabus"))
    
    # 2. מדעי המחשב - כללי
    all_documents.extend(load_and_tag_files("data/cs/general", "cs", "general"))

    # 3. מערכות מידע - סילבוסים
    all_documents.extend(load_and_tag_files("data/is/syll", "is", "syllabus"))

    # 4. מערכות מידע - כללי
    all_documents.extend(load_and_tag_files("data/is/general", "is", "general"))

    if not all_documents:
        print("\n❌ שגיאה קריטית: לא נטענו שום מסמכים! המאגר יהיה ריק.")
        return

    print(f"\n📚 סך הכל מסמכים לעיבוד: {len(all_documents)}")

    # חיתוך
    print("✂️ חותך נתונים (Chunk Size: 350)...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=350, chunk_overlap=75)
    splits = text_splitter.split_documents(all_documents)

    # יצירת מאגר
    print("🧠 טוען מודל Embeddings ושומר לדיסק...")
    
    # --- כאן היה התיקון ---
    embeddings = HuggingFaceEmbeddings(
        model_name="paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={'device': 'cpu'}
    )

    vectorstore = Chroma.from_documents(
        documents=splits, 
        embedding=embeddings,
        persist_directory=VECTOR_STORE_PATH
    )
    
    print("\n✨ סיימנו! המאגר מוכן לעבודה.")

if __name__ == "__main__":
    main()