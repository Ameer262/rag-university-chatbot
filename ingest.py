import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader, DirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

# הגדרות נתיבים
DATA_PATH = "data/"
VECTOR_STORE_PATH = "./vector_store"

def main():
    print(f"--- מתחיל תהליך טעינת כל המידע מ- {DATA_PATH} (כולל תתי-תיקיות) ---")
    
    documents = []
    
    # 1. טעינת קבצי PDF
    print("🔍 סורק קבצי PDF...")
    # recursive=True מאפשר לחפש בתוך תתי-תיקיות
    pdf_loader = DirectoryLoader(DATA_PATH, glob="**/*.pdf", loader_cls=PyPDFLoader, recursive=True)
    try:
        pdf_docs = pdf_loader.load()
        print(f"   ✅ נמצאו {len(pdf_docs)} דפים/מסמכים מסוג PDF.")
        documents.extend(pdf_docs)
    except Exception as e:
        print(f"   ⚠️ שגיאה בטעינת PDF (אולי אין קבצים כאלה?): {e}")

    # 2. טעינת קבצי Word (DOCX)
    print("🔍 סורק קבצי Word...")
    docx_loader = DirectoryLoader(DATA_PATH, glob="**/*.docx", loader_cls=Docx2txtLoader, recursive=True)
    try:
        docx_docs = docx_loader.load()
        if docx_docs:
            print(f"   ✅ נמצאו {len(docx_docs)} מסמכים מסוג Word.")
            documents.extend(docx_docs)
        else:
            print("   ℹ️ לא נמצאו קבצי Word.")
    except Exception as e:
        print(f"   ⚠️ שגיאה בטעינת Word: {e}")

    # 3. טעינת קבצי TXT (עם תיקון קריטי לעברית!)
    print("🔍 סורק קבצי טקסט (TXT)...")
    # loader_kwargs={'encoding': 'utf-8'} פותר את בעיית הג'יבריש בווינדוס
    txt_loader = DirectoryLoader(
        DATA_PATH, 
        glob="**/*.txt", 
        loader_cls=TextLoader, 
        recursive=True, 
        loader_kwargs={'encoding': 'utf-8'}
    )
    try:
        txt_docs = txt_loader.load()
        if txt_docs:
            print(f"   ✅ נמצאו {len(txt_docs)} מסמכים מסוג Text.")
            documents.extend(txt_docs)
        else:
            print("   ℹ️ לא נמצאו קבצי טקסט.")
    except Exception as e:
        print(f"   ⚠️ שגיאה בטעינת הטקסט: {e}")

    # בדיקה אם מצאנו משהו בכלל
    if not documents:
        print("❌ שגיאה קריטית: לא נמצאו שום קבצים בתיקיית data או בתתי-התיקיות שלה.")
        return

    print(f"\n📚 סה'כ מסמכים לעיבוד: {len(documents)}")

    # 4. חיתוך הנתונים
    print("✂️ חותך את המידע לחתיכות קטנות (Chunks)...")
    # נשארים עם ההגדרה הכירורגית שעבדה טוב
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=350, chunk_overlap=75)
    splits = text_splitter.split_documents(documents)
    print(f"   נוצרו {len(splits)} חתיכות מידע.")

    # 5. יצירת Embeddings ושמירה
    print("🧠 טוען את מודל ה-Embeddings (רב-לשוני)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={'device': 'cpu'}
    )

    print(f"💾 שומר את המידע למסד הנתונים ב- {VECTOR_STORE_PATH}...")
    vectorstore = Chroma.from_documents(
        documents=splits, 
        embedding=embeddings,
        persist_directory=VECTOR_STORE_PATH
    )
    
    print("\n✨ תהליך ההכנה הסתיים בהצלחה! הבוט מוכן.")

if __name__ == "__main__":
    main()