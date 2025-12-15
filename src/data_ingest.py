import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# ייבוא ההגדרות מקובץ הקונפיגורציה שלנו
from src import config

# קטלוג קורסים (data/course_catalog.json)
from src.preprocessor import _load_course_catalog


def match_course_by_source_file(source_file: str, dept_tag: str):
    """
    התאמת קורס לקובץ סילבוס לפי:
    1) source_files (מיפוי מפורש ב-JSON)
    2) אחרת: aliases / course_name / course_code בתוך שם הקובץ
    מחזיר dict של הקורס או None.
    """
    catalog = _load_course_catalog(os.path.join("data", "course_catalog.json"))
    if not catalog or not source_file:
        return None

    sf_lower = source_file.lower().strip()

    for course in catalog:
        if course.get("department") != dept_tag:
            continue

        # 1) מיפוי מפורש לפי שם קובץ
        source_files = course.get("source_files", []) or []
        for f in source_files:
            if str(f).lower().strip() == sf_lower:
                return course

        # 2) fallback: חיפוש לפי alias בתוך שם הקובץ
        aliases = list(course.get("aliases", []) or [])
        if course.get("course_name"):
            aliases.append(course["course_name"])
        if course.get("course_code"):
            aliases.append(course["course_code"])

        for a in aliases:
            a = str(a).lower().strip()
            if a and a in sf_lower:
                return course

    return None


def load_and_tag_files(path_to_folder, dept_tag, category_tag):
    """
    פונקציה שטוענת קבצים מתיקייה, מדביקה תגיות מטא-דאטה,
    ובמקרה של syllabus גם מוסיפה course_code/course_name לפי course_catalog.json.
    """
    normalized_path = os.path.normpath(path_to_folder)
    print(f"\n📂 בודק את התיקייה: {normalized_path}")

    if not os.path.exists(normalized_path):
        print(f"   ❌ שגיאה: התיקייה לא קיימת!")
        return []

    documents = []

    loaders = {
        "**/*.pdf": PyPDFLoader,
        "**/*.docx": Docx2txtLoader,
        "**/*.txt": TextLoader
    }

    for glob_pattern, loader_cls in loaders.items():
        loader_kwargs = {"encoding": "utf-8"} if loader_cls == TextLoader else {}

        try:
            loader = DirectoryLoader(
                normalized_path,
                glob=glob_pattern,
                loader_cls=loader_cls,
                loader_kwargs=loader_kwargs
            )
            docs = loader.load()

            # דיבאג: דוגמא של source + page
            for d in docs[:5]:
                print("   sample:", d.metadata.get("source"), "page:", d.metadata.get("page"))

            if docs:
                # זה מספר Documents (עמודים/טקסטים), לא בהכרח מספר קבצים
                print(f"   found {len(docs)} docs of type {glob_pattern}")
                documents.extend(docs)

        except Exception as e:
            print(f"   ⚠️ load failed for {glob_pattern} in {normalized_path}: {e}")

    # תיוג + הוספת מטאדאטה לקורס (לסילבוסים בלבד)
    if documents:
        for doc in documents:
            doc.metadata["department"] = dept_tag
            doc.metadata["category"] = category_tag

            source_path = doc.metadata.get("source", "")
            source_file = os.path.basename(source_path)

            # ברירת מחדל
            doc.metadata["course_code"] = None
            doc.metadata["course_name"] = None

            # רק לסילבוס: נסה לזהות קורס לפי שם קובץ/מיפוי
            if category_tag == "syllabus":
                matched = match_course_by_source_file(source_file, dept_tag)
                if matched:
                    doc.metadata["course_code"] = matched.get("course_code")
                    doc.metadata["course_name"] = matched.get("course_name")

            print(
                f"   ✅ נטען: {source_file} -> "
                f"[חוג: {dept_tag}, סוג: {category_tag}, "
                f"קורס: {doc.metadata['course_name']} {doc.metadata['course_code']}]"
            )
    else:
        print("   ⚠️ לא נמצאו קבצים בתיקייה זו.")

    return documents


def build_vector_store():
    """
    הפונקציה הראשית שבונה את המאגר
    """
    print("--- מתחיל תהליך בניית המאגר (Refactored) ---")

    all_documents = []
    base_data = config.DATA_PATH

    # 1. מדעי המחשב
    all_documents.extend(load_and_tag_files(os.path.join(base_data, "cs", "syll"), "cs", "syllabus"))
    all_documents.extend(load_and_tag_files(os.path.join(base_data, "cs", "general"), "cs", "general"))

    # 2. מערכות מידע
    all_documents.extend(load_and_tag_files(os.path.join(base_data, "is", "syll"), "is", "syllabus"))
    all_documents.extend(load_and_tag_files(os.path.join(base_data, "is", "general"), "is", "general"))

    if not all_documents:
        print("\n❌ לא נטענו מסמכים via src/data_ingest.py")
        return

    print(f"\n✂️ חותך נתונים (Size: {config.CHUNK_SIZE})...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP
    )
    splits = text_splitter.split_documents(all_documents)

    print(f"🧠 טוען מודל Embeddings ({config.EMBEDDING_MODEL_NAME})...")
    embeddings = HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL_NAME,
        model_kwargs={"device": config.EMBEDDING_DEVICE}
    )

    print(f"💾 שומר לתיקייה: {config.VECTOR_STORE_PATH}")
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=config.VECTOR_STORE_PATH
    )

    # לשמור בוודאות
    try:
        vectorstore.persist()
    except Exception:
        pass

    print("\n✨ סיימנו! המאגר מוכן.")
