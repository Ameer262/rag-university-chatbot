from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from src import config

# חדש: מזהה קורס לפי השאלה מתוך course_catalog.json
from src.preprocessor import detect_course_from_query


# --- 1. טעינת המאגר (פעם אחת בלבד) ---
def get_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL_NAME,
        model_kwargs={'device': config.EMBEDDING_DEVICE}
    )
    vectorstore = Chroma(
        persist_directory=config.VECTOR_STORE_PATH,
        embedding_function=embeddings
    )
    return vectorstore


# --- 2. המוח המשפר (Preprocessing) ---
def optimize_query(original_query, department, api_key):
    """
    שלב ה-Preprocessing: משכתב את השאלה כדי שתהיה ברורה יותר למנוע החיפוש.
    """
    llm = ChatNVIDIA(model="meta/llama-3.1-70b-instruct", nvidia_api_key=api_key)

    system_prompt = f"""
    אתה עוזר לנסח מחדש שאלות לצורך חיפוש במסמכים (RAG) עבור חוג '{department}'.

    כללים חשובים:
    1) אסור לשנות את משמעות השאלה. אסור להוסיף נושאים חדשים.
    2) אסור להוסיף "בחינת ביניים" / "מטלה" / כל דבר שהמשתמש לא שאל עליו.
    3) אם השאלה קצרה, רק תרחיב אותה בצורה טבעית באותו נושא.
    4) אם השאלה מתייחסת לקורס, תשאיר את שם הקורס כפי שהמשתמש כתב (לדוגמה: "למידה עמוקה" / "Deep Learning").
    5) תשמור על שפה של חיפוש בסילבוס: Lecturer/Teaching Assistant/Tutorial/Lecture time/Office hours/Grading וכו'.

    דוגמאות:
    שאלה: "מתי ההרצאות?"
    שכתוב: "מהם זמני ההרצאות (Lectures) של הקורס שהוזכר בשאלה?"

    שאלה: "יש מתרגל?"
    שכתוב: "האם יש מתרגל/עוזר הוראה (Teaching Assistant) בקורס? אם כן מה שמו ומה פרטי הקשר שלו?"

    החזר רק את השאלה המשוכתבת. בלי הסברים.
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "{query}")
    ])

    chain = prompt | llm | StrOutputParser()

    try:
        new_query = chain.invoke({"query": original_query})
        print(f"🔄 Query Rewritten: '{original_query}' -> '{new_query}'")
        return new_query
    except:
        return original_query


# --- 3. הפונקציה הראשית: RAG ---
def ask_question(original_query, department, category, api_key):
    """
    הפונקציה שמבצעת את כל התהליך: זיהוי קורס -> שכתוב -> חיפוש -> תשובה
    """

    # 0) זיהוי קורס לפי השאלה (אם המשתמש כתב שם קורס/קוד/alias)
    matched_course = detect_course_from_query(original_query, department)
    if matched_course:
        print(f"🎯 Detected Course: {matched_course.get('course_name')} {matched_course.get('course_code')}")
    else:
        print("🎯 Detected Course: None")

    # א. שכתוב השאלה (Preprocessing)
    optimized_query = optimize_query(original_query, department, api_key)

    # ב. בניית פילטר
    and_filters = [
        {"department": department},
        {"category": category}
    ]

    # אם זיהינו קורס: נוסיף פילטר לפי course_code
    if matched_course and matched_course.get("course_code"):
        and_filters.append({"course_code": matched_course["course_code"]})

    filter_dict = {"$and": and_filters}
    print(f"🔍 Searching with filter: {filter_dict}")

    # ג. הכנת ה-Retriever
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 6,
            "fetch_k": 30,
            "lambda_mult": 0.5,
            "filter": filter_dict
        }
    )

    # ד. מודל התשובות
    llm = ChatNVIDIA(model="meta/llama-3.1-70b-instruct", nvidia_api_key=api_key)

    # ה. התבנית לתשובה הסופית
    answer_prompt = ChatPromptTemplate.from_template("""
    אתה עוזר אקדמי של הפקולטה.

    חוקים (חשוב):
    1) ענה בעברית בלבד.
    2) שמות פרטיים, שמות קורסים, כתובות אימייל, מספרי קורס, ושעות — השאר בדיוק כפי שהם מופיעים בהקשר (באנגלית אם צריך).
    3) ענה אך ורק על סמך ההקשר (Context) המצורף. אל תנחש ואל תוסיף מידע שלא מופיע בהקשר.
    4) אם התשובה לא מופיעה בהקשר, כתוב בדיוק: "איני יודע את התשובה על סמך המסמכים שסופקו".

    הקשר:
    {context}

    שאלה:
    {question}

    תשובה בעברית:
    """)

    # ו. בניית השרשרת (Chain)
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | answer_prompt
        | llm
        | StrOutputParser()
    )

    # ז. הרצה וקבלת תשובה + מקורות
    try:
        response = chain.invoke(optimized_query)

        # שליפת המקורות ידנית כדי להציג ב-UI
        source_docs = retriever.get_relevant_documents(optimized_query)

        print("\n--- RETRIEVED DOCS (debug) ---")
        for i, doc in enumerate(source_docs, 1):
            src = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", None)
            course_code = doc.metadata.get("course_code", None)
            preview = doc.page_content[:200].replace("\n", " ")
            print(f"{i}) source={src} page={page} course_code={course_code} preview={preview}...")
        print("--- END RETRIEVED DOCS ---\n")

        sources = [doc.metadata.get("source", "Unknown") for doc in source_docs]
        return response, sources

    except Exception as e:
        return f"שגיאה בתהליך: {str(e)}", []
