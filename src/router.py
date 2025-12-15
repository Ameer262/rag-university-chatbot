from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# פונקציה שמחליטה לאן לנתב את השאלה
def classify_intent(user_question, api_key):
    """
    מקבלת שאלה ומחליטה אם היא קשורה לסילבוס ספציפי או למידע כללי.
    מחזירה: 'syllabus' או 'general'
    """
    
    # עדכון שם המודל לגרסה החדשה (מתקן את שגיאת 404)
    llm = ChatNVIDIA(
        model="meta/llama-3.1-70b-instruct", 
        nvidia_api_key=api_key,
        temperature=0.0
    )

    # --- ההוראות המשופרות (המוח של הפקיד) ---
    system_instruction = """
    תפקידך לסווג שאלות של סטודנטים לשתי קטגוריות בלבד.
    
    קטגוריה 1: 'syllabus'
    השתמש בקטגוריה זו עבור:
    - כל שאלה הקשורה לתוכן לימודי ספציפי.
    - שאלות על שמות של קורסים (למשל "למידה עמוקה", "אלגוריתמים").
    - שאלות על מרצים, מתרגלים, שעות קבלה של מרצה.
    - שאלות על ציונים, מבחנים, תרגילי בית, חובות הקורס.
    
    קטגוריה 2: 'general'
    השתמש בקטגוריה זו אך ורק עבור:
    - נהלים מנהלתיים ברמת הפקולטה (סגירת תואר, הרשמה לקורסים).
    - פרטי קשר של המזכירות (טלפון, מייל מזכירות).
    - מיקום כיתות כללי או שעות פתיחה של הקמפוס.
    
    החזר אך ורק מילה אחת בתשובה: 'syllabus' או 'general'. אל תוסיף שום טקסט אחר.
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_instruction),
        ("user", "{question}")
    ])

    chain = prompt | llm | StrOutputParser()

    try:
        category = chain.invoke({"question": user_question})
        return category.strip().lower()
    except Exception as e:
        print(f"⚠️ שגיאה בסיווג: {e}")
        return "general"