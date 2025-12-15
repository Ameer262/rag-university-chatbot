import streamlit as st
import os
from dotenv import load_dotenv  # <--- הרכיב החדש

# טעינת המפתח מקובץ ה-.env הנסתר
load_dotenv()

# ייבוא המוח שבנינו בתיקיית src
from src import router, rag_engine

# --- הגדרות עמוד ---
st.set_page_config(page_title="Academic Chatbot Pro", layout="wide", page_icon="🎓")

# --- ניהול זיכרון (Session State) ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# זיכרון של "הקורס האחרון" כדי לשמור הקשר בין שאלות עוקבות
if "last_course_hint" not in st.session_state:
    st.session_state.last_course_hint = None

def reset_history():
    """פונקציה לניקוי ההיסטוריה בעת החלפת חוג"""
    st.session_state.chat_history = []
    st.session_state.last_course_hint = None

# --- סרגל צד (Sidebar) ---
with st.sidebar:
    st.title("⚙️ הגדרות מערכת")
    
    # בדיקה אוטומטית אם המפתח קיים
    if not os.getenv("NVIDIA_API_KEY"):
        st.error("❌ מפתח API חסר! נא ליצור קובץ .env")
        st.stop()
    else:
        st.success("✅ מחובר למערכת ה-AI")
    
    st.divider()
    
    st.header("1. בחר תחום לימוד")
    # מיפוי בין השם בעברית לקוד התיקייה
    dept_mapping = {
        "מדעי המחשב": "cs",
        "מערכות מידע": "is"
    }
    
    selected_dept_name = st.selectbox(
        "חוג:",
        list(dept_mapping.keys()),
        on_change=reset_history
    )
    
    # שמירת הקוד (cs/is) לשימוש המנוע
    current_dept_code = dept_mapping[selected_dept_name]
    
    st.info("💡 **טיפ:** המערכת מזהה לבד אם שאלתם על סילבוס או נהלים כלליים.")
    
    if st.button("🗑️ נקה שיחה ידנית"):
        reset_history()
        st.rerun()

# --- חלון הצ'אט הראשי ---
st.title("🤖 העוזר האקדמי החכם (v2.0)")

# 1. הצגת היסטוריית השיחה
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 2. קליטת שאלה מהמשתמש
if prompt := st.chat_input("שאל אותי משהו על הקורסים או הנהלים..."):
    
    # קבלת המפתח מהסביבה (לא מהמשתמש!)
    api_key = os.getenv("NVIDIA_API_KEY")

    # הצגת שאלת המשתמש מיידית
    st.chat_message("user").markdown(prompt)
    st.session_state.chat_history.append({"role": "user", "content": prompt})

    # --- עדכון זיכרון "קורס אחרון" (כרגע בסיסי) ---
    p_lower = prompt.lower()
    if "למידה עמוקה" in prompt or "deep learning" in p_lower:
        st.session_state.last_course_hint = "למידה עמוקה"

    # --- הוספת הקשר לשאלות קצרות (אם אין אישור לקורס בשאלה עצמה) ---
    short_q = len(prompt.strip()) <= 25
    has_course_word = ("למידה" in prompt) or ("deep" in p_lower)

    final_prompt = prompt
    if short_q and (not has_course_word) and st.session_state.last_course_hint:
        final_prompt = f"{prompt} (בהקשר של הקורס {st.session_state.last_course_hint})"

    # 3. תהליך המחשבה של הבוט
    with st.chat_message("assistant"):
        
        # שלב א': הראוטר (המוח הממיין)
        with st.status("🧠 מנתח את כוונת השאלה...", expanded=True) as status:
            
            # קריאה ל-Router (על השאלה אחרי הקשר)
            intent = router.classify_intent(final_prompt, api_key)
            
            st.write(f"סיווג זוהה: **{intent}**")
            st.write("מבצע אופטימיזציה לשאלה ושולף מסמכים...")
            
            status.update(label=f"✅ סווג כ: {intent}", state="complete", expanded=False)

        # שלב ב': המנוע (RAG Engine)
        response_text, sources = rag_engine.ask_question(
            original_query=final_prompt,
            department=current_dept_code,
            category=intent,
            api_key=api_key
        )
        
        # הצגת התשובה
        st.markdown(response_text)
        
        # הצגת מקורות
        if sources:
            with st.expander("📚 מקורות מידע ששימשו לתשובה"):
                for source in sources:
                    st.markdown(f"- 📄 `{source}`")

        # שמירה בהיסטוריה
        st.session_state.chat_history.append({"role": "assistant", "content": response_text})
