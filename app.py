import os
import streamlit as st  # <-- ייבוא חדש!
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# הגדרות
VECTOR_STORE_PATH = "./vector_store"
os.environ["NVIDIA_API_KEY"] = "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" # ודאו שהמפתח כאן

# -----------------------------------------------------------------
# פונקציה לטעינת השרשרת - עם "זיכרון מטמון"
# זה החלק הכי חשוב!
# Streamlit מריץ את כל הסקריפט מחדש בכל אינטראקציה.
# @st.cache_resource אומר ל-Streamlit "תריץ את הפונקציה הזו רק פעם אחת,
# ותשמור את התוצאה שלה בזיכרון".
# זה מונע מהמודלים הכבדים להיטען מחדש כל פעם.
# -----------------------------------------------------------------
@st.cache_resource
def load_rag_chain():
    print("טוען את מודל ה-RAG... (זה קורה רק פעם אחת)")
    
    # 1. טעינת ה-LLM (המוח החושב של NVIDIA)
    llm = ChatNVIDIA(model="meta/llama3-8b-instruct")
    
    # 2. טעינת מודל ה-Embeddings (הספרן המקומי)
    embeddings = HuggingFaceEmbeddings(
        model_name="paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={'device': 'cpu'}
    )
    
    # 3. טעינת מסד הנתונים הקיים מהדיסק
    if not os.path.exists(VECTOR_STORE_PATH):
        # זו שגיאה קריטית, אז נעצור את האפליקציה אם אין DB
        st.error(f"שגיאה: תיקיית מסד הנתונים '{VECTOR_STORE_PATH}' לא נמצאה.")
        st.stop()
        
    vectorstore = Chroma(
        persist_directory=VECTOR_STORE_PATH, 
        embedding_function=embeddings
    )

    # 4. הגדרת ה-RAG (ה"ספרן" והפרומפט)
    retriever = vectorstore.as_retriever()

    prompt_template = """
אתה עוזר אוניברסיטאי. ענה על שאלת המשתמש אך ורק 
בהתבסס על ההקשר (Context) הבא:
<context>
{context}
</context>
שאלה: {input}
"""
    prompt = ChatPromptTemplate.from_template(prompt_template)

    # 5. בניית השרשרת (Chain)
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)
    
    return retrieval_chain

# --- פונקציית main חדשה (מבוססת Streamlit) ---
def main():
    # --- הגדרות עמוד ---
    st.set_page_config(page_title="צ'אטבוט הפקולטה", layout="wide")
    st.title("🤖 צ'אטבוט הפקולטה (מבוסס RAG)")

    # --- טעינת השרשרת ---
    # נטען את השרשרת פעם אחת בזכות ה-cache
    try:
        retrieval_chain = load_rag_chain()
    except Exception as e:
        # אם ה-API Key לא נכון, נראה שגיאה יפה
        if "Authorization failed" in str(e):
            st.error("שגיאת התחברות ל-NVIDIA. אנא ודא שה-NVIDIA_API_KEY שלך נכון.")
            st.stop()
        else:
            st.error(f"אירעה שגיאה בטעינת המודל: {e}")
            st.stop()

    # --- ניהול זיכרון (היסטוריית צ'אט) ---
    # Streamlit לא זוכר משתנים בין ריצות. 
    # 'st.session_state' הוא "זיכרון" שנשמר
    if "messages" not in st.session_state:
        st.session_state.messages = [] # אתחול היסטוריית צ'אט ריקה

    # הצגת הודעות ישנות מההיסטוריה
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # --- קבלת קלט מהמשתמש ---
    # 'st.chat_input' יוצר תיבת טקסט בתחתית המסך
    if prompt := st.chat_input("שאל אותי משהו על הקורסים..."):
        # 1. הוסף את הודעת המשתמש להיסטוריה ולהצגה
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 2. קבל תשובה מה-RAG chain
        with st.chat_message("assistant"):
            # נוסיף "ספינר" בזמן שהבוט חושב
            with st.spinner("חושב..."):
                response = retrieval_chain.invoke({"input": prompt})
                answer = response["answer"]
                st.markdown(answer)
        
        # 3. הוסף את תשובת הבוט להיסטוריה
        st.session_state.messages.append({"role": "assistant", "content": answer})

# הפעלה
if __name__ == "__main__":
    main()