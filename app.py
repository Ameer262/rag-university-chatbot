import os
import streamlit as st
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder # <-- ייבוא חדש
from langchain.chains.combine_documents import create_stuff_documents_chain
# --- ייבואים חדשים לניהול היסטוריה ---
from langchain.chains import create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_core.messages import HumanMessage, AIMessage

# הגדרות
VECTOR_STORE_PATH = "./vector_store"
os.environ["NVIDIA_API_KEY"] = "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" # ודאו שהמפתח כאן

# -----------------------------------------------------------------
# פונקציה לטעינת הרכיבים (LLM, Embeddings, VectorStore)
# -----------------------------------------------------------------
@st.cache_resource
def get_components():
    print("טוען רכיבים... (זה קורה רק פעם אחת)")
    
    # 1. טעינת ה-LLM (המוח החושב)
    llm = ChatNVIDIA(model="meta/llama3-8b-instruct")
    
    # 2. טעינת מודל ה-Embeddings (הספרן)
    embeddings = HuggingFaceEmbeddings(
        model_name="paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={'device': 'cpu'}
    )
    
    # 3. טעינת מסד הנתונים
    if not os.path.exists(VECTOR_STORE_PATH):
        st.error(f"שגיאה: תיקיית מסד הנתונים '{VECTOR_STORE_PATH}' לא נמצאה.")
        st.stop()
        
    vectorstore = Chroma(
        persist_directory=VECTOR_STORE_PATH, 
        embedding_function=embeddings
    )
    
    # 4. הגדרת ה-Retriever (עם שדה ראייה רחב)
    retriever = vectorstore.as_retriever(
    search_type="mmr", 
    search_kwargs={"k": 8, "fetch_k": 20}
    )
    
    return llm, retriever

# -----------------------------------------------------------------
# פונקציה ליצירת שרשרת RAG (הפעם עם זיכרון)
# -----------------------------------------------------------------
def create_rag_chain(llm, retriever):
    
    # --- פרומפט 1: שכתוב השאלה ---
    # פרומפט שמנחה את ה-LLM לקחת את ההיסטוריה ואת השאלה החדשה, 
    # וליצור שאלה עצמאית
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    )
    
    # --- שרשרת 1: ה-Retriever שיודע "לזכור" ---
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )
    
    # --- פרומפט 2: עניית התשובה ---
    # זה הפרומפט המקורי שלנו, רק עם תוספת זיכרון
    qa_system_prompt = (
        "אתה עוזר אוניברסיטאי. ענה על שאלת המשתמש אך ורק "
        "בהתבסס על ההקשר (Context) הבא:\n\n"
        "<context>\n{context}\n</context>"
    )
    
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    )

    # --- שרשרת 2: יצירת תשובה מהמסמכים ---
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    
    # --- השרשרת המלאה ---
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    return rag_chain

# --- פונקציית main (מבוססת Streamlit) ---
def main():
    st.set_page_config(page_title="צ'אטבוט הפקולטה", layout="wide")
    st.title("🤖 צ'אטבוט הפקולטה (עם זיכרון)")

    # --- טעינת הרכיבים ---
    try:
        llm, retriever = get_components()
    except Exception as e:
        if "Authorization failed" in str(e):
            st.error("שגיאת התחברות ל-NVIDIA. אנא ודא שה-NVIDIA_API_KEY שלך נכון.")
            st.stop()
        else:
            st.error(f"אירעה שגיאה בטעינת המודל: {e}")
            st.stop()

    # --- יצירת השרשרת ---
    rag_chain = create_rag_chain(llm, retriever)

    # --- ניהול זיכרון (היסטוריית צ'אט) ---
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [] # אתחול היסטוריית צ'אט ריקה

    # הצגת הודעות ישנות מההיסטוריה
    for msg in st.session_state.chat_history:
        # המרת אובייקטים של LangChain לטקסט פשוט עבור ההצגה
        if isinstance(msg, HumanMessage):
            with st.chat_message("user"):
                st.markdown(msg.content)
        elif isinstance(msg, AIMessage):
            with st.chat_message("assistant"):
                st.markdown(msg.content)

    # --- קבלת קלט מהמשתמש ---
    if prompt := st.chat_input("שאל אותי משהו על הקורסים..."):
        # הצגת הודעת המשתמש
        with st.chat_message("user"):
            st.markdown(prompt)

        # --- קבלת תשובה מה-RAG chain (עם היסטוריה) ---
        with st.chat_message("assistant"):
            with st.spinner("חושב..."):
                # הפעם אנחנו שולחים גם את ההיסטוריה לשרשרת
                response = rag_chain.invoke({
                    "input": prompt,
                    "chat_history": st.session_state.chat_history
                })
                answer = response["answer"]
                st.markdown(answer)
        
        # עדכון ההיסטוריה (עם האובייקטים המיוחדים של LangChain)
        st.session_state.chat_history.append(HumanMessage(content=prompt))
        st.session_state.chat_history.append(AIMessage(content=answer))

# הפעלה
if __name__ == "__main__":
    main()