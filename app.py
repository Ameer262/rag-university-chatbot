import os
import streamlit as st
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_core.messages import HumanMessage, AIMessage

# הגדרות
VECTOR_STORE_PATH = "./vector_store"
os.environ["NVIDIA_API_KEY"] = "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" # ודאו שהמפתח כאן

@st.cache_resource
def get_components():
    print("טוען רכיבים...")
    llm = ChatNVIDIA(model="meta/llama3-8b-instruct")
    embeddings = HuggingFaceEmbeddings(
        model_name="paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={'device': 'cpu'}
    )
    if not os.path.exists(VECTOR_STORE_PATH):
        st.error(f"שגיאה: תיקיית מסד הנתונים '{VECTOR_STORE_PATH}' לא נמצאה.")
        st.stop()
        
    vectorstore = Chroma(
        persist_directory=VECTOR_STORE_PATH, 
        embedding_function=embeddings
    )
    return llm, vectorstore

# פונקציה שיוצרת את השרשרת *מחדש* בכל פעם שמשנים את הפילטר
def create_chain_with_filter(llm, vectorstore, department_filter):
    
    # 1. הגדרת הפילטר (החלק החדש והחכם!)
    # אם נבחר "הכל", לא מסננים. אחרת, מסננים לפי השדה 'department'
    if department_filter == "הכל":
        filter_dict = None
    else:
        # המרה מהשם בעברית לקוד באנגלית (כמו ששמרנו ב-ingest)
        dept_map = {"מדעי המחשב": "cs", "מערכות מידע": "is"}
        selected_dept = dept_map.get(department_filter)
        filter_dict = {"department": selected_dept}

    # 2. יצירת ה-Retriever עם הפילטר
    # שימו לב לפרמטר 'filter' החדש!
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 8, 
            "fetch_k": 20,
            "filter": filter_dict  # <--- כאן קורה הקסם
        }
    )

    # 3. בניית השרשרת (כמו קודם)
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [("system", contextualize_q_system_prompt), MessagesPlaceholder("chat_history"), ("human", "{input}")]
    )
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
    
    qa_system_prompt = (
        "אתה עוזר אוניברסיטאי. ענה על שאלת המשתמש אך ורק "
        "בהתבסס על ההקשר (Context) הבא:\n\n<context>\n{context}\n</context>"
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [("system", qa_system_prompt), MessagesPlaceholder("chat_history"), ("human", "{input}")]
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    return create_retrieval_chain(history_aware_retriever, question_answer_chain)

def main():
    st.set_page_config(page_title="צ'אטבוט הפקולטה", layout="wide")
    
    # --- סרגל צד (Sidebar) ---
    with st.sidebar:
        st.header("הגדרות")
        # תיבת בחירה לחוג
        selected_dept = st.selectbox(
            "בחר חוג:",
            ["הכל", "מדעי המחשב", "מערכות מידע"]
        )
        st.write("---")
        if st.button("נקה היסטוריית צ'אט"):
            st.session_state.chat_history = []
            st.rerun()

    st.title(f"🤖 צ'אטבוט - {selected_dept}")

    # טעינת רכיבים בסיסיים
    try:
        llm, vectorstore = get_components()
    except Exception as e:
        st.error(f"שגיאה: {e}")
        st.stop()

    # יצירת השרשרת הספציפית לפי הפילטר שנבחר
    rag_chain = create_chain_with_filter(llm, vectorstore, selected_dept)

    # ניהול היסטוריה והצגה (כמו קודם)
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for msg in st.session_state.chat_history:
        if isinstance(msg, HumanMessage):
            with st.chat_message("user"): st.markdown(msg.content)
        elif isinstance(msg, AIMessage):
            with st.chat_message("assistant"): st.markdown(msg.content)

    if prompt := st.chat_input("שאל אותי משהו..."):
        with st.chat_message("user"): st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("חושב..."):
                try:
                    response = rag_chain.invoke({
                        "input": prompt, 
                        "chat_history": st.session_state.chat_history
                    })
                    answer = response["answer"]
                    st.markdown(answer)
                    st.session_state.chat_history.extend([HumanMessage(content=prompt), AIMessage(content=answer)])
                except Exception as e:
                    st.error(f"אירעה שגיאה: {e}")

if __name__ == "__main__":
    main()