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
os.environ["NVIDIA_API_KEY"] = "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" # <--- ודאו שהמפתח שלכם כאן!

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

# --- הפונקציה המתוקנת לסינון ---
def create_chain_with_filter(llm, vectorstore, department_filter, category_filter):
    
    conditions = []

    # 1. תנאי חוג
    if department_filter != "הכל":
        dept_map = {"מדעי המחשב": "cs", "מערכות מידע": "is"}
        selected_dept = dept_map.get(department_filter)
        if selected_dept:
            conditions.append({"department": selected_dept})

    # 2. תנאי סוג מידע
    if category_filter != "הכל":
        cat_map = {"סילבוס קורס": "syllabus", "מידע כללי ונהלים": "general"}
        selected_cat = cat_map.get(category_filter)
        if selected_cat:
            conditions.append({"category": selected_cat})

    # --- בניית הפילטר הסופי (התיקון הקריטי) ---
    if len(conditions) == 0:
        final_filter = None
    elif len(conditions) == 1:
        # אם יש רק תנאי אחד, מעבירים אותו ישירות
        final_filter = conditions[0]
    else:
        # אם יש יותר מתנאי אחד, חייבים להשתמש ב-$and
        final_filter = {"$and": conditions}

    print(f"DEBUG: Active Filter: {final_filter}") 

    # יצירת ה-Retriever
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 8, 
            "fetch_k": 20,
            "filter": final_filter 
        }
    )

    # --- בניית השרשרת (היסטוריה + תשובה) ---
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
    
    # --- סרגל צד ---
    with st.sidebar:
        st.header("סינון וחיפוש")
        
        selected_dept = st.selectbox(
            "1. בחר חוג:",
            ["הכל", "מדעי המחשב", "מערכות מידע"]
        )

        selected_category = st.selectbox(
            "2. איזה מידע מעניין אותך?",
            ["הכל", "סילבוס קורס", "מידע כללי ונהלים"]
        )

        st.info(f"מצב נוכחי: {selected_dept} -> {selected_category}")
        
        st.write("---")
        if st.button("נקה היסטוריית צ'אט"):
            st.session_state.chat_history = []
            st.rerun()

    st.title(f"🤖 צ'אטבוט הפקולטה")

    try:
        llm, vectorstore = get_components()
    except Exception as e:
        st.error(f"שגיאה: {e}")
        st.stop()

    # יצירת השרשרת עם הפילטר החדש
    rag_chain = create_chain_with_filter(llm, vectorstore, selected_dept, selected_category)

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
            with st.spinner("מחפש..."):
                try:
                    response = rag_chain.invoke({
                        "input": prompt, 
                        "chat_history": st.session_state.chat_history
                    })
                    answer = response["answer"]
                    sources = response["context"]

                    st.markdown(answer)

                    # הצגת מקורות
                    with st.expander("📚 הצג מקורות מידע"):
                        seen_sources = set()
                        for doc in sources:
                            source_name = os.path.basename(doc.metadata.get("source", "לא ידוע"))
                            dept = doc.metadata.get("department", "?")
                            cat = doc.metadata.get("category", "?")
                            
                            # מזהה ייחודי למקור (כדי לא להציג כפילויות)
                            source_id = f"{source_name} ({dept}/{cat})"
                            
                            if source_id not in seen_sources:
                                st.markdown(f"- 📄 **{source_id}**")
                                seen_sources.add(source_id)

                    st.session_state.chat_history.extend([HumanMessage(content=prompt), AIMessage(content=answer)])
                
                except Exception as e:
                    st.error(f"אירעה שגיאה: {e}")

if __name__ == "__main__":
    main()