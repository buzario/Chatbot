import streamlit as st
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langdetect import detect
import os
from datetime import datetime

# Function to retrieve the API key from Streamlit secrets
api_key = st.secrets.get("OPENAI_API_KEY")

if api_key:
    os.environ["OPENAI_API_KEY"] = api_key
else:
    st.error("Failed to retrieve the API key. Please ensure it is set in Streamlit secrets.")

# Define the LLM model
llm_model = "gpt-4o"

# Load PDF document
pdf_file_path = 'Biogas1_merged.pdf'  # Adjust the path accordingly
loader = PyMuPDFLoader(file_path=pdf_file_path)
data = loader.load()

# Split the text into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
data = text_splitter.split_documents(data)

# Create vector store with specific embedding model
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = FAISS.from_documents(data, embedding=embeddings)

# Create conversation chain
llm = ChatOpenAI(temperature=0.2, model_name=llm_model)
memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
conversation_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(),
    memory=memory
)

# Define the function to handle the conversation
def handle_conversation_with_details(query):
    # Detect the language of the query
    language = detect(query)
    
    if language != 'de':
        return "Bitte stellen Sie Ihre Fragen auf Deutsch."
    else:
        detailed_query = f"Bitte geben Sie innerhalb von 6 Zeilen eine ausführliche Erklärung der folgenden Frage. Wenn das LLM die Antwort in den bereitgestellten Dokumenten nicht findet, soll es mit 'Informationen zu einer bestimmten Frage können nicht gefunden werden' antworten: {query}"
        result = conversation_chain({"question": detailed_query})
        answer = result["answer"]
        return query, answer

# Initialize the conversation history and unanswered questions list
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

unanswered_questions_file = "unanswered_questions.txt"

# Load unanswered questions from the file
if "unanswered_questions" not in st.session_state:
    st.session_state.unanswered_questions = []
    if os.path.exists(unanswered_questions_file):
        with open(unanswered_questions_file, "r") as file:
            for line in file:
                parts = line.strip().split(" | ", 1)
                if len(parts) == 2:
                    timestamp, question = parts
                    st.session_state.unanswered_questions.append((timestamp, question))

# Function to save unanswered questions to a file
def save_unanswered_question(question):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.unanswered_questions.append((timestamp, question))
    with open(unanswered_questions_file, "a") as file:
        file.write(f"{timestamp} | {question}\n")

# Streamlit UI
st.title("BIOGASANLAGE CHATBOT")
st.write("Stellen Sie Ihre Fragen zur Biogasproduktion auf Deutsch und erhalten Sie detaillierte Antworten.")

# Chat input
query = st.chat_input("Ihre Frage:")

if query:
    response = handle_conversation_with_details(query)
    if isinstance(response, tuple):
        question, answer = response
        st.session_state.conversation_history.append((question, answer))
        if answer == "Informationen zu einer bestimmten Frage können nicht gefunden werden.":
            save_unanswered_question(question)
    else:
        st.session_state.conversation_history.append((query, response))

# Display the conversation history
for question, answer in st.session_state.conversation_history:
    with st.container():
        st.markdown(
            f"""
            <div style="background-color: #d9edf7; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
                <strong>Ihre Frage:</strong>
                <p>{question}</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown(
            f"""
            <div style="background-color: #dff0d8; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
                <strong>Antwort:</strong>
                <p>{answer}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

# Function to create a text file with unanswered questions
def create_unanswered_questions_file():
    with open("unanswered_questions.txt", "w") as file:
        for timestamp, question in st.session_state.unanswered_questions:
            file.write(f"{timestamp} | {question}\n")

# Add download button with authentication
def authenticate_user():
    username = st.text_input("Benutzername")
    password = st.text_input("Passwort", type="password")
    if username == st.secrets["USERNAME"] and password == st.secrets["PASSWORD"]:
        create_unanswered_questions_file()
        st.success("Authentifizierung erfolgreich!")
        with open("unanswered_questions.txt", "rb") as file:
            btn = st.download_button(
                label="Unbeantwortete Fragen herunterladen",
                data=file,
                file_name="unanswered_questions.txt",
                mime="text/plain"
            )
    else:
        st.error("Ungültiger Benutzername oder Passwort.")

# Display authentication and download button on the sidebar
with st.sidebar:
    st.header("Datei-Download")
    authenticate_user()
