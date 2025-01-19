import streamlit as st
from PyPDF2 import PdfReader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os
from langdetect import detect
from pathlib import Path

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    st.error("Google API Key is missing. Please configure it in your .env file.")
    st.stop()

# Configuring Google Generative AI
from google.generativeai import configure
configure(api_key=api_key)

class DocumentProcessor:
    @staticmethod
    def get_pdf_text(pdf_docs):
        text = ""
        try:
            for pdf in pdf_docs:
                reader = PdfReader(pdf)
                for page in reader.pages:
                    text += page.extract_text()
        except Exception as e:
            st.error(f"Error reading PDF files: {e}")
        return text

    @staticmethod
    def get_text_chunks(text):
        splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
        return splitter.split_text(text)

    @staticmethod
    def get_vector_store(text_chunks):
        try:
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
            vector_store.save_local("faiss_index")
            return True
        except Exception as e:
            st.error(f"Error creating vector store: {e}")
            return False

class MultilingualChatTools:
    @staticmethod
    def detect_language(text):
        return detect(text)

    @staticmethod
    def transliterate_romanized_to_devanagari(text):
        transliteration_prompt = f"""
        Convert the following Romanized Nepali text into Devanagari Nepali:
        {text}
        
        Devanagari Text:
        """
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
        try:
            response = llm(transliteration_prompt)
            return response.get("output_text", "").strip()
        except Exception as e:
            #st.error(f"Error during transliteration: {e}")
            return text

class ChatbotUI:
    def __init__(self):
        self.doc_processor = DocumentProcessor()
        self.tools = MultilingualChatTools()
        self.setup_qa_chain()

    def setup_qa_chain(self):
        nepali_prompt_template = """
        उत्तरलाई प्रदान गरिएको सन्दर्भको आधारमा विस्तृत रूपमा जवाफ दिनुहोस्। 
        यदि जवाफ सन्दर्भमा उपलब्ध छैन भने, भन्नुहोस्: "सन्दर्भमा जवाफ उपलब्ध छैन।" अनुमान नगर्नुहोस्।
        सन्दर्भ:\n {context}\n
        प्रश्न:\n {question}\n
        उत्तर:
        """
        english_prompt_template = """
        Provide a detailed answer based on the given context. 
        If the answer is not available in the context, say: "The answer is not available in the context." Do not make assumptions.
        Context:\n {context}\n
        Question:\n {question}\n
        Answer:
        """
        self.nepali_prompt = PromptTemplate(template=nepali_prompt_template, input_variables=["context", "question"])
        self.english_prompt = PromptTemplate(template=english_prompt_template, input_variables=["context", "question"])
        self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)

    def user_query(self, user_question):
        try:
            # Detect input language
            lang = self.tools.detect_language(user_question)
            if lang == "ne":
                prompt = self.nepali_prompt
            elif lang == "en":
                prompt = self.english_prompt
            else:  # Assume Romanized Nepali if detected as English
                user_question = self.tools.transliterate_romanized_to_devanagari(user_question)
                prompt = self.nepali_prompt

            # Check if vector store exists
            if not Path("faiss_index").exists():
                return "Please upload and process the PDF documents first."

            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
            docs = db.similarity_search(user_question)

            chain = load_qa_chain(self.llm, chain_type="stuff", prompt=prompt)
            response = chain(
                {"input_documents": docs, "question": user_question}, 
                return_only_outputs=True
            )

            return response["output_text"]
        except Exception as e:
            return f"Error processing your question: {str(e)}"

def main():
    st.set_page_config(page_title="संवाद AI", layout="wide")
    st.header("संवाद AI: Intelligent Docs Interaction💬📚")

    # Initialize chatbot UI
    chatbot = ChatbotUI()

    # Sidebar for PDF processing
    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload PDF files", accept_multiple_files=True)
        if st.button("Process"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    raw_text = chatbot.doc_processor.get_pdf_text(pdf_docs)
                    if raw_text:
                        text_chunks = chatbot.doc_processor.get_text_chunks(raw_text)
                        if chatbot.doc_processor.get_vector_store(text_chunks):
                            st.success("Documents processed successfully!")
            else:
                st.error("Please upload PDF files.")

    # Main chat interface
    user_question = st.text_input("Ask a question:")
    if user_question:
        with st.spinner("Processing your question..."):
            response = chatbot.user_query(user_question)
            st.write(response)

if __name__ == "__main__":
    main()
