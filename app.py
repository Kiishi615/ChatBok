# ============================================
# SQLITE FIX - MUST BE FIRST
# ============================================
# Fix for Streamlit Cloud SQLite version issues
import sys

try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

# ============================================
# IMPORTS
# ============================================
import streamlit as st
from langchain_anthropic import ChatAnthropic
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import os
import tempfile
import logging
from datetime import datetime
from pathlib import Path

# ============================================
# LOGGING CONFIGURATION
# ============================================

def setup_logging():
    """Configure logging for the application."""
    
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    log_filename = log_dir / f"app_{datetime.now().strftime('%Y%m%d')}.log"
    
    logger = logging.getLogger("RAG_App")
    logger.setLevel(logging.DEBUG)
    
    if logger.handlers:
        logger.handlers.clear()
    
    detailed_formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    simple_formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )
    
    file_handler = logging.FileHandler(log_filename, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

logger = setup_logging()

# ============================================
# ENVIRONMENT SETUP
# ============================================

load_dotenv()

# Load from Streamlit secrets if available (for cloud deployment)
if hasattr(st, 'secrets'):
    if 'ANTHROPIC_API_KEY' in st.secrets:
        os.environ['ANTHROPIC_API_KEY'] = st.secrets['ANTHROPIC_API_KEY']
    if 'OPENAI_API_KEY' in st.secrets:
        os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']

logger.info("Environment variables loaded")

def validate_api_keys():
    """Check if required API keys are present."""
    missing_keys = []
    
    if not os.getenv("ANTHROPIC_API_KEY"):
        missing_keys.append("ANTHROPIC_API_KEY")
    if not os.getenv("OPENAI_API_KEY"):
        missing_keys.append("OPENAI_API_KEY")
    
    if missing_keys:
        logger.error(f"Missing API keys: {', '.join(missing_keys)}")
        return False, missing_keys
    
    logger.info("All required API keys are present")
    return True, []

# ============================================
# STREAMLIT PAGE CONFIGURATION
# ============================================

st.set_page_config(
    page_title="PDF Question Answering with RAG",
    page_icon="📚",
    layout="wide"
)

logger.info("Streamlit page configured")

st.title("📚 PDF Question Answering with RAG")
st.markdown("Upload a PDF document and ask questions about its content!")

# Check API keys
keys_valid, missing_keys = validate_api_keys()
if not keys_valid:
    st.error(f"❌ Missing API keys: {', '.join(missing_keys)}")
    st.info("Please set the required API keys in your environment or .env file")
    logger.critical("Application stopped due to missing API keys")
    st.stop()

# ============================================
# SESSION STATE INITIALIZATION
# ============================================

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
    logger.debug("Initialized vector_store in session state")

if "chunks" not in st.session_state:
    st.session_state.chunks = None
    logger.debug("Initialized chunks in session state")

if "embedding" not in st.session_state:
    st.session_state.embedding = None
    logger.debug("Initialized embedding in session state")
    
if "processed_file" not in st.session_state:
    st.session_state.processed_file = None
    logger.debug("Initialized processed_file in session state")
    
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    logger.debug("Initialized chat_history in session state")



# ============================================
# SIDEBAR CONFIGURATION
# ============================================

with st.sidebar:
    st.header("⚙️ Configuration")
    
    model_option = st.selectbox(
        "Select Claude Model",
        ["claude-3-haiku-20240307", "claude-sonnet-4-5-20250929"]
    )
    logger.debug(f"Model selected: {model_option}")
    
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.1)
    logger.debug(f"Temperature set to: {temperature}")
    
    st.subheader("📄 Chunk Settings")
    chunk_size = st.number_input(
        "Chunk Size", 
        min_value=100, 
        max_value=2000, 
        value=1000, 
        step=100
    )
    chunk_overlap = st.number_input(
        "Chunk Overlap", 
        min_value=0, 
        max_value=500, 
        value=200, 
        step=50
    )
    logger.debug(f"Chunk settings - Size: {chunk_size}, Overlap: {chunk_overlap}")

    
    st.divider()
    
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        logger.info("Chat history cleared by user")
        st.rerun()
    
    if st.button("🔄 Reset All"):
        st.session_state.vector_store = None
        st.session_state.chunks = None
        st.session_state.embedding = None
        st.session_state.processed_file = None
        st.session_state.chat_history = []
        st.session_state.processing_stats = {}
        logger.info("All session data reset by user")
        st.rerun()

# ============================================
# VECTOR STORE FUNCTIONS (USING FAISS)
# ============================================

def create_vector_store(chunks, embedding):
    """
    Create a FAISS vector store - more reliable than ChromaDB for Streamlit.
    """
    try:
        logger.info("Creating FAISS vector store...")
        vector_store = FAISS.from_documents(
            documents=chunks,
            embedding=embedding
        )
        logger.info("FAISS vector store created successfully")
        return vector_store
        
    except Exception as e:
        logger.error(f"Error creating vector store: {e}", exc_info=True)
        raise

# ============================================
# PDF PROCESSING FUNCTIONS
# ============================================

def process_pdf(uploaded_file, chunk_size, chunk_overlap):
    """
    Process uploaded PDF file and create vector store.
    """
    start_time = datetime.now()
    logger.info(f"Starting PDF processing: {uploaded_file.name}")
    
    tmp_file_path = None
    
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
            logger.debug(f"Temporary file created: {tmp_file_path}")
        
        # Load the PDF
        logger.info("Loading PDF document...")
        loader = PyPDFLoader(file_path=tmp_file_path)
        doc = loader.load()
        logger.info(f"PDF loaded successfully. Pages: {len(doc)}")
        
        if len(doc) == 0:
            raise ValueError("PDF appears to be empty or unreadable")
        
        # Initialize embeddings
        logger.info("Initializing OpenAI embeddings...")
        embedding = OpenAIEmbeddings(model='text-embedding-3-small')
        logger.debug("Embeddings initialized")
        
        # Split documents
        logger.info(f"Splitting documents (chunk_size={chunk_size}, overlap={chunk_overlap})...")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap
        )
        chunks = splitter.split_documents(doc)
        logger.info(f"Document split into {len(chunks)} chunks")
        
        if len(chunks) == 0:
            raise ValueError("No text chunks were created from the PDF")
        
        # Create vector store
        logger.info("Creating vector store...")
        vector_store = create_vector_store(chunks, embedding)
        logger.info("Vector store created successfully")
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Prepare stats
        stats = {
            "pages": len(doc),
            "chunks": len(chunks),
            "time": processing_time,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap
        }
        
        logger.info(f"PDF processing completed in {processing_time:.2f} seconds")
        
        return vector_store, stats, chunks, embedding
        
    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}", exc_info=True)
        return None, None, None, None
        
    finally:
        # Clean up temp file
        if tmp_file_path and os.path.exists(tmp_file_path):
            try:
                os.unlink(tmp_file_path)
                logger.debug("Cleaned up temporary file")
            except Exception as e:
                logger.warning(f"Could not delete temp file: {e}")

def get_rag_response(question, vector_store, model_option, temperature):
    """
    Generate RAG response for a given question.
    """
    start_time = datetime.now()
    logger.info(f"Processing question: {question[:100]}...")
    
    try:
        # Initialize LLM
        logger.debug(f"Initializing LLM: {model_option} with temperature={temperature}")
        llm = ChatAnthropic(model=model_option, temperature=temperature)
        
        # Create retriever
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        )
        logger.debug("Retriever created")
        
        # Create prompt template
        prompt = ChatPromptTemplate.from_template("""
            Answer the question based on the provided context. 
            If you don't know the answer or if the context doesn't contain 
            relevant information, say you don't know.
            
            Context: {context}
            
            Question: {question}
            
            Answer:
        """)
        
        # Create and execute chain
        chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        
        logger.debug("Invoking RAG chain...")
        response = chain.invoke(question)
        
        response_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Response generated in {response_time:.2f} seconds")
        
        return response
        
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}", exc_info=True)
        return f"Error: {str(e)}"

# ============================================
# FILE UPLOAD SECTION
# ============================================

st.header("📤 Upload Document")
uploaded_file = st.file_uploader(
    "Choose a PDF file", 
    type="pdf",
    help="Upload a PDF document to analyze"
)

if uploaded_file is not None:
    logger.debug(f"File uploaded: {uploaded_file.name}, Size: {uploaded_file.size} bytes")
    
    if uploaded_file.name != st.session_state.processed_file:
        logger.info(f"New file detected: {uploaded_file.name}")
        
        with st.spinner("🔄 Processing PDF... This may take a few moments."):
            vector_store, stats, chunks, embedding = process_pdf(
                uploaded_file, 
                chunk_size, 
                chunk_overlap
            )
            
            if vector_store is not None:
                st.session_state.vector_store = vector_store
                st.session_state.chunks = chunks
                st.session_state.embedding = embedding
                st.session_state.processed_file = uploaded_file.name
                st.session_state.processing_stats = stats
                st.session_state.chat_history = []
                
                st.success(f"✅ Successfully processed {uploaded_file.name}")

                    
                logger.info(f"File {uploaded_file.name} ready for questioning")
            else:
                st.error("❌ Failed to process PDF. Check the logs for details.")
                logger.error(f"Failed to process file: {uploaded_file.name}")
                st.stop()
    else:
        st.info(f"📁 Using previously processed file: {uploaded_file.name}")
        logger.debug(f"Using cached data for: {uploaded_file.name}")

# ============================================
# QUESTION ANSWERING SECTION
# ============================================

if st.session_state.vector_store is not None:
    st.divider()
    st.header("❓ Ask Questions")
    
    question = st.text_input(
        "Enter your question:",
        placeholder="What is this document about?",
        key="question_input"
    )
    
    col1, col2 = st.columns([1, 5])
    with col1:
        ask_button = st.button("🔍 Ask", type="primary")
    
    if ask_button and question:
        logger.info(f"User asked: {question}")
        
        with st.spinner(" Thinking..."):
            response = get_rag_response(
                question,
                st.session_state.vector_store,
                model_option,
                temperature
            )
            
            if not response.startswith("Error:"):
                st.session_state.chat_history.append({
                    "question": question,
                    "answer": response,
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "model": model_option
                })
                logger.info("Response added to chat history")
            else:
                st.error(response)
                logger.error(f"Error response: {response}")
    
    elif ask_button and not question:
        st.warning("⚠️ Please enter a question first!")
        logger.warning("Empty question submitted")
    
    # Display chat history
    if st.session_state.chat_history:
        st.divider()
        st.header("💬 Chat History")
        
        for i, chat in enumerate(reversed(st.session_state.chat_history)):
            idx = len(st.session_state.chat_history) - i
            with st.container():
                st.markdown(f"** Q{idx}:** {chat['question']}")
                st.markdown(f"** A:** {chat['answer']}")
                st.caption(f" {chat.get('timestamp', 'N/A')} |  {chat.get('model', 'N/A')}")
                st.divider()

else:
    st.info(" Please upload a PDF file to get started!")
    logger.debug("Waiting for PDF upload")

# ============================================
# FOOTER
# ============================================

st.markdown("---")
st.markdown("Built with ❤️ using LangChain & Streamlit")


logger.debug("App render completed")

