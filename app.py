import streamlit as st
from langchain_anthropic import ChatAnthropic
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os
import tempfile

# Load environment variables
load_dotenv()

# Streamlit page configuration
st.set_page_config(
    page_title="PDF Question Answering with RAG",
    page_icon="📚",
    layout="wide"
)

st.title("📚 PDF Question Answering with RAG")
st.markdown("Upload a PDF document and ask questions about its content!")

# Initialize session state
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "processed_file" not in st.session_state:
    st.session_state.processed_file = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    
    # Model selection
    model_option = st.selectbox(
        "Select Claude Model",
        ["claude-3-haiku-20240307", "claude-sonnet-4-5-20250929"]
    )
    
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.1)
    
    # Chunk settings
    st.subheader("Chunk Settings")
    chunk_size = st.number_input("Chunk Size", min_value=100, max_value=2000, value=1000, step=100)
    chunk_overlap = st.number_input("Chunk Overlap", min_value=0, max_value=500, value=200, step=50)
    
    # Clear chat history button
    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

# File upload section
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    # Check if this is a new file
    if uploaded_file.name != st.session_state.processed_file:
        with st.spinner("Processing PDF... This may take a few moments."):
            try:
                # Create a temporary file to save the uploaded PDF
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name
                
                # Load the PDF
                loader = PyPDFLoader(file_path=tmp_file_path)
                doc = loader.load()
                
                # Initialize embeddings
                embedding = OpenAIEmbeddings(model='text-embedding-3-small')
                
                # Split documents
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size, 
                    chunk_overlap=chunk_overlap
                )
                chunks = splitter.split_documents(doc)
                
                # Create vector store
                st.session_state.vector_store = Chroma.from_documents(
                    documents=chunks, 
                    embedding=embedding
                )
                
                # Update processed file name
                st.session_state.processed_file = uploaded_file.name
                
                # Clean up temporary file
                os.unlink(tmp_file_path)
                
                st.success(f"✅ Successfully processed {uploaded_file.name}")
                st.info(f"Document split into {len(chunks)} chunks")
                
            except Exception as e:
                st.error(f"Error processing PDF: {str(e)}")
                st.stop()
    else:
        st.info(f"Using previously processed file: {uploaded_file.name}")

# Question answering section
if st.session_state.vector_store is not None:
    st.divider()
    st.header("Ask Questions")
    
    # Initialize LLM and retriever
    llm = ChatAnthropic(model=model_option, temperature=temperature)
    retriever = st.session_state.vector_store.as_retriever()
    
    # Create prompt template
    prompt = ChatPromptTemplate.from_template("""
        Answer the question based on the provided context. If you don't know the answer or if the context doesn't contain relevant information, say you don't know.
        
        Context: {context}
        
        Question: {question}
        
        Answer:
    """)
    
    # Create chain
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    # Question input
    question = st.text_input("Enter your question:", placeholder="What is this document about?")
    
    col1, col2 = st.columns([1, 5])
    with col1:
        ask_button = st.button("Ask", type="primary")
    
    # Process question
    if ask_button and question:
        with st.spinner("Thinking..."):
            try:
                response = chain.invoke(question)
                
                # Add to chat history
                st.session_state.chat_history.append({
                    "question": question,
                    "answer": response
                })
                
            except Exception as e:
                st.error(f"Error generating response: {str(e)}")
    
    # Display chat history
    if st.session_state.chat_history:
        st.divider()
        st.header("Chat History")
        
        for i, chat in enumerate(reversed(st.session_state.chat_history)):
            with st.container():
                st.markdown(f"**Q{len(st.session_state.chat_history)-i}:** {chat['question']}")
                st.markdown(f"**A:** {chat['answer']}")
                st.divider()

else:
    st.info("👆 Please upload a PDF file to get started!")

# Footer
st.markdown("---")
st.markdown("Built with LangChain and Streamlit")