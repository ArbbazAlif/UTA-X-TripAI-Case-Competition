import os
import uuid
import streamlit as st
import boto3
from pinecone import Pinecone
from langchain.vectorstores import Pinecone as VectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from botocore.exceptions import NoCredentialsError, ClientError
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
r2_endpoint = os.getenv("CLOUDFLARE_R2_ENDPOINT")
r2_access_key = os.getenv("CLOUDFLARE_R2_ACCESS_KEY")
r2_secret_key = os.getenv("CLOUDFLARE_R2_SECRET_KEY")
bucket_name = os.getenv("R2_BUCKET_NAME")

# Initialize Pinecone with existing index
pc = Pinecone(api_key=pinecone_api_key)
index_name = "rag1"
index = pc.Index(index_name)

# Initialize embeddings and vector store
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key, model="text-embedding-ada-002")
vector_store = VectorStore(index, embeddings.embed_query, "text")

# Initialize LLM
chat_llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=openai_api_key,temperature=0)

# Initialize Cloudflare R2 client
s3_client = boto3.client(
    "s3",
    endpoint_url=r2_endpoint,
    aws_access_key_id=r2_access_key,
    aws_secret_access_key=r2_secret_key
)

def upload_to_r2(uploaded_file):
    """Upload file to R2 with unique filename"""
    try:
        file_id = str(uuid.uuid4())
        file_name = f"{file_id}_{uploaded_file.name}"
        s3_client.upload_fileobj(uploaded_file, bucket_name, file_name)
        return file_name
    except NoCredentialsError:
        st.error("❌ Cloudflare R2 credentials not found")
        return None
    except Exception as e:
        st.error(f"❌ Upload failed: {str(e)}")
        return None

def process_document(file_name):
    """Process document from R2 and index in Pinecone"""
    temp_path = f"/tmp/{file_name}"
    try:
        # Download from R2
        with open(temp_path, 'wb') as f:
            s3_client.download_fileobj(bucket_name, file_name, f)
        
        if not os.path.exists(temp_path):
            st.error("❌ Failed to download file")
            return

        # Determine file type
        file_ext = file_name.split(".")[-1].lower()
        
        # Load document
        try:
            if file_ext == "pdf":
                loader = PyPDFLoader(temp_path)
            elif file_ext == "txt":
                loader = TextLoader(temp_path)
            elif file_ext == "csv":
                loader = CSVLoader(temp_path)
            else:
                st.error("❌ Unsupported file type")
                return
        except Exception as e:
            st.error(f"❌ Error loading document: {str(e)}")
            return

        # Split and index documents
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_documents(documents)
        
        if chunks:
            texts = [doc.page_content for doc in chunks]
            metadatas = [{"source": file_name} for doc in chunks]
            vector_store.add_texts(texts, metadatas=metadatas)
            st.success(f"✅ Processed {len(chunks)} chunks from {file_name}")

    except Exception as e:
        st.error(f"❌ Processing error: {str(e)}")
    finally:
        # Cleanup files
        if os.path.exists(temp_path):
            os.remove(temp_path)
        try:
            s3_client.delete_object(Bucket=bucket_name, Key=file_name)
        except Exception as e:
            st.error(f"❌ R2 cleanup failed: {str(e)}")

def get_conversation_chain():
    """Create conversation chain with session-specific memory"""
    if 'memory' not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
    
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 10}
    )
    
    return ConversationalRetrievalChain.from_llm(
        llm=chat_llm,
        retriever=retriever,
        memory=st.session_state.memory,
        get_chat_history=lambda h: h,
        verbose=True
    )

# Streamlit UI
st.title("📚 Aircraft Analytics Chatbot")

# File upload section
uploaded_file = st.file_uploader("Upload document (PDF, TXT, CSV)", type=["pdf", "txt", "csv"])
if uploaded_file:
    with st.spinner("⏳ Uploading and processing document..."):
        file_name = upload_to_r2(uploaded_file)
        if file_name:
            process_document(file_name)

# Chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.spinner("💭 Thinking..."):
        try:
            chain = get_conversation_chain()
            response = chain({"question": prompt})["answer"]
        except Exception as e:
            response = f"❌ Error: {str(e)}"
    
    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})