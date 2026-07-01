# Load PDF
# Split into chunks
# Create embeddings
# Store in Pinecone

from dotenv import load_dotenv

# pyrefly: ignore [missing-import]
from langchain_community.document_loaders import PyPDFLoader
# pyrefly: ignore [missing-import]
from langchain_text_splitters import RecursiveCharacterTextSplitter
# pyrefly: ignore [missing-import]
from langchain_mistralai import MistralAIEmbeddings
# pyrefly: ignore [missing-import]
from langchain_pinecone import PineconeVectorStore

load_dotenv()

# Load PDF
loader = PyPDFLoader(
    "/Users/kk/AI_Projects/Vachnamrut/document_loder/vachnamrut-english.pdf"
)

docs = loader.load()

# Split documents
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)

chunks = splitter.split_documents(docs)

# Mistral Embeddings
embedding_model = MistralAIEmbeddings(
    model="mistral-embed"
)

# Upload to Pinecone
vectorstore = PineconeVectorStore.from_documents(
    documents=chunks,
    embedding=embedding_model,
    index_name="vachnamrut-rag"   # Your Pinecone index name
)

print("Documents uploaded successfully!")