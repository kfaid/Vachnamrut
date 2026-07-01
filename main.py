from dotenv import load_dotenv
import os

from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# -----------------------------
# Embedding Model
# -----------------------------
embedding_model = MistralAIEmbeddings(
    model="mistral-embed"
)

# -----------------------------
# Pinecone Vector Store
# -----------------------------
vectorstore = PineconeVectorStore(
    index_name=os.getenv("PINECONE_INDEX_NAME"),
    embedding=embedding_model,
)

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 4,
        "fetch_k": 10,
        "lambda_mult": 0.5,
    },
)

# -----------------------------
# LLM
# -----------------------------
llm = ChatMistralAI(
    model="mistral-small-2506"
)

# -----------------------------
# Prompt
# -----------------------------
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a helpful AI assistant.

Answer ONLY using the provided context.

If the answer is not present in the context, reply exactly:

"I could not find the answer in the document."

Keep your answer concise and do not use Markdown formatting.
"""
        ),
        (
            "human",
            """
Context:
{context}

Question:
{question}
"""
        ),
    ]
)

print("✅ RAG System Created")
print("Type 0 to exit.\n")

while True:
    query = input("You: ")

    if query == "0":
        break

    # Retrieve documents
    docs = retriever.invoke(query)

    context = "\n\n".join(
        doc.page_content for doc in docs
    )

    final_prompt = prompt.invoke(
        {
            "context": context,
            "question": query,
        }
    )

    response = llm.invoke(final_prompt)

    print("\nAI:", response.content)
    print("-" * 80)