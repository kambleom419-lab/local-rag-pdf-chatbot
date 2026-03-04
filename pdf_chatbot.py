from langchain_ollama import OllamaLLM
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

# Load PDF
loader = PyPDFLoader("sample.pdf")  # Put your PDF in same folder
documents = loader.load()

# Split into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

# Create embeddings
embeddings = OllamaEmbeddings(model="phi3:mini")

# Store in Chroma
vectorstore = Chroma.from_documents(docs, embeddings, persist_directory="./chroma_db")

# Create LLM
llm = OllamaLLM(model="phi3:mini")

# Ask question
while True:
    query = input("Ask a question: ")
    results = vectorstore.similarity_search(query, k=3)

    context = "\n\n".join([doc.page_content for doc in results])
    prompt = f"Answer the question using the context below:\n\n{context}\n\nQuestion: {query}"

    response = llm.invoke(prompt)
    print("\nAnswer:\n", response)
