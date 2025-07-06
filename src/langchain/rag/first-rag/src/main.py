import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI


load_dotenv()

# 1. Load the PDF file
pdf_path = 'data/odisseia_homero.pdf'  # Adjust path if needed

if not os.path.exists(pdf_path):
    raise ValueError(f"PDF file path {pdf_path} not found")

loader = PyPDFLoader(pdf_path)
documents = loader.load()

print(f"Loaded {len(documents)} pages from the PDF.")

# 2. Split the PDF into chunks
# Why? LLMs work best with small, overlapping pieces of text
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,      # Each chunk will have up to 500 characters
    chunk_overlap=100    # Chunks will overlap by 100 characters for context
)

chunks = text_splitter.split_documents(documents)
print(f"Split into {len(chunks)} chunks.")

# For learning: print the first chunk as an example
print("First chunk:")
print(chunks[0].page_content)



# 3. Create embeddings for each chunk
# Why? Embeddings turn text into vectors so we can search for similar content
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set. Please set it before running.")

embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

# 4. Store the embeddings in a FAISS vector store (in-memory database for fast similarity search)
vectorstore = FAISS.from_documents(chunks, embeddings)
print("Embeddings created and stored in FAISS vector store.")


# 5. Set up the retriever from the vector store
retriever = vectorstore.as_retriever()

# 6. Set up the OpenAI LLM (ChatGPT)
llm = ChatOpenAI(openai_api_key=openai_api_key, temperature=0)

# 7. Create the RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",  # "stuff" is a simple way to combine retrieved docs
    retriever=retriever,
    return_source_documents=True  # So you can see which chunks were used
)

# 8. Ask questions interactively
print("\nYou can now ask questions about your PDF! Type 'exit' to quit.\n")
while True:
    query = input("Question: ")
    if query.lower() in ("exit", "quit"): break
    result = qa_chain({"query": query})
    print("\nAnswer:", result["result"])
    print("\nSource document excerpt:")
    print(result["source_documents"][0].page_content[:300], "...\n")
