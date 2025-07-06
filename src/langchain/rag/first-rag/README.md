# RAG PDF QA Study

This project is an **initial study of Retrieval-Augmented Generation (RAG) in Generative AI**. The goal is to learn how to use RAG to answer questions about the content of a PDF using OpenAI's language models and LangChain.

## What is RAG?
RAG (Retrieval-Augmented Generation) is a technique that combines information retrieval (finding relevant documents or text chunks) with generative AI (like GPT-3.5/4) to answer questions based on external knowledge sources, such as PDFs, websites, or databases.

## Project Goal
- **Learn the basics of RAG** by building a simple question-answering system over a PDF.
- **Understand each step**: loading, chunking, embedding, storing, retrieving, and generating answers.

## How It Works
The main logic is in `src/main.py`. Here's what happens step by step:

### 1. Load Environment Variables
- Loads your OpenAI API key from a `.env` file using `dotenv`.

### 2. Load the PDF
- Loads the PDF file (`data/odisseia_homero.pdf`) using LangChain's `PyPDFLoader`.

### 3. Split the PDF into Chunks
- Splits the PDF into small, overlapping text chunks using `RecursiveCharacterTextSplitter`.
- This helps the language model understand and retrieve relevant context.

### 4. Create Embeddings
- Uses OpenAI's embedding model to convert each chunk into a vector (numerical representation).

### 5. Store Embeddings in a Vector Store
- Stores all chunk embeddings in a FAISS vector database for fast similarity search.

### 6. Set Up the Retriever
- Prepares a retriever that can find the most relevant chunks for any user question.

### 7. Set Up the LLM (OpenAI)
- Initializes the OpenAI language model (ChatGPT) for answer generation.

### 8. Create the RetrievalQA Chain
- Combines the retriever and LLM into a RetrievalQA chain using LangChain.
- This chain retrieves relevant chunks and uses them to answer questions.

### 9. Interactive Q&A
- The script enters a loop where you can type questions about the PDF.
- For each question, it:
  - Retrieves the most relevant chunks from the PDF.
  - Uses the LLM to generate an answer based on those chunks.
  - Prints the answer and an excerpt from the source chunk(s).

## How to Run
1. Place your PDF in the `data/` folder (default: `odisseia_homero.pdf`).
2. Set your OpenAI API key in a `.env` file:
   ```
   OPENAI_API_KEY=sk-...
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Run the script:
   ```
   python src/main.py
   ```
5. Ask questions about the PDF in the terminal.

## Learning Focus
- Understand the RAG pipeline: retrieval + generation.
- See how LLMs can answer questions using your own documents.
- Learn how to inspect which parts of the document were used for each answer.

---

Feel free to experiment with different PDFs, chunk sizes, or questions to deepen your understanding of RAG!


