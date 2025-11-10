# 🧠 RAG Document Q&A System (Flask + FAISS + FLAN-T5)

### 📄 Overview
This project is a **Retrieval-Augmented Generation (RAG)** web application built using **Flask**, **FAISS**, and **HuggingFace Transformers**.  
It allows you to **upload PDF documents**, automatically extract and embed their text, and then **ask natural language questions** about them.  

The system retrieves relevant chunks from the document and generates accurate answers using a **FLAN-T5 language model** — all running locally and free.

---

## ⚙️ Key Features

- 📄 **PDF Upload & Extraction:** Automatically extracts text from PDF files using PyPDF2.  
- 🔍 **Semantic Search with FAISS:** Efficient similarity search using embeddings from SentenceTransformers.  
- 🧠 **Retrieval-Augmented Generation (RAG):** Combines retrieval and generation for context-aware answers.  
- 💬 **Question Answering Interface:** Ask questions in plain English about your documents.  
- 🔄 **Document Management:** Upload, list, and delete PDFs with RESTful endpoints.  
- 🧰 **Free Models:** Uses open-source models (SentenceTransformers + FLAN-T5) — no API keys needed.  
- 🌐 **Flask API:** Simple REST API for document upload, querying, and deletion.

---

## 🧩 Architecture

```
          ┌──────────────────┐
          │   PDF Document   │
          └────────┬─────────┘
                   │
             Text Extraction
                   │
          ┌────────▼────────┐
          │  Chunk Splitting │
          └────────┬────────┘
                   │
           Sentence Embeddings
                   │
          ┌────────▼────────┐
          │   FAISS Index   │
          └────────┬────────┘
                   │
             Query Embedding
                   │
          ┌────────▼────────┐
          │  Top-K Retrieval│
          └────────┬────────┘
                   │
            Context + Question
                   │
          ┌────────▼────────┐
          │   FLAN-T5 LLM   │
          └─────────────────┘
                   │
              Final Answer
```

---

## 📁 Project Structure

```
📦 RAG_LEO/
├── app.py                # Flask web server
├── rag_pipeline.py       # Core RAG pipeline (retrieval + generation)
├── utils.py              # Utility functions (save/load pickle, directory setup)
├── templates/
│   └── index.html        # Frontend UI
├── uploads/              # Uploaded PDFs
├── indexes/              # FAISS vector indexes
├── metadata/             # Stored text chunks
└── requirements.txt      # Dependencies
```

---

## 🧰 Tech Stack

| Component | Library |
|------------|----------|
| Backend | Flask |
| Embeddings | SentenceTransformers (`all-MiniLM-L6-v2`) |
| Vector Search | FAISS |
| PDF Parsing | PyPDF2 |
| Generation | HuggingFace Transformers (FLAN-T5) |
| Language | Python 3.8+ |

---

## 🚀 Setup Instructions

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/<your-username>/rag-flask-app.git
cd rag-flask-app
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Flask App
```bash
python app.py
```

Visit: 👉 [http://localhost:5000](http://localhost:5000)

---

## 🧪 API Endpoints

| Endpoint | Method | Description |
|-----------|---------|-------------|
| `/` | GET | Renders main page |
| `/upload` | POST | Upload a PDF and create embeddings |
| `/documents` | GET | List all uploaded documents |
| `/ask` | POST | Query a document using RAG |
| `/document/<doc_id>` | DELETE | Delete a document and its index |
| `/health` | GET | Health and system status |

---

## 💬 Example Usage

### 1️⃣ Upload a PDF
```bash
curl -X POST -F "file=@report.pdf" http://localhost:5000/upload
```

### 2️⃣ Ask a Question
```bash
curl -X POST http://localhost:5000/ask   -H "Content-Type: application/json"   -d '{"query": "What are the key insights from the report?", "doc_id": "<your_doc_id_here>"}'
```

### 3️⃣ Delete a Document
```bash
curl -X DELETE http://localhost:5000/document/<doc_id>
```

---

## 🧠 Model Details

- **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2`  
  → Converts text chunks into semantic vectors.

- **Generation Model:** `google/flan-t5-base`  
  → Generates contextual answers based on retrieved text.

- **Vector Index:** `faiss.IndexFlatIP`  
  → Enables cosine similarity search for top-K document chunks.

---

## ⚙️ Configuration

Modify these parameters in `rag_pipeline.py` for performance tuning:

| Parameter | Description | Default |
|------------|--------------|----------|
| `chunk_size` | Size of text chunks | 1000 |
| `chunk_overlap` | Overlap between chunks | 200 |
| `top_k` | Number of chunks retrieved per query | 4 |
| `max_length` | Max tokens in generated answer | 256 |

---

## 🧩 Example Workflow

1️⃣ Upload `document.pdf`  
2️⃣ Text is extracted, chunked, and embedded  
3️⃣ FAISS index is built and stored  
4️⃣ When queried, top relevant chunks are retrieved  
5️⃣ FLAN-T5 generates an answer using those chunks  

---

## 🧠 Example Response (API)

```json
{
  "answer": "The report highlights that renewable energy investments have grown by 25% in the last year.",
  "retrieved_chunks": ["... relevant text snippet ..."],
  "doc_id": "b2a9b0f3-22e1-4b67-9e7f-90ff3f18c48b",
  "filename": "report.pdf",
  "query": "What are the main points in the report?"
}
```

---

## 📊 Future Enhancements
- Multi-document question answering  
- Persistent vector database (e.g., Chroma or Milvus)  
- UI enhancements using Streamlit or React  
- Source citation and context display  
- Support for DOCX and TXT files  

---

## 🧑‍💻 Author
**Mark Rodrigues**

---

## 📜 License
This project is licensed under the **MIT License** — free to use, modify, and distribute.
