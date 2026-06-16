---
title: DocuRAG
emoji: 🤖
colorFrom: indigo
colorTo: purple
sdk: streamlit
sdk_version: 1.35.0
app_file: app/frontend/app.py
pinned: false
---

# DOCURAG - AI Document Assistant with RAG


DOCURAG is a complete, lightweight, and modern **Retrieval-Augmented Generation (RAG)** system designed to run entirely with free resources. It extracts text from PDF and TXT files, chunks them, generates dense vector representations locally using `sentence-transformers`, stores them in a local `FAISS` index, and queries the **Gemini API** using a grounded system prompt to deliver accurate, non-hallucinated QA.

---

## 🚀 Key Features

1. **Multi-format parsing:** Supports direct ingestion of standard PDF and UTF-8 TXT files.
2. **Deterministic Sliding Chunker:** Custom word-boundary-aware text splitter that maintains paragraph semantics.
3. **Local Vector Engine:** Embedding calculations run on your CPU using `all-MiniLM-L6-v2` (384 dimensions) via `sentence-transformers`—completely offline and free.
4. **FAISS Indexing:** Fast vector similarity searching using Facebook AI Similarity Search (`IndexFlatL2`).
5. **Grounded QA Response:** Generates a structured response based *only* on context references, ensuring Gemini does not synthesize external or hallucinated knowledge.
6. **Dual-Layer Architecture:** Features a clean decoupling with a FastAPI backend server and a premium, responsive Streamlit chat client.
7. **Zero-Configuration Startup:** Streamlit will automatically launch the backend FastAPI process in the background. Running the application requires just a single command.

---

## 🛠️ Tech Stack

- **Backend API:** FastAPI + Uvicorn
- **Frontend UI:** Streamlit
- **PDF Extraction:** PyPDF
- **Vector Index:** FAISS (CPU)
- **Embeddings:** Hugging Face `sentence-transformers` (`all-MiniLM-L6-v2`)
- **LLM Reasoning:** Google Gemini API (`gemini-1.5-flash` - Free Tier)

---

## 📁 Folder Structure

```
DOCURAG/
├── .streamlit/
│   └── config.toml             # Streamlit premium theme configurations
├── app/
│   ├── __init__.py
│   ├── backend/
│   │   ├── __init__.py
│   │   ├── config.py           # Configuration values & system paths
│   │   ├── main.py             # FastAPI entry point & routers
│   │   └── rag.py              # Main RAG logic (extraction, embeddings, FAISS, Gemini)
│   └── frontend/
│       ├── __init__.py
│       └── app.py              # Streamlit UI & background backend starter
├── requirements.txt            # Package dependencies
├── README.md                   # System documentation
└── run.py                      # Local run shortcut script
```

---

## 💻 Local Installation & Setup

Follow these steps to run DOCURAG locally:

### 1. Clone or Move to Project Folder
Ensure you are in the root directory:
```bash
cd DOCURAG
```

### 2. Set Up a Python Virtual Environment
We recommend using a Python 3.10+ virtual environment:
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
Install all required libraries:
```bash
pip install -r requirements.txt
```

### 4. Configure Your API Key
Create a `.env` file in the root of the project:
```env
GEMINI_API_KEY=your_gemini_api_key_here
```
*(Optionally, you can enter the Gemini API Key directly inside the Streamlit sidebar during runtime.)*

### 5. Run the Application
Launch both Streamlit and FastAPI with a single command:
```bash
python run.py
```
After launching, your default web browser will open to the Streamlit UI (typically `http://localhost:8501`).

---

## 🤗 Deploying to Hugging Face Spaces

DOCURAG is pre-configured to be deployed directly on **Hugging Face Spaces** for free! Follow these simple steps:

### 1. Create a Hugging Face Space
1. Log in to [Hugging Face](https://huggingface.co/).
2. Click on **New Space** (under your profile menu or at [huggingface.co/new-space](https://huggingface.co/new-space)).
3. Provide a Name (e.g., `docurag`).
4. Select **Streamlit** as the Space SDK.
5. Choose **Public** or **Private** visibility.
6. Click **Create Space**.

### 2. Configure Environment Secrets
To prevent exposing your Gemini API key in Git commits, add it as a Hugging Face Space secret:
1. In your newly created Space, navigate to **Settings**.
2. Scroll to the **Variables and secrets** section.
3. Click **New secret**.
4. Set the **Name** to `GEMINI_API_KEY`.
5. Set the **Value** to your Google Gemini API key.
6. Click **Save**.

### 3. Upload Code files
You can upload the repository code to Hugging Face either by linking it with Git or using the Hugging Face Web interface.

If using Git:
```bash
# Initialize and push to HF Space git repository
git init
git lfs install
git remote add origin https://huggingface.co/spaces/<your-username>/<your-space-name>
git add .
git commit -m "Initial deploy of DOCURAG"
git push -u origin main
```

### 4. Hugging Face Execution Flow
Upon deployment, Hugging Face automatically detects the Streamlit application structure. It will execute the entry script, which starts the Streamlit interface. 

Streamlit's internal helper code inside `app.py` detects that the backend FastAPI server on port `8000` is offline and automatically runs `uvicorn` in a background subprocess. The frontend then establishes communications seamlessly, providing a fully operational RAG server out-of-the-box!
