# Chatbot using Multiagent RAG

## Overview
This project implements a **Multiagent RAG (Retrieval-Augmented Generation)** chatbot using **FastAPI**, **LangChain**, and **ChromaDB**. The chatbot retrieves relevant documents from a vector store and uses an **LLM (Large Language Model)** to generate responses.

## Features
- Multi-agent RAG pipeline
- Uses **FAISS** or **ChromaDB** for retrieval
- FastAPI backend for easy deployment
- Supports open-source LLMs like **LLaMA 2, Mistral, Falcon**
- Easy installation with a virtual environment

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/your-repo/Chatbot-using-Multiagent-RAG.git
cd Chatbot-using-Multiagent-RAG
```

### 2. Create a Virtual Environment
```bash
python -m venv rag_env
```

### 3. Activate the Virtual Environment
- **macOS/Linux**:
  ```bash
  source rag_env/bin/activate
  ```
- **Windows (Command Prompt)**:
  ```bash
  rag_env\Scripts\activate
  ```

### 4. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Run the FastAPI Server
```bash
uvicorn main:app --reload
```

### 6. API Endpoints
Once the server is running, you can interact with the chatbot using the following endpoints:

#### **1. Ask a Question**
- **Endpoint:** `GET /ask`
- **Description:** Sends a query to the chatbot and receives a response.
- **Request Example:**
  ```bash
  curl -X GET "http://127.0.0.1:8000/ask?query=What is RAG?"
  ```
- **Response Example:**
  ```json
  {
    "answer": "Retrieval-Augmented Generation (RAG) is...",
    "CODE": "200"
  }
  ```

#### **2. List Available Agents**
- **Endpoint:** `GET /agents`
- **Description:** Returns a list of available agents in the chatbot.
- **Request Example:**
  ```bash
  curl -X GET "http://127.0.0.1:8000/agents"
  ```
- **Response Example:**
  ```json
  {
    "agents": ["Agent1", "Agent2"]
  }
  ```

### 7. Open API Documentation
You can explore the API using Swagger UI:
- Open `http://127.0.0.1:8000/docs` in a browser.

## Folder Structure
```
Chatbot-using-Multiagent-RAG/
│── main.py               # FastAPI entry point
│── multiagent_rag.py     # RAG pipeline implementation
│── vectorstore.py        # ChromaDB/FAISS retriever setup
│── requirements.txt      # Python dependencies
│── README.md             # Project documentation
```

## Troubleshooting
If you get **ModuleNotFoundError**, ensure:
1. The virtual environment is activated (`source rag_env/bin/activate`).
2. Dependencies are installed (`pip install -r requirements.txt`).
3. You're running `uvicorn` inside the virtual environment.


## Author
Developed by **Rohit Kumar** 

