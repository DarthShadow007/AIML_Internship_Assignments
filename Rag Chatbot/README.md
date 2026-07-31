# 🚀 Magnum Opus: Autonomous AI & RAG Engineering Suite

An enterprise-grade, full-stack Artificial Intelligence repository featuring an autonomous Retrieval-Augmented Generation (RAG) assistant, dense vector search engines, live web knowledge fallback, and generative LLM integration.

---

## 🌟 Key Features

* **Autonomous Knowledge Retrieval**: Multi-stage retrieval combining dense vector similarity with live Wikipedia API fallback when local confidence thresholds are not met.
* **Generative SOTA Intelligence**: Powered by **Gemini 3.1 Flash Lite** for conversational synthesis and context-grounded response generation.
* **Zero External DB Dependencies**: Custom-engineered TF-IDF vector space and Cosine Similarity scoring built directly into the system for lightning-fast local execution.
* **Live Ingestion & Active Learning**: Real-time vector memory updating allows new facts and identity anchors to be embedded on the fly without restarting the engine.
* **Integrated Interactive Web UI**: Modern dark-mode web application providing live response latency tracking, active context sources, and system telemetry.

---

## 🛠️ Tech Stack

| Component | Technology |
| :--- | :--- |
| **Language** | Python 3.10+ |
| **Generative LLM** | Google Gemini 3.1 Flash Lite API |
| **Retrieval Engine** | Custom TF-IDF Vector Store + Cosine Similarity |
| **Web Server** | Standard HTTP Server & REST API (`http.server`) |
| **Frontend UI** | HTML5, CSS3 (Flexbox/Grid), JavaScript (Fetch API) |
| **Web API Fallback** | Wikipedia REST API |

---

## [IMPORTANT] 🔑 Setup & Execution Guide for Evaluators

To run the interactive RAG Chatbot locally:

1. **Set your Gemini API Key:**
   Open `09_rag_chatbot.py` and replace `"YOUR_API_KEY_HERE"` with your Google Gemini API key:
   ```python
   GEMINI_API_KEY = "YOUR_API_KEY_HERE"


## 📂 Repository Structure

```text
.
├── Rag Chatbot/
│   ├── 09_rag_chatbot.py      # Main Autonomous RAG & Server Script
│   └── README.md              # Detailed RAG Architecture Documentation
├── datasets/                  # Knowledge base sources & data pipelines
└── README.md                  # Root Repository Documentation

# 2. RAG Chatbot Module: `Rag Chatbot/README.md`

```markdown
# 🧠 Autonomous RAG Chatbot & Generative Engine

This module implements a full-lifecycle **Retrieval-Augmented Generation (RAG)** system capable of vector indexing, real-time semantic retrieval, autonomous fallback retrieval over the open web, and natural language synthesis using **Gemini 3.1 Flash Lite**.

---

## 📐 System Architecture

```text
               +-----------------------+
               |   User Input Query    |
               +-----------+-----------+
                           |
                           v
              +------------+------------+
              | Vector Space Search     |
              | (TF-IDF + Cosine Sim)   |
              +------------+------------+
                           |
            [Relevance Score >= 15%] ?
                     /           \
               YES  /             \  NO
                   /               \
                  v                 v
   +--------------+-----+   +-------+---------------+
   | Extract Local      |   | Autonomous Wikipedia  |
   | Vector Context     |   | Live Search Fallback  |
   +--------------+-----+   +-------+---------------+
                  \                /
                   \              /
                    v            v
            +-------+------------+-------+
            |  Gemini 3.1 Flash Lite     |
            |  Prompt Construction       |
            +-------+------------+-------+
                    |
                    v
            +-------+------------+-------+
            | Grounded Conversational   |
            | Response + Latency Stats  |
            +----------------------------+

