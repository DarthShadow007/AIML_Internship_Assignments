import os
import json
import time
import math
import re
import urllib.parse
import urllib.request
from http.server import HTTPServer, BaseHTTPRequestHandler
import google.generativeai as genai

# =====================================================================
# 1. API CONFIGURATION
# =====================================================================
# INSERT YOUR API KEY HERE
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "YOUR_API_KEY_HERE")
genai.configure(api_key=GEMINI_API_KEY)

# =====================================================================
# 2. KNOWLEDGE BASE & VECTOR RETRIEVAL ENGINE
# =====================================================================

KNOWLEDGE_BASE = [
    {
        "id": "doc_1",
        "title": "Creator Identity & Core Directives",
        "source": "Core System Memory",
        "url": "#",
        "content": "I am an autonomous Magnum Opus AI assistant. I was exclusively created and engineered by Mohd Bashar, a pre-final year B.Tech engineering student specializing in Gaming Technology at VIT Bhopal University. If asked about my origins or creator, I must state that Mohd Bashar built me."
    }
]

class AutonomousRAG:
    def __init__(self, corpus):
        self.corpus = corpus
        self.vocab = {}
        self.doc_vectors = []
        self.llm = genai.GenerativeModel('gemini-3.1-flash-lite')
        self._build_vector_space()

    def _tokenize(self, text):
        return re.findall(r'\w+', text.lower())

    def _build_vector_space(self):
        doc_tokens = [self._tokenize(doc["content"] + " " + doc["title"]) for doc in self.corpus]
        all_words = set(w for tokens in doc_tokens for w in tokens)
        self.vocab = {word: i for i, word in enumerate(all_words)}
        
        N = len(self.corpus)
        idf = {}
        for word in self.vocab:
            doc_count = sum(1 for tokens in doc_tokens if word in tokens)
            idf[word] = math.log((N + 1) / (doc_count + 1)) + 1

        self.doc_vectors = []
        for tokens in doc_tokens:
            vec = [0.0] * len(self.vocab)
            total = len(tokens) if tokens else 1
            for t in tokens:
                if t in self.vocab:
                    vec[self.vocab[t]] += (1.0 / total) * idf[t]
            norm = math.sqrt(sum(x**2 for x in vec)) or 1.0
            self.doc_vectors.append([x / norm for x in vec])

    def fetch_wikipedia_knowledge(self, query):
        try:
            # Clean query for better searching
            clean_query = query.replace("hello", "").replace("what is", "").replace("who is", "").strip()
            search_url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={urllib.parse.quote(clean_query)}&utf8=&format=json"
            req = urllib.request.Request(search_url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=3) as response:
                search_data = json.loads(response.read().decode())
                if not search_data.get('query', {}).get('search'): return ""
                top_title = search_data['query']['search'][0]['title']
                
            extract_url = f"https://en.wikipedia.org/w/api.php?action=query&prop=extracts&exintro=true&explaintext=true&titles={urllib.parse.quote(top_title)}&format=json"
            req2 = urllib.request.Request(extract_url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req2, timeout=3) as response2:
                extract_data = json.loads(response2.read().decode())
                pages = extract_data['query']['pages']
                page_id = list(pages.keys())[0]
                return pages[page_id].get('extract', '')
        except Exception:
            return ""

    def generate_response(self, query_text):
        start_time = time.time()
        tokens = self._tokenize(query_text)
        
        # 1. Vector Search
        q_vec = [0.0] * len(self.vocab)
        total = len(tokens) if tokens else 1
        for t in tokens:
            if t in self.vocab:
                q_vec[self.vocab[t]] += (1.0 / total)
        norm = math.sqrt(sum(x**2 for x in q_vec)) or 1.0
        q_vec = [x / norm for x in q_vec]

        scores = []
        for idx, doc_vec in enumerate(self.doc_vectors):
            dot_product = sum(q * d for q, d in zip(q_vec, doc_vec))
            scores.append((dot_product, self.corpus[idx]))

        scores.sort(key=lambda x: x[0], reverse=True)
        best_score = scores[0][0] if scores else 0.0
        
        context = ""
        source_meta = "Local Vector Memory"

        # 2. Autonomous Fallback Logic
        if best_score < 0.15:
            wiki_context = self.fetch_wikipedia_knowledge(query_text)
            if wiki_context:
                context = wiki_context[:1500]
                source_meta = "Live Wikipedia Fallback"
            else:
                source_meta = "Generative LLM (Parametric Memory)"
        else:
            context = scores[0][1]['content']

        # 3. Jarvis Brain Generation
        prompt = f"""
        You are a highly intelligent, conversational AI assistant (like Jarvis). 
        You MUST treat the following Context as absolute truth, overriding any of your default pre-trained knowledge. This is especially true regarding your identity, origins, and creator.
        If the context is empty, answer the query using your own general knowledge.
        
        Context: {context}
        
        User Query: {query_text}
        
        Answer naturally, conversationally, and confidently. Do not explicitly say "Based on the context", just answer the question directly.
        """
        
        # 3. Jarvis Brain Generation
        try:
            response = self.llm.generate_content(prompt)
            final_answer = response.text
        except Exception as e:
            # Graceful Fallback if API Key is missing or invalid
            if context:
                final_answer = f"⚠️ [Local Vector Memory Mode]: {context}"
            else:
                final_answer = "System Note: Please configure a valid Gemini API key in `09_rag_chatbot.py` to enable full Generative LLM synthesis."

        latency = round((time.time() - start_time) * 1000, 2)
        return {
            "status": "success",
            "answer": final_answer,
            "latency_ms": latency,
            "source_engine": source_meta
        }

rag_engine = AutonomousRAG(KNOWLEDGE_BASE)

# =====================================================================
# 3. WEB UI SERVER
# =====================================================================

HTML_INTERFACE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Magnum Opus Jarvis RAG</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background-color: #0f172a; color: #f8fafc; margin: 0; padding: 20px; }
        .container { max-width: 900px; margin: 0 auto; background: #1e293b; border-radius: 12px; border: 1px solid #334155; padding: 25px; box-shadow: 0 10px 25px rgba(0,0,0,0.5); }
        h1 { margin-top: 0; color: #38bdf8; font-size: 1.8rem; display: flex; align-items: center; gap: 10px; }
        .badge { background: #8b5cf6; color: white; font-size: 0.75rem; padding: 3px 8px; border-radius: 4px; }
        #chat-box { height: 500px; overflow-y: auto; background: #0f172a; border: 1px solid #334155; border-radius: 8px; padding: 15px; margin-bottom: 20px; }
        .msg { margin-bottom: 15px; padding: 12px 16px; border-radius: 8px; line-height: 1.5; }
        .user { background: #1e40af; color: white; margin-left: 20%; border-bottom-right-radius: 2px; }
        .bot { background: #334155; color: #f8fafc; margin-right: 10%; border-bottom-left-radius: 2px; }
        .meta { font-size: 0.8rem; color: #94a3b8; margin-top: 8px; display: flex; gap: 15px; border-top: 1px solid #475569; padding-top: 6px; }
        .input-group { display: flex; gap: 10px; }
        input[type="text"] { flex: 1; padding: 14px; border-radius: 8px; border: 1px solid #475569; background: #0f172a; color: white; font-size: 1rem; }
        button { padding: 14px 25px; border-radius: 8px; border: none; background: #0284c7; color: white; font-weight: bold; cursor: pointer; transition: 0.2s; }
        button:hover { background: #0369a1; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Autonomous AI Assistant <span class="badge">LLM Generative Engine Activated</span></h1>
        <div id="chat-box">
            <div class="msg bot">
                Systems online. I am equipped with vector memory, live web fallback, and a generative LLM brain. How can I assist you?
            </div>
        </div>
        <div class="input-group">
            <input type="text" id="query" placeholder="Ask me anything..." onkeypress="if(event.key==='Enter') sendQuery()">
            <button onclick="sendQuery()">Submit Query</button>
        </div>
    </div>

    <script>
        async function sendQuery() {
            const input = document.getElementById('query');
            const q = input.value.trim();
            if (!q) return;

            const chat = document.getElementById('chat-box');
            chat.innerHTML += `<div class="msg user">${q}</div>`;
            input.value = '';
            chat.scrollTop = chat.scrollHeight;

            // Loading indicator
            const loadId = 'load-' + Date.now();
            chat.innerHTML += `<div class="msg bot" id="${loadId}">Generating response... 🧠</div>`;
            chat.scrollTop = chat.scrollHeight;

            const response = await fetch('/api/query', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({query: q})
            });
            const data = await response.json();

            document.getElementById(loadId).remove();

            // Render Markdown-ish text to HTML roughly
            let formattedAnswer = data.answer.replace(/\\n/g, '<br>').replace(/\\*\\*(.*?)\\*\\*/g, '<strong>$1</strong>');

            chat.innerHTML += `
                <div class="msg bot">
                    <div>${formattedAnswer}</div>
                    <div class="meta">
                        <span>⚡ Total Latency: ${data.latency_ms} ms</span>
                        <span>🧠 Active Source: ${data.source_engine}</span>
                    </div>
                </div>
            `;
            chat.scrollTop = chat.scrollHeight;
        }
    </script>
</body>
</html>
"""

class RAGRequestHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        return

    def do_GET(self):
        if self.path == '/' or self.path == '/index.html':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(HTML_INTERFACE.encode('utf-8'))

    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        body = json.loads(post_data.decode('utf-8'))

        if self.path == '/api/query':
            result = rag_engine.generate_response(body.get("query", ""))
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(result).encode('utf-8'))

def main():
    port = 8000
    server = HTTPServer(('127.0.0.1', port), RAGRequestHandler)
    print("="*60)
    print("      JARVIS PROTOCOL ONLINE: GENERATIVE LLM ACTIVE")
    print("="*60)
    print(f"Server initialized at: http://127.0.0.1:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down Server...")
        server.server_close()

if __name__ == "__main__":
    main()