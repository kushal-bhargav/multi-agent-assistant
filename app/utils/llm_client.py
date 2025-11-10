# import os
# import requests

# OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
# CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "llama3.2:3b")
# EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

# OLLAMA_OPTIONS = {
#     "num_predict": int(os.getenv("OLLAMA_NUM_PREDICT", "256")),
#     "num_ctx": int(os.getenv("OLLAMA_NUM_CTX", "1024")),
#     "temperature": float(os.getenv("OLLAMA_TEMPERATURE", "0.2")),
# }

# class ChatResponse:
#     class Choice:
#         class Message:
#             def __init__(self, content: str):
#                 self.content = content
#         def __init__(self, content: str):
#             self.message = ChatResponse.Choice.Message(content)
#     def __init__(self, content: str):
#         self.choices = [ChatResponse.Choice(content)]

# def chat(messages, model: str = CHAT_MODEL):
#     prompt = "\n".join([m.get("content", "") for m in messages])
#     try:
#         r = requests.post(f"{OLLAMA_HOST}/api/generate", json={"model": model, "prompt": prompt, "options": OLLAMA_OPTIONS, "stream": False})
#         r.raise_for_status()
#         data = r.json()
#         return ChatResponse(data.get("response", ""))
#     except requests.HTTPError as e:
#         if e.response is not None and e.response.status_code == 404:
#             # fallback to chat API
#             chat_body = {
#                 "model": model,
#                 "messages": [{"role": m.get("role", "user"), "content": m.get("content", "")} for m in messages],
#                 "options": OLLAMA_OPTIONS,
#                 "stream": False,
#             }
#             rc = requests.post(f"{OLLAMA_HOST}/api/chat", json=chat_body)
#             rc.raise_for_status()
#             data = rc.json()
#             return ChatResponse(data.get("message", {}).get("content", ""))
#         raise

# class EmbeddingsResponse:
#     class DataItem:
#         def __init__(self, embedding):
#             self.embedding = embedding
#     def __init__(self, embedding):
#         self.data = [EmbeddingsResponse.DataItem(embedding)]

# def embed(texts, model: str = EMBED_MODEL):
#     if isinstance(texts, str):
#         texts = [texts]
#     r = requests.post(f"{OLLAMA_HOST}/api/embeddings", json={"model": model, "prompt": "\n".join(texts)})
#     r.raise_for_status()
#     data = r.json()
#     return EmbeddingsResponse(data.get("embedding", []))



import os
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "bge-m3:latest")
EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "qwen2:7b")
USE_OPENAI = os.getenv("USE_OPENAI", "False").lower() in ("true", "1", "yes")

if USE_OPENAI:
    chat_client = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4"),
                             api_key=os.getenv("OPENAI_API_KEY"),
                             base_url=os.getenv("OPENAI_BASE_URL", None),
                             temperature=0)
    embed_client = OpenAIEmbeddings(model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002"),
                                   api_key=os.getenv("OPENAI_API_KEY"),
                                   base_url=os.getenv("OPENAI_BASE_URL", None))
else:
    chat_client = OllamaLLM(model=CHAT_MODEL, base_url=OLLAMA_HOST, temperature=0, format='json')
    embed_client = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_HOST)

class ChatResponse:
    class Choice:
        class Message:
            def __init__(self, content: str):
                self.content = content
        def __init__(self, content: str):
            self.message = ChatResponse.Choice.Message(content)
    def __init__(self, content: str):
        self.choices = [ChatResponse.Choice(content)]

def chat(messages, model: str = CHAT_MODEL):
    prompt = "\n".join([m.get("content", "") for m in messages])
    try:
        if USE_OPENAI:
            response = chat_client.call(messages)
            return ChatResponse(response)
        else:
            # Use Ollama LLM chat call
            response = chat_client.invoke(messages)
            return ChatResponse(response)
    except Exception as e:
        raise e

class EmbeddingsResponse:
    class DataItem:
        def __init__(self, embedding):
            self.embedding = embedding
    def __init__(self, embedding):
        self.data = [EmbeddingsResponse.DataItem(embedding)]

def embed(texts, model: str = EMBED_MODEL):
    if isinstance(texts, str):
        texts = [texts]
    if USE_OPENAI:
        embeddings = embed_client.embed_documents(texts)
    else:
        embeddings = embed_client.embed_documents(texts)
    return EmbeddingsResponse(embeddings)
