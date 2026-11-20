"""
config.py
---------
Configuración compartida por la ingesta y la consulta.

Es la ÚNICA fuente de verdad para el modelo de embeddings: tanto `ingesta.py`
como `rag.py` llaman a `obtener_embeddings()`, así que es imposible indexar con
un modelo y consultar con otro (el error #1 de un RAG). Además, la ingesta guarda
la firma del modelo en `vectorstore/manifest.json` y la consulta la verifica.
"""

import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ChromaDB envía telemetría anónima por defecto; se desactiva antes de importarlo.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------
RAIZ = Path(__file__).resolve().parent
DATA_DIR = RAIZ / "data"
PDF_DIR = DATA_DIR / "pdf"
VECTORSTORE_DIR = RAIZ / "vectorstore"
MANIFEST = VECTORSTORE_DIR / "manifest.json"
RESULTADOS_DIR = RAIZ / "resultados"
FUENTES = RAIZ / "fuentes.json"

COLLECTION_NAME = os.getenv("COLLECTION_NAME", "papers_ciencias_deporte")

# ---------------------------------------------------------------------------
# Embeddings (los mismos para indexar y para consultar)
# ---------------------------------------------------------------------------
EMBEDDINGS_MODEL = os.getenv("EMBEDDINGS_MODEL", "gemini-embedding-001")
EMBEDDINGS_DIM = int(os.getenv("EMBEDDINGS_DIM", "768"))

# Plan gratuito de gemini-embedding-001: ~30.000 tokens por minuto.
# La ingesta envía los fragmentos en lotes que no superan este presupuesto.
EMBEDDINGS_TOKENS_POR_MINUTO = int(os.getenv("EMBEDDINGS_TOKENS_POR_MINUTO", "25000"))

# ---------------------------------------------------------------------------
# Chunking y recuperación
# ---------------------------------------------------------------------------
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))  # tokens (tiktoken cl100k_base)
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))  # tokens
TOP_K = int(os.getenv("TOP_K", "4"))  # fragmentos que recibe el LLM

if CHUNK_SIZE < 500 or CHUNK_OVERLAP < 50:
    raise ValueError("La consigna pide fragmentos de al menos 500 tokens con 50 de solapamiento.")
if not 3 <= TOP_K <= 5:
    raise ValueError("TOP_K debe estar entre 3 y 5 para evitar el 'contexto infinito'.")

# ---------------------------------------------------------------------------
# LLM (lógica de los módulos anteriores: proveedor principal + fallback)
# ---------------------------------------------------------------------------
PROVEEDOR_PRINCIPAL = os.getenv("PROVEEDOR_PRINCIPAL", "gemini").strip().lower()
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-haiku-5-5")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1024"))
MAX_INTENTOS = int(os.getenv("MAX_INTENTOS", "3"))  # 1 intento + 2 reintentos


def hay_clave_gemini() -> bool:
    return bool(os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"))


def hay_clave_claude() -> bool:
    return bool(os.getenv("ANTHROPIC_API_KEY"))


def obtener_embeddings():
    """Devuelve el modelo de embeddings que usan la ingesta y la consulta."""
    if not hay_clave_gemini():
        raise RuntimeError("Falta GOOGLE_API_KEY en el .env: se necesita para los embeddings.")
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    # embed_documents usa task_type RETRIEVAL_DOCUMENT y embed_query usa RETRIEVAL_QUERY.
    # Es el MISMO modelo optimizado para ambos lados de la búsqueda, no una mezcla de modelos.
    return GoogleGenerativeAIEmbeddings(model=EMBEDDINGS_MODEL, output_dimensionality=EMBEDDINGS_DIM)


def firma_embeddings() -> dict:
    """Identifica el espacio vectorial: si cambia, la base debe re-indexarse."""
    return {"embeddings_model": EMBEDDINGS_MODEL, "embeddings_dim": EMBEDDINGS_DIM}


def configurar_logging(archivo: str | None = None) -> None:
    """Log en consola y, opcionalmente, en resultados/<archivo> (evidencia)."""
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if archivo:
        RESULTADOS_DIR.mkdir(exist_ok=True)
        handlers.append(logging.FileHandler(RESULTADOS_DIR / archivo, mode="w", encoding="utf-8"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
        force=True,
    )
    for ruidoso in ("httpx", "httpcore", "chromadb", "google_genai", "anthropic", "urllib3"):
        logging.getLogger(ruidoso).setLevel(logging.WARNING)
