"""
ingesta.py
----------
Módulo de Ingesta (Setup): data/*.md|*.txt -> chunking -> ChromaDB persistente.

    1. Lee los documentos de /data (con sus metadatos y marcas de página).
    2. Los fragmenta con RecursiveCharacterTextSplitter medido en TOKENS
       (500 tokens con 50 de solapamiento, configurable en el .env).
    3. Calcula los embeddings con el modelo de config.obtener_embeddings().
    4. Los guarda en ./vectorstore (ChromaDB persistente, distancia coseno).

Persistencia: antes de indexar compara una "huella" del dataset (contenido de los
archivos + parámetros de chunking + modelo de embeddings) con la guardada en
vectorstore/manifest.json. Si coinciden, NO re-indexa: ahorra tiempo y cuota.

Uso:
    python ingesta.py            # indexa solo si hace falta
    python ingesta.py --forzar   # borra la colección y re-indexa todo
"""

import argparse
import hashlib
import json
import logging
import re
import time
from datetime import datetime
from pathlib import Path

import tiktoken
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config
from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    DATA_DIR,
    EMBEDDINGS_TOKENS_POR_MINUTO,
    MANIFEST,
    VECTORSTORE_DIR,
)

logger = logging.getLogger("ingesta")

MARCA_PAGINA = re.compile(r"<!-- pagina (\d+) -->\n?")
EXTENSIONES = {".md", ".txt"}


# ---------------------------------------------------------------------------
# 1) Lectura de documentos
# ---------------------------------------------------------------------------
def _leer_cabecera(texto: str) -> tuple[dict, str]:
    """Separa la cabecera '--- clave: valor ---' del cuerpo del documento."""
    if not texto.startswith("---\n"):
        return {}, texto
    fin = texto.index("\n---\n", 4)
    metadatos = {}
    for linea in texto[4:fin].splitlines():
        clave, _, valor = linea.partition(":")
        metadatos[clave.strip()] = valor.strip()
    return metadatos, texto[fin + 5 :]


def archivos_dataset() -> list[Path]:
    if not DATA_DIR.exists():
        return []
    return sorted(p for p in DATA_DIR.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONES)


def cargar_documentos() -> list[Document]:
    """Un Document por archivo. Las marcas de página se quitan del texto y se
    guardan como posiciones, para saber luego en qué página empieza cada fragmento."""
    documentos = []
    for ruta in archivos_dataset():
        metadatos, cuerpo = _leer_cabecera(ruta.read_text(encoding="utf-8"))

        texto, inicios_pagina, cursor = [], [], 0
        partes = MARCA_PAGINA.split(cuerpo)  # [antes, n1, texto1, n2, texto2, ...]
        if partes[0].strip():
            texto.append(partes[0])
            cursor += len(partes[0])
        for numero, contenido in zip(partes[1::2], partes[2::2]):
            inicios_pagina.append((cursor, int(numero)))
            texto.append(contenido)
            cursor += len(contenido)

        documentos.append(
            Document(
                page_content="".join(texto),
                metadata={
                    "fuente": ruta.name,
                    "doc_id": metadatos.get("id", ruta.stem),
                    "titulo": metadatos.get("titulo", ruta.stem),
                    "autores": metadatos.get("autores", "desconocido"),
                    "anio": int(metadatos["anio"]) if metadatos.get("anio", "").isdigit() else 0,
                    "doi": metadatos.get("doi", ""),
                    "url": metadatos.get("url", ""),
                    "licencia": metadatos.get("licencia", ""),
                    "_inicios_pagina": inicios_pagina,
                },
            )
        )
        logger.info("Leído %s (%d caracteres, %d páginas)", ruta.name, cursor, len(inicios_pagina))
    return documentos


# ---------------------------------------------------------------------------
# 2) Chunking
# ---------------------------------------------------------------------------
def crear_splitter() -> RecursiveCharacterTextSplitter:
    # from_tiktoken_encoder mide el tamaño en TOKENS (no en caracteres), como pide la consigna.
    # Corta primero por párrafos, luego por líneas, oraciones y palabras.
    return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        add_start_index=True,
    )


def fragmentar(documentos: list[Document]) -> list[Document]:
    splitter = crear_splitter()
    fragmentos = []
    for doc in documentos:
        inicios_pagina = doc.metadata.pop("_inicios_pagina")
        for n, chunk in enumerate(splitter.split_documents([doc])):
            # Página donde está el centro del fragmento (para citarla en las referencias).
            centro = chunk.metadata.pop("start_index", 0) + len(chunk.page_content) // 2
            pagina = max((num for pos, num in inicios_pagina if pos <= centro), default=0)
            chunk.metadata.update(
                chunk_id=f"{doc.metadata['doc_id']}-{n:03d}",
                pagina=pagina,
                tokens=len(tiktoken.get_encoding("cl100k_base").encode(chunk.page_content)),
            )
            fragmentos.append(chunk)
        n_doc = sum(1 for f in fragmentos if f.metadata["doc_id"] == doc.metadata["doc_id"])
        logger.info("Fragmentado %s -> %d chunks", doc.metadata["fuente"], n_doc)
    return fragmentos


# ---------------------------------------------------------------------------
# 3) Persistencia en ChromaDB
# ---------------------------------------------------------------------------
def huella_dataset() -> str:
    """Cambia si cambia cualquier archivo, el chunking o el modelo de embeddings."""
    h = hashlib.sha256()
    for ruta in archivos_dataset():
        h.update(ruta.name.encode())
        h.update(ruta.read_bytes())
    h.update(json.dumps({"chunk": CHUNK_SIZE, "overlap": CHUNK_OVERLAP, **config.firma_embeddings()}).encode())
    return h.hexdigest()


def leer_manifest() -> dict | None:
    if MANIFEST.exists():
        return json.loads(MANIFEST.read_text(encoding="utf-8"))
    return None


def abrir_vectorstore():
    """Abre (o crea) la colección persistente. La usan la ingesta y la consulta."""
    from langchain_chroma import Chroma

    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=config.obtener_embeddings(),
        persist_directory=str(VECTORSTORE_DIR),
        collection_configuration={"hnsw": {"space": "cosine"}},
    )


def _lotes_por_tokens(fragmentos: list[Document]) -> list[list[Document]]:
    """Agrupa los fragmentos para no superar el límite de tokens por minuto del plan gratuito."""
    lotes, actual, tokens = [], [], 0
    for f in fragmentos:
        if actual and tokens + f.metadata["tokens"] > EMBEDDINGS_TOKENS_POR_MINUTO:
            lotes.append(actual)
            actual, tokens = [], 0
        actual.append(f)
        tokens += f.metadata["tokens"]
    if actual:
        lotes.append(actual)
    return lotes


def _agregar_con_reintentos(vectorstore, lote: list[Document], intentos: int = 3) -> None:
    for intento in range(1, intentos + 1):
        try:
            vectorstore.add_documents(lote, ids=[f.metadata["chunk_id"] for f in lote])
            return
        except Exception as exc:  # 429 = límite de cuota del plan gratuito
            if ("429" in str(exc) or "RESOURCE_EXHAUSTED" in str(exc)) and intento < intentos:
                logger.warning("Límite de cuota de embeddings (429). Espero 60 s y reintento (%d/%d)", intento, intentos)
                time.sleep(60)
            else:
                raise


def ingestar(forzar: bool = False) -> int:
    """Indexa /data en ChromaDB si hace falta. Devuelve la cantidad de chunks en la colección."""
    if not archivos_dataset():
        raise RuntimeError(f"No hay archivos .md/.txt en {DATA_DIR}. Ejecuta antes descargar_papers.py.")

    huella = huella_dataset()
    manifest = leer_manifest()
    vectorstore = abrir_vectorstore()
    en_coleccion = len(vectorstore.get(include=[])["ids"])

    if not forzar and manifest and manifest.get("huella") == huella and en_coleccion == manifest.get("chunks"):
        logger.info(
            "La base vectorial ya existe y está actualizada (%d chunks, embeddings %s): no se re-indexa.",
            en_coleccion, manifest["embeddings_model"],
        )
        return en_coleccion

    motivo = "re-indexación forzada" if forzar else ("primera indexación" if not manifest else "el dataset o la configuración cambió")
    logger.info("Indexando (%s)...", motivo)
    if en_coleccion:
        vectorstore.reset_collection()
    MANIFEST.unlink(missing_ok=True)  # si la ingesta se interrumpe, la próxima ejecución re-indexa

    fragmentos = fragmentar(cargar_documentos())
    lotes = _lotes_por_tokens(fragmentos)
    inicio = time.perf_counter()
    for i, lote in enumerate(lotes, start=1):
        if i > 1:
            logger.info("Pausa de 60 s para respetar el límite de tokens por minuto de los embeddings...")
            time.sleep(60)
        _agregar_con_reintentos(vectorstore, lote)
        logger.info("Lote %d/%d: %d chunks embebidos y guardados", i, len(lotes), len(lote))

    total = len(vectorstore.get(include=[])["ids"])
    MANIFEST.write_text(
        json.dumps(
            {
                "huella": huella,
                "chunks": total,
                **config.firma_embeddings(),
                "chunk_size_tokens": CHUNK_SIZE,
                "chunk_overlap_tokens": CHUNK_OVERLAP,
                "documentos": [p.name for p in archivos_dataset()],
                "fecha": datetime.now().isoformat(timespec="seconds"),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    logger.info("Ingesta completa: %d chunks en '%s' (%.1f s)", total, COLLECTION_NAME, time.perf_counter() - inicio)
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description="Indexa /data en ChromaDB.")
    parser.add_argument("--forzar", action="store_true", help="borra la colección y re-indexa todo")
    args = parser.parse_args()
    config.configurar_logging("log_ingesta.txt")
    ingestar(forzar=args.forzar)


if __name__ == "__main__":
    main()
