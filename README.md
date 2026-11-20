# Pre-entrega 3 — Sistema de recuperación semántica local (RAG)

**Asistente de consulta sobre papers de ciencias del deporte, con respuestas basadas solo en los documentos.**

Autora: Alexandra Blasetti Kuhn — Comisión #102135

---

## La aplicación real

Un preparador físico tiene una biblioteca de artículos científicos sobre monitoreo de la carga de entrenamiento y quiere hacerle preguntas en lenguaje natural ("¿cómo se calcula la carga con session-RPE?", "¿qué error tienen las unidades GNSS de 10 Hz al medir distancia?"). Un LLM solo responde de memoria y puede inventar datos. Este sistema, en cambio:

1. **Toma los papers desde su enlace**: descarga el PDF, extrae el texto y lo guarda en `/data` con sus metadatos (título, autores, DOI, licencia).
2. **Los indexa una sola vez** en una base vectorial local (ChromaDB).
3. **Ante cada pregunta**, busca los 4 fragmentos más relevantes y le pide al LLM que responda **solo con ese contexto**, citando de qué paper y de qué página sale cada dato.
4. **Si la respuesta no está en los papers, dice "No lo sé."** en lugar de inventarla.

Es el mismo patrón que usa una empresa para que un asistente responda sobre sus manuales o normativas: el conocimiento vive en los documentos, no en el modelo.

## Dataset: 3 papers reales de acceso abierto

| Archivo en `/data` | Paper | Licencia |
|---|---|---|
| `torres-ronda-2022.md` | Torres-Ronda L, et al. (2022). *Tracking Systems in Team Sports: A Narrative Review of Applications of the Data and Sport Specific Analysis.* Sports Medicine - Open, 8:15. [doi:10.1186/s40798-022-00408-z](https://doi.org/10.1186/s40798-022-00408-z) | CC BY 4.0 |
| `haddad-2017.md` | Haddad M, et al. (2017). *Session-RPE Method for Training Load Monitoring: Validity, Ecological Usefulness, and Influencing Factors.* Frontiers in Neuroscience, 11:612. [doi:10.3389/fnins.2017.00612](https://doi.org/10.3389/fnins.2017.00612) | CC BY 4.0 |
| `beato-2018.md` | Beato M, et al. (2018). *The Validity and Between-Unit Variability of GNSS Units (STATSports Apex 10 and 18 Hz) for Measuring Distance and Peak Speed in Team Sports.* Frontiers in Physiology, 9:1288. [doi:10.3389/fphys.2018.01288](https://doi.org/10.3389/fphys.2018.01288) | CC BY 4.0 |

Los tres artículos tienen licencia Creative Commons Attribution 4.0, que permite redistribuir el texto citando la fuente; cada `.md` conserva la referencia completa en su cabecera. Los enlaces de descarga están en `fuentes.json`, y `descargar_papers.py` genera los `.md` a partir de ellos (sin la bibliografía ni los encabezados de página). Los papers están en inglés y el sistema responde en español.

## Estructura del repositorio

| Archivo | Contenido |
|---|---|
| `fuentes.json` | Enlaces y metadatos de los papers. Para agregar uno, se suma una entrada aquí. |
| `descargar_papers.py` | Paso 0 (opcional): enlace → PDF → texto limpio → `data/<id>.md`. |
| `ingesta.py` | **Módulo de Ingesta**: lee `/data`, fragmenta (500 tokens, 50 de solapamiento) y persiste en ChromaDB. No re-indexa si la base ya está al día. |
| `rag.py` | **Retriever + Generación Grounded**: cadena LCEL asíncrona y función `get_rag_response(query)`. También sirve para hacer preguntas desde la terminal. |
| `schemas.py` | Modelos Pydantic: `RespuestaLLM` (salida del `PydanticOutputParser`) y `RespuestaRAG` (texto + referencias). |
| `config.py` | Configuración compartida y la función única `obtener_embeddings()`. |
| `test_rag.py` | Las dos pruebas de la consigna, con verificación automática PASA/FALLA, y la demostración del fallback. |
| `data/` | Dataset de ejemplo (`.md`). |
| `resultados/` | Evidencia de ejecuciones reales: logs y salidas JSON. |
| `requirements.txt` / `.env.example` | Dependencias y variables de entorno (todas las que lee el código). |

## Arquitectura

```
INGESTA (una vez)                                       CONSULTA (cada pregunta)
fuentes.json ─► descargar_papers.py                     pregunta
                  PDF ─► texto ─► data/*.md                │
                                     │                     ▼
                                     ▼              recuperar(): embedding de la pregunta
                     ingesta.py: chunking 500/50 tokens     (mismo modelo) ─► ChromaDB ─► top 4
                                     │                     │
                     embeddings (gemini-embedding-001)     ▼
                                     │              formatear_contexto(): "[1] (Haddad et al., 2017, p. 3) ..."
                                     ▼                     │
                     ./vectorstore  (ChromaDB, coseno)     ▼
                     + manifest.json (huella y modelo)  PROMPT | LLM | PydanticOutputParser | validar_citas
                                                           │   .with_retry()  .with_fallbacks()
                                                           ▼
                                                    RespuestaRAG: respuesta + referencias (título, DOI, página)
```

## Cumplimiento de la consigna

| Requisito | Dónde se cumple |
|---|---|
| Ingesta: leer `/data`, chunking estratégico (mínimo 500 tokens, 50 de overlap) | `ingesta.py` → `cargar_documentos()` y `fragmentar()`: `RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=500, chunk_overlap=50)`, medido en tokens. |
| Persistir en ChromaDB local | `ingesta.py` → `abrir_vectorstore()`: `persist_directory="./vectorstore"`. |
| Verificar si la base ya existe antes de re-indexar | `ingesta.py` → `ingestar()`: compara la huella del dataset con `vectorstore/manifest.json`; si coincide, no re-indexa. |
| Mismo modelo de embeddings para indexar y consultar | `config.py` → `obtener_embeddings()` es la única fuente. Además, `rag.py` → `obtener_vectorstore()` verifica el modelo y la dimensión guardados en el manifest y se detiene si no coinciden. |
| Retriever: pregunta → embedding → fragmentos relevantes | `rag.py` → `recuperar()`: `aembed_query()` + búsqueda por similitud con `TOP_K=4` (validado entre 3 y 5). |
| Prompt "filtro de veracidad" que diga "No lo sé" | `rag.py` → `MENSAJE_SISTEMA`. |
| Cadena LCEL con `PydanticOutputParser` | `rag.py` → `construir_generador()` y `obtener_cadena_rag()`. |
| Función asíncrona `get_rag_response(query)` con respuesta Pydantic (texto + referencias) | `rag.py` → `get_rag_response()` devuelve `RespuestaRAG` (`schemas.py`). |
| Dos pruebas: respuesta presente y pregunta trampa | `test_rag.py` (evidencia en `resultados/`). |
| Sin claves en el repositorio | `.env` excluido en `.gitignore`; plantilla en `.env.example`. |

## Decisiones técnicas

**Referencias que no se pueden inventar.** El modelo no escribe las citas: solo indica los números `[n]` de los fragmentos que usó (`fragmentos_citados`). Luego, `validar_citas()` rechaza cualquier número que no exista en el contexto, y `armar_respuesta()` completa título, autores, DOI y página con los metadatos reales de ChromaDB.

**Dos capas contra la alucinación.** La primera es el prompt, que exige usar solo el contexto. La segunda es el esquema: si `respuesta_encontrada` es `false`, el validador de `RespuestaLLM` fija el texto en "No lo sé." y vacía las citas; si es `true`, exige al menos una cita.

**Pregunta trampa del mismo dominio.** La prueba 2 pregunta por la dosis de cafeína para mejorar el sprint. Es un tema de ciencias del deporte que el LLM conoce de su entrenamiento, pero que no aparece en ninguno de los tres papers. Si el sistema respondiera con una cifra, estaría usando conocimiento externo, que es justo lo que no debe hacer.

**Chunking en tokens y con número de página.** El tamaño se mide con `tiktoken` (`cl100k_base`), como en la Unidad 2. Es una aproximación a los tokens de Gemini, que usa otro tokenizador. Las marcas `<!-- pagina N -->` de cada `.md` permiten guardar en qué página está cada fragmento.

**Persistencia con huella.** La huella combina el contenido de los archivos, los parámetros de chunking y el modelo de embeddings. Si cambia algo, la ingesta re-indexa sola; si no, la base se reutiliza sin gastar cuota.

**Plan gratuito de Gemini.** `gemini-embedding-001` admite unos 30.000 tokens por minuto en el plan gratuito, así que la ingesta envía los fragmentos en lotes de hasta 25.000 tokens, con una pausa de 60 s entre lotes y reintento ante un error 429.

**Proveedores (lógica de los módulos anteriores).** Gemini (`gemini-3.1-flash-lite`) es el proveedor principal por ser gratuito. Claude queda como fallback si se define `ANTHROPIC_API_KEY`, y el orden se cambia con `PROVEEDOR_PRINCIPAL`. Cada proveedor tiene `.with_retry()` ante JSON inválido o respuestas cortadas por límite de tokens (`finish_reason`).

## Cómo ejecutarlo (Windows, Command Prompt)

Requiere Python 3.12 o superior (desarrollado y probado con 3.14) y una clave de Gemini de [Google AI Studio](https://aistudio.google.com/apikey).

`requirements.txt` instala LangChain en sus paquetes modulares (`langchain-core`, `langchain-text-splitters`, `langchain-chroma`), ChromaDB, Pydantic y los proveedores elegidos: `langchain-google-genai` en lugar de OpenAI y `langchain-anthropic` para el fallback opcional.

```
git clone https://github.com/ablasettikuhn/ai-engineering-pre-entrega-3.git
cd ai-engineering-pre-entrega-3
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
notepad .env
```

En el `.env`, completar `GOOGLE_API_KEY`. Luego:

| Comando | Qué hace |
|---|---|
| `python ingesta.py` | Indexa `/data` en `./vectorstore`. La primera vez tarda 1 o 2 minutos por las pausas de cuota. |
| `python ingesta.py` (otra vez) | Detecta que la base existe y está al día, y no re-indexa. |
| `python test_rag.py` | Ejecuta las dos pruebas de la consigna. |
| `python test_rag.py --fallback` | Demuestra el fallback Claude → Gemini. No necesita clave de Claude: usa una inválida a propósito. |
| `python rag.py "¿Qué error de distancia tienen las unidades GNSS de 10 Hz?"` | Hace una pregunta. |
| `python rag.py` | Modo interactivo (`salir` para terminar). |

Para regenerar el dataset desde los enlaces: `python descargar_papers.py --forzar` y después `python ingesta.py`. Si un sitio bloquea la descarga automática, el script indica cómo bajar el PDF desde el navegador y dónde guardarlo.

La primera ejecución de `tiktoken` descarga su tabla de tokens (~1,7 MB) y la guarda en caché.

## Evidencia (`resultados/`)

| Archivo | Qué muestra |
|---|---|
| `log_descarga.txt` | Generación de los 3 `.md` a partir de los PDF (el de Torres-Ronda, descargado a mano). |
| `log_ingesta.txt` | Fragmentación por documento, lotes de embeddings y total de chunks persistidos. |
| `log_pruebas.txt` | Las dos pruebas: fragmentos recuperados con su similitud, intentos, tokens, validación del parser y resultado PASA/FALLA. Al inicio, la ingesta detecta que la base ya existe y no re-indexa. |
| `salidas_pruebas.json` | Las dos respuestas validadas (`RespuestaRAG`) con sus referencias. |
| `log_fallback.txt` | Claude como principal con una clave **inventada** responde 401 y la cadena pasa sola a Gemini. Es una falla simulada a propósito, sin costo. |

## Limitaciones

- La extracción de texto de PDF a dos columnas puede desordenar tablas o leyendas de figuras. Para este caso es suficiente, porque las respuestas salen del texto corrido.
- El PDF de Torres-Ronda se descargó manualmente desde el navegador, porque el sitio de la editorial bloquea las descargas automáticas; el script lo detecta y lo procesa igual.
- El número de página es el de la página del PDF donde está la mayor parte del fragmento.
- Las respuestas son tan buenas como la recuperación: si la pregunta usa términos muy distintos a los del paper, los fragmentos relevantes pueden no entrar en el top 4.
