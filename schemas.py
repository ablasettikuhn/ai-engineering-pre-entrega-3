"""
schemas.py
----------
Contratos de salida del sistema RAG.

- `RespuestaLLM`: lo que el modelo debe devolver. La cadena LCEL la valida con
  `PydanticOutputParser`. Incluye el texto y las referencias, expresadas como los
  números de los fragmentos del CONTEXTO que respaldan la respuesta.
- `RespuestaRAG`: la respuesta final de `get_rag_response()`. Las referencias se
  completan con los metadatos reales de ChromaDB (título, DOI, página), así el
  modelo nunca puede inventar una cita: solo puede señalar fragmentos que recibió.
"""

from pydantic import BaseModel, Field, model_validator

NO_LO_SE = "No lo sé."


class RespuestaLLM(BaseModel):
    respuesta: str = Field(
        description=(
            "Respuesta en español basada ÚNICAMENTE en el CONTEXTO. "
            f"Si el contexto no contiene la información, exactamente: '{NO_LO_SE}'"
        )
    )
    respuesta_encontrada: bool = Field(
        description="true si el CONTEXTO contiene la información para responder; false en caso contrario."
    )
    fragmentos_citados: list[int] = Field(
        default_factory=list,
        description="Números [n] de los fragmentos del CONTEXTO usados para responder. Vacía si no se encontró la respuesta.",
    )

    @model_validator(mode="after")
    def coherencia(self) -> "RespuestaLLM":
        if self.respuesta_encontrada and not self.fragmentos_citados:
            raise ValueError("Una respuesta encontrada debe citar al menos un fragmento del contexto.")
        if not self.respuesta_encontrada:
            # Sin información en el contexto: no hay texto inventado ni citas.
            self.respuesta = NO_LO_SE
            self.fragmentos_citados = []
        self.fragmentos_citados = sorted(set(self.fragmentos_citados))
        return self


class Referencia(BaseModel):
    fragmento: int = Field(description="Número del fragmento en el contexto enviado al LLM.")
    chunk_id: str
    titulo: str
    autores: str
    anio: int | None = None
    doi: str | None = None
    url: str | None = None
    pagina: int | None = Field(default=None, description="Página del PDF donde empieza el fragmento.")
    similitud: float = Field(description="Similitud coseno entre la pregunta y el fragmento (0 a 1).")
    extracto: str = Field(description="Inicio del fragmento, para verificar la cita.")


class RespuestaRAG(BaseModel):
    pregunta: str
    respuesta: str
    respuesta_encontrada: bool
    referencias: list[Referencia] = Field(default_factory=list)
    fragmentos_recuperados: int = Field(description="Cantidad de fragmentos (top_k) enviados al LLM.")
    proveedor: str | None = None
