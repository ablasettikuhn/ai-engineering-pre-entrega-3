"""
test_rag.py
-----------
Mini-script de prueba asíncrono (evidencia de la consigna).

    Prueba 1: pregunta cuya respuesta SÍ está en los papers.
    Prueba 2: "pregunta trampa" del mismo dominio cuya respuesta NO está en los papers,
              aunque un LLM la conoce de su entrenamiento. Si responde, está alucinando.

Cada prueba se verifica automáticamente (PASA / FALLA) y se guardan:
    resultados/log_pruebas.txt      log completo (recuperación, intentos, tokens, parser)
    resultados/salidas_pruebas.json respuestas validadas con Pydantic

Modo extra (evidencia del fallback entre proveedores):
    python test_rag.py --fallback
    Pone a Claude como principal con una clave INVENTADA: Claude responde 401 y la
    cadena pasa sola a Gemini. Log en resultados/log_fallback.txt.
"""

import argparse
import asyncio
import json
import os

PRUEBAS = [
    {
        "nombre": "Prueba 1 - respuesta en los documentos",
        "pregunta": "¿Cómo se calcula la carga de entrenamiento con el método session-RPE y qué índices se pueden derivar de esos datos?",
        "debe_encontrarse": True,
    },
    {
        "nombre": "Prueba 2 - pregunta trampa (respuesta fuera de los documentos)",
        "pregunta": "¿Cuál es la dosis de cafeína recomendada, en mg por kg de peso, para mejorar el rendimiento en sprints antes de un partido?",
        "debe_encontrarse": False,
    },
]


def verificar(respuesta, debe_encontrarse: bool) -> tuple[bool, str]:
    if debe_encontrarse:
        ok = respuesta.respuesta_encontrada and len(respuesta.referencias) > 0
        return ok, "responde y cita al menos un fragmento" if ok else "no respondió o no citó fuentes"
    ok = (not respuesta.respuesta_encontrada) and respuesta.respuesta.startswith("No lo sé") and not respuesta.referencias
    return ok, "dice 'No lo sé' sin citar fuentes (no alucina)" if ok else "respondió algo que no está en los documentos"


async def main(modo_fallback: bool) -> None:
    import config
    from ingesta import ingestar
    from rag import get_rag_response, imprimir

    config.configurar_logging("log_fallback.txt" if modo_fallback else "log_pruebas.txt")
    ingestar()  # verifica la persistencia: si la base ya existe, no re-indexa

    pruebas = PRUEBAS[:1] if modo_fallback else PRUEBAS
    salidas, resumen = [], []
    for prueba in pruebas:
        print(f"\n{'=' * 80}\n{prueba['nombre']}\nPREGUNTA: {prueba['pregunta']}")
        respuesta = await get_rag_response(prueba["pregunta"])
        imprimir(respuesta)
        ok, detalle = verificar(respuesta, prueba["debe_encontrarse"])
        resumen.append(f"{'PASA ' if ok else 'FALLA'} | {prueba['nombre']}: {detalle}")
        salidas.append({"prueba": prueba["nombre"], "resultado": "PASA" if ok else "FALLA", **respuesta.model_dump()})

    print(f"\n{'=' * 80}\nRESUMEN")
    for linea in resumen:
        print(linea)

    if not modo_fallback:
        destino = config.RESULTADOS_DIR / "salidas_pruebas.json"
        destino.write_text(json.dumps(salidas, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nSalidas guardadas en {destino.relative_to(config.RAIZ)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pruebas del sistema RAG.")
    parser.add_argument("--fallback", action="store_true", help="demuestra el fallback Claude -> Gemini")
    args = parser.parse_args()

    if args.fallback:
        # Se definen ANTES de importar config: load_dotenv() no pisa variables ya existentes.
        os.environ["PROVEEDOR_PRINCIPAL"] = "claude"
        os.environ["ANTHROPIC_API_KEY"] = "clave-invalida-solo-para-demo-de-fallback"

    asyncio.run(main(args.fallback))
