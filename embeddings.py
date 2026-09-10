"""Wrapper do provedor de embeddings (Google Gemini).

Ponto ÚNICO de contato com o modelo de embeddings: a ingestão e a busca
passam por aqui, garantindo que os vetores venham sempre do mesmo modelo —
vetores de modelos diferentes não são comparáveis entre si.

A chave é lida do .env na raiz do projeto (GOOGLE_API_KEY).
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai

PROJECT_ROOT = Path(__file__).resolve().parent
EMBEDDING_MODEL = "text-embedding-004"
# A API de embeddings aceita lotes; livros grandes viram vários pedidos.
BATCH_SIZE = 100

load_dotenv(PROJECT_ROOT / ".env")

_client = None


def _get_client():
    """Cria o cliente Gemini no primeiro uso; erro claro se faltar a chave."""
    global _client
    if _client is None:
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GOOGLE_API_KEY não definida — crie o arquivo .env na raiz "
                "do projeto com GOOGLE_API_KEY=sua_chave antes de ingerir "
                "ou consultar."
            )
        _client = genai.Client(api_key=api_key)
    return _client


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embeda vários textos (usado na ingestão), em lotes de BATCH_SIZE."""
    texts = list(texts)
    vectors: list[list[float]] = []
    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start:start + BATCH_SIZE]
        response = _get_client().models.embed_content(
            model=EMBEDDING_MODEL,
            contents=batch,  # type: ignore[arg-type]  # o SDK aceita list[str]; a tipagem é frouxa
        )
        # Defensivo: a tipagem do SDK marca values como opcional.
        vectors.extend(e.values or [] for e in response.embeddings or [])
    return vectors


def embed_query(text: str) -> list[float]:
    """Embeda uma consulta única (usado na busca semântica)."""
    return embed_texts([text])[0]
