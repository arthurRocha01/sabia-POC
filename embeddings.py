"""Wrapper do provedor de embeddings (Google Gemini).

Ponto ÚNICO de contato com o modelo de embeddings: a ingestão e a busca
passam por aqui, garantindo que os vetores venham sempre do mesmo modelo —
vetores de modelos diferentes não são comparáveis entre si.

A chave é lida do .env na raiz do projeto (GOOGLE_API_KEY).

O free tier do gemini-embedding-001 tem limites apertados (na ordem de ~30k
tokens por minuto). Um livro inteiro passa disso, então os textos são enviados
em lotes pequenos, espaçados no tempo, com espera e nova tentativa quando a API
responde 429 (RESOURCE_EXHAUSTED). Os quatro parâmetros abaixo têm override por
variável de ambiente.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

PROJECT_ROOT = Path(__file__).resolve().parent

# Carrega o .env antes de resolver os nomes/limites (permite override por env).
load_dotenv(PROJECT_ROOT / ".env")

# Modelo de embedding (override por EMBEDDING_MODEL no .env). Trocar de modelo
# exige RE-INGERIR tudo: vetores são específicos do modelo e da dimensão.
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "gemini-embedding-001")

# Orçamento por requisição, em caracteres (~4 caracteres por token). 16k chars
# ≈ 4k tokens: folga confortável dentro do TPM do free tier.
BATCH_CHARS = int(os.environ.get("EMBEDDING_BATCH_CHARS", "16000"))
# Pausa entre lotes, em segundos, para não estourar o TPM.
BATCH_DELAY = float(os.environ.get("EMBEDDING_BATCH_DELAY", "10"))
# Papel do texto no embedding: trechos são DOCUMENTO, consultas são QUERY.
# Sem esses rótulos o modelo usa um comportamento genérico, não otimizado
# para busca assimétrica. Os dois lados PRECISAM ser consistentes.
DOCUMENT_TASK = os.environ.get("EMBEDDING_TASK_DOCUMENT", "RETRIEVAL_DOCUMENT")
QUERY_TASK = os.environ.get("EMBEDDING_TASK_QUERY", "RETRIEVAL_QUERY")

# Em caso de 429: quantas tentativas e quanto esperar entre elas.
MAX_ATTEMPTS = int(os.environ.get("EMBEDDING_MAX_ATTEMPTS", "5"))
RETRY_SLEEP = float(os.environ.get("EMBEDDING_RETRY_SLEEP", "20"))

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


def _batches(texts: list[str], budget_chars: int) -> list[list[str]]:
    """Agrupa textos em lotes que cabem no orçamento de caracteres.

    Sempre coloca ao menos um texto por lote, mesmo que ele sozinho passe do
    orçamento (um chunk enorme não pode travar a ingestão).
    """
    batches: list[list[str]] = []
    current: list[str] = []
    size = 0
    for text in texts:
        if current and size + len(text) > budget_chars:
            batches.append(current)
            current, size = [], 0
        current.append(text)
        size += len(text)
    if current:
        batches.append(current)
    return batches


def _embed_batch(texts: list[str], task_type: str) -> list[list[float]]:
    """Embeda um lote (com o papel explicado em task_type) e trata o limite."""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = _get_client().models.embed_content(
                model=EMBEDDING_MODEL,
                contents=texts,  # type: ignore[arg-type]  # o SDK aceita list[str]
                config=types.EmbedContentConfig(task_type=task_type),
            )
            embeddings = response.embeddings or []
            return [embedding.values or [] for embedding in embeddings]
        except Exception as error:  # noqa: BLE001 — classifica e decide
            limited = getattr(error, "code", None) == 429 or "RESOURCE_EXHAUSTED" in str(error)
            if not limited or attempt == MAX_ATTEMPTS:
                raise
            print(
                f"embeddings: limite de taxa atingido (tentativa {attempt}/"
                f"{MAX_ATTEMPTS}); aguardando {RETRY_SLEEP:.0f}s",
                file=sys.stderr,
            )
            time.sleep(RETRY_SLEEP)
    return []  # inalcançável: o loop ou retorna ou levanta


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embeda vários textos (usado na ingestão), respeitando o limite por minuto."""
    texts = list(texts)
    batches = _batches(texts, BATCH_CHARS)
    vectors: list[list[float]] = []
    for index, batch in enumerate(batches, start=1):
        if index > 1:
            time.sleep(BATCH_DELAY)  # espaça os lotes para caber no TPM
        print(
            f"embeddings: lote {index}/{len(batches)} ({len(batch)} textos)",
            file=sys.stderr,
        )
        vectors.extend(_embed_batch(batch, DOCUMENT_TASK))
    return vectors


def embed_query(text: str) -> list[float]:
    """Embeda uma consulta única (usado na busca semântica)."""
    vectors = _embed_batch([text], QUERY_TASK)
    return vectors[0]
