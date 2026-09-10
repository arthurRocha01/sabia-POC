"""Sondagem de modelos: descobre, com a SUA chave, o que dá para usar.

Uso:  .venv/bin/python scripts/probe_models.py

Não grava nada e não altera o projeto: só lista os modelos disponíveis para a
chave e testa um embedding e uma geração em cada candidato, mostrando o
tamanho do vetor ou o erro exato (modelo inexistente, sem free tier, etc.).
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(PROJECT_ROOT / ".env")

from google import genai  # noqa: E402

# Candidatos a testar — o desenho original do projeto usava os dois primeiros.
EMBEDDING_CANDIDATES = ["text-embedding-004", "gemini-embedding-001", "gemini-embedding-2"]
LLM_CANDIDATES = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-3.5-flash",
    "gemini-3.1-flash-lite",
]


def main() -> None:
    key = _api_key()
    client = genai.Client(api_key=key)

    print("=== modelos visiveis para esta chave ===")
    try:
        for model in client.models.list():
            actions = getattr(model, "supported_actions", None) or []
            flags = ",".join(a for a in actions if a in ("generateContent", "embedContent"))
            print(f" - {model.name}  [{flags or '?'}]")
    except Exception as error:  # noqa: BLE001 — só diagnóstico
        print(f" ! falha ao listar modelos: {error}")

    print("\n=== embeddings ===")
    for name in EMBEDDING_CANDIDATES:
        try:
            response = client.models.embed_content(model=name, contents=["teste de sondagem"])
            embeddings = response.embeddings or []
            vector = embeddings[0].values if embeddings else None
            dims = len(vector) if vector else 0
            print(f" OK   {name}: vetor de {dims} dimensoes")
        except Exception as error:  # noqa: BLE001
            print(f" ERRO {name}: {type(error).__name__}: {error}")

    print("\n=== geracao (interpretacao) ===")
    for name in LLM_CANDIDATES:
        try:
            response = client.models.generate_content(
                model=name, contents="Responda apenas com a palavra: ok"
            )
            print(f" OK   {name}: {(response.text or '').strip()[:40]!r}")
        except Exception as error:  # noqa: BLE001
            print(f" ERRO {name}: {type(error).__name__}: {error}")


def _api_key() -> str:
    """Lê GOOGLE_API_KEY do ambiente/.env, com mensagem clara se faltar."""
    import os

    key = os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise SystemExit(
            "GOOGLE_API_KEY não encontrada. Copie .env.example para .env e "
            "preencha com sua chave do AI Studio."
        )
    return key


if __name__ == "__main__":
    main()
