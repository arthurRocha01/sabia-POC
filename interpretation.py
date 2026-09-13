"""Interpretação: transforma os trechos recuperados em um card de insights.

Responsabilidade única: montar o prompt grounded (texto do usuário + trechos de
outros autores, com fonte) e pedir ao LLM a síntese + classificação da relação.

O guard anti-alucinação é parte do contrato: sem trechos recuperados não há
interpretação — e o prompt proíbe inferir além do que foi fornecido.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai

PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")

# Modelo de interpretação (override por LLM_MODEL no .env).
LLM_MODEL = os.environ.get("LLM_MODEL", "gemini-3.6-flash")

# Erros temporários do serviço (não do nosso código): vale esperar e tentar de novo.
TRANSIENT_CODES = {429, 500, 502, 503, 504}
MAX_ATTEMPTS = int(os.environ.get("LLM_MAX_ATTEMPTS", "4"))
RETRY_SLEEP = float(os.environ.get("LLM_RETRY_SLEEP", "10"))

# Resposta obrigatória quando não há relação sustentada pelos trechos.
NO_CONNECTION = "Nenhuma conexão relevante identificada."

PROMPT_TEMPLATE = """Você é um assistente acadêmico de síntese e leitura comparativa.

[TEXTO SELECIONADO PELO USUÁRIO]:
"{selected_text}"

[TRECHOS RECUPERADOS DE OUTROS LIVROS]:
{chunks}

INSTRUÇÕES:
1. Analise a relação entre o texto selecionado e os trechos de outros autores.
2. Escreva uma síntese de no máximo 3 frases concisas.
3. Classifique explicitamente a relação (COMPLEMENTO, CONTRADIÇÃO, NUANCE ou MESMO CONCEITO).
4. Cite a fonte exata (Autor, Nome do Livro e Página).
5. Responda apenas com base nos trechos fornecidos. Se não houver relação direta relevante, responda: "Nenhuma conexão relevante identificada."
"""

_client = None


def _get_client():
    """Cria o cliente Gemini no primeiro uso; erro claro se faltar a chave."""
    global _client
    if _client is None:
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GOOGLE_API_KEY não definida — crie o arquivo .env na raiz do "
                "projeto com GOOGLE_API_KEY=sua_chave antes de interpretar."
            )
        _client = genai.Client(api_key=api_key)
    return _client


def format_citation(hit: dict) -> str:
    """Citação legível do trecho: usa a numeração do livro (rótulo) do chunk."""
    start, end = hit["label_start"], hit["label_end"]
    page = f"pág. {start}" if start == end else f"págs. {start}-{end}"
    return f"{hit['author']}, {hit['title']}, {page}"


def _format_sources(hits: list[dict]) -> str:
    """Bloco de trechos no formato que o prompt espera, cada um com sua fonte."""
    return "\n\n".join(
        f"[{format_citation(hit)}]\n{hit['text']}" for hit in hits
    )


def build_prompt(selected_text: str, hits: list[dict]) -> str:
    """Monta o prompt grounded com o texto do usuário e os trechos recuperados."""
    return PROMPT_TEMPLATE.format(
        selected_text=selected_text.strip(),
        chunks=_format_sources(hits),
    )


def interpret(selected_text: str, hits: list[dict]) -> str:
    """Devolve o card de insights (síntese + relação + fontes).

    Sem trechos recuperados não há o que interpretar: devolve o guard, sem
    gastar chamada de API.
    """
    if not hits:
        return NO_CONNECTION

    prompt = build_prompt(selected_text, hits)
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = _get_client().models.generate_content(
                model=LLM_MODEL, contents=prompt
            )
            return (response.text or "").strip() or NO_CONNECTION
        except Exception as error:  # noqa: BLE001 — classifica e decide
            code = getattr(error, "code", None)
            if code not in TRANSIENT_CODES or attempt == MAX_ATTEMPTS:
                # Falha definitiva (ou tentativas esgotadas): o CLI mostra o
                # motivo sem despejar traceback, e ainda exibe os trechos.
                raise RuntimeError(
                    f"não foi possível gerar o card agora: "
                    f"{type(error).__name__} (código {code}) — tente de novo em instantes"
                ) from error
            print(
                f"interpretacao: modelo indisponível (tentativa {attempt}/"
                f"{MAX_ATTEMPTS}); aguardando {RETRY_SLEEP:.0f}s",
                file=sys.stderr,
            )
            time.sleep(RETRY_SLEEP)
    return NO_CONNECTION  # inalcançável: o loop ou retorna ou levanta
