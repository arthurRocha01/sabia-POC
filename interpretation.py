"""Interpretação: transforma os trechos recuperados em um card de insights.

Responsabilidade única: montar o prompt grounded (texto do usuário + trechos de
outros autores, com fonte) e pedir ao LLM a síntese + classificação da relação.

Provedores suportados (LLM_PROVIDER no .env): "deepseek" (padrão) e "gemini".
O DeepSeek fala a API compatível com a OpenAI (base https://api.deepseek.com).

Duas proteções contra o que já nos atrapalhou na prática:
  - timeout explícito e max_retries=0: sem isso o SDK fica retentando por dentro
    e uma chamada pode pendurar por minutos sem nenhuma mensagem;
  - retry próprio, visível, só para erros temporários (429/5xx), usando o
    retryDelay sugerido pela API quando disponível.

O guard anti-alucinação é parte do contrato: sem trechos recuperados não há
interpretação — e o prompt proíbe inferir além do que foi fornecido.
"""

from __future__ import annotations

import os
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types
from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")

# Provedor e modelo do card (override por LLM_PROVIDER / LLM_MODEL no .env).
# "deepseek-v4-flash" é o mesmo modelo que o Hermes usa aqui; a API também
# aceita o nome atual "deepseek-flash" (ambos servidos pelo V4.1 Flash).
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "deepseek").strip().lower()
DEFAULT_MODELS = {
    "deepseek": "deepseek-v4-flash",
    "gemini": "gemini-3.6-flash",
}
LLM_MODEL = os.environ.get("LLM_MODEL", DEFAULT_MODELS.get(LLM_PROVIDER, ""))

DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
REQUEST_TIMEOUT = float(os.environ.get("LLM_TIMEOUT", "60"))

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

_gemini_client = None
_deepseek_client = None


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


# ---------------------------------------------------------------------------
# Chamada por provedor (timeout explícito, sem retry interno do SDK)
# ---------------------------------------------------------------------------

def _get_gemini_client():
    """Cliente Gemini no primeiro uso; erro claro se faltar a chave."""
    global _gemini_client
    if _gemini_client is None:
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GOOGLE_API_KEY não definida — necessária para LLM_PROVIDER=gemini."
            )
        _gemini_client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=int(REQUEST_TIMEOUT * 1000)),
        )
    return _gemini_client


def _get_deepseek_client():
    """Cliente DeepSeek (OpenAI-compatível) no primeiro uso."""
    global _deepseek_client
    if _deepseek_client is None:
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError(
                "DEEPSEEK_API_KEY não definida — crie o .env na raiz com "
                "DEEPSEEK_API_KEY=sua_chave (ou use LLM_PROVIDER=gemini)."
            )
        # max_retries=0: o retry é nosso, com log visível e respeitando o
        # retryDelay da API. Sem isso, o SDK retenta por dentro e a chamada
        # pode pendurar por minutos sem sinal nenhum.
        _deepseek_client = OpenAI(
            api_key=api_key,
            base_url=DEEPSEEK_BASE_URL,
            timeout=REQUEST_TIMEOUT,
            max_retries=0,
        )
    return _deepseek_client


def _call_gemini(prompt: str) -> str:
    """Uma chamada ao Gemini (sem retry)."""
    response = _get_gemini_client().models.generate_content(
        model=LLM_MODEL, contents=prompt
    )
    return (response.text or "").strip()


def _call_deepseek(prompt: str) -> str:
    """Uma chamada ao DeepSeek (API compatível com a da OpenAI)."""
    response = _get_deepseek_client().chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return (response.choices[0].message.content or "").strip()


def _call_provider(prompt: str) -> str:
    """Despacha para o provedor configurado."""
    if LLM_PROVIDER == "gemini":
        return _call_gemini(prompt)
    return _call_deepseek(prompt)


def _error_code(error: Exception) -> int | None:
    """Código HTTP do erro, seja ele do SDK do Google (.code) ou da OpenAI (.status_code)."""
    for attribute in ("code", "status_code"):
        value = getattr(error, attribute, None)
        if isinstance(value, int):
            return value
    return None


def _retry_delay(error: Exception) -> float | None:
    """Extrai o retryDelay sugerido pela API, quando informado."""
    match = re.search(r"""['"]retryDelay['"]\s*:\s*['"]([0-9.]+)s""", str(error))
    return float(match.group(1)) if match else None


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
            return _call_provider(prompt) or NO_CONNECTION
        except Exception as error:  # noqa: BLE001 — classifica e decide
            code = _error_code(error)
            if code not in TRANSIENT_CODES or attempt == MAX_ATTEMPTS:
                raise RuntimeError(
                    f"não foi possível gerar o card agora ({LLM_PROVIDER}/"
                    f"{LLM_MODEL}): {type(error).__name__} (código {code}) — "
                    "tente de novo em instantes"
                ) from error
            delay = _retry_delay(error) or RETRY_SLEEP
            print(
                f"interpretacao: provedor indisponível (tentativa {attempt}/"
                f"{MAX_ATTEMPTS}); aguardando {delay:.0f}s",
                file=sys.stderr,
            )
            time.sleep(delay)
    return NO_CONNECTION  # inalcançável: o loop ou retorna ou levanta
