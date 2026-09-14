"""Testes do interpretation.py — offline: nenhuma chamada de rede é feita.

Rode com:  .venv/bin/python tests/test_interpretation.py

Cobre o que já nos atrapalhou na prática: retry só para erro temporário (429/5xx),
falha rápida para erro definitivo, e o despacho entre provedores (deepseek/gemini).
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import interpretation  # noqa: E402

HIT = {
    "book_id": "maquiavel",
    "title": "O Principe",
    "author": "Maquiavel",
    "line": "default",
    "page_start": 45,
    "page_end": 45,
    "label_start": "45",
    "label_end": "45",
    "text": "o príncipe deve conhecer a natureza do povo",
    "score": 0.81,
}


class FakeError(Exception):
    """Erro com .code (SDK do Google) ou .status_code (SDK da OpenAI)."""

    def __init__(self, message: str, code: int | None = None, status_code: int | None = None):
        super().__init__(message)
        if code is not None:
            self.code = code
        if status_code is not None:
            self.status_code = status_code


class StubProvider:
    """Substitui _call_provider: segue um roteiro de erros/respostas."""

    def __init__(self, script: list[object]):
        self.script = list(script)
        self.calls = 0

    def __call__(self, prompt: str) -> str:
        item = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        if isinstance(item, Exception):
            raise item
        return str(item)


def _run(script: list[object]) -> tuple[str, StubProvider, list[float]]:
    """Roda interpret() com provedor e sleep falsos; devolve saída, stub e esperas."""
    stub = StubProvider(script)
    sleeps: list[float] = []
    original_provider = interpretation._call_provider
    original_sleep = interpretation.time.sleep
    interpretation._call_provider = stub
    interpretation.time.sleep = lambda seconds: sleeps.append(float(seconds))
    try:
        output = interpretation.interpret("trecho do usuário", [HIT])
    finally:
        interpretation._call_provider = original_provider
        interpretation.time.sleep = original_sleep
    return output, stub, sleeps


def test_no_hits_skips_api():
    """Sem trechos: guard, sem chamada e sem espera."""
    assert interpretation.interpret("texto", []) == interpretation.NO_CONNECTION
    print("guard sem trechos OK")


def test_retries_transient_error():
    """503 transitório: espera e tenta de novo, e devolve o card quando volta."""
    output, stub, sleeps = _run([FakeError("503 UNAVAILABLE", code=503), "MESMO CONCEITO — card ok"])
    assert output == "MESMO CONCEITO — card ok", output
    assert stub.calls == 2, stub.calls
    assert sleeps == [interpretation.RETRY_SLEEP], sleeps
    print("retry em erro transitorio OK")


def test_transient_error_from_openai_shape():
    """Erro no formato da OpenAI (status_code) também é reconhecido como transitório."""
    output, stub, sleeps = _run([FakeError("503 high demand", status_code=503), "card ok"])
    assert output == "card ok"
    assert stub.calls == 2 and len(sleeps) == 1
    print("erro no formato da OpenAI OK")


def test_gives_up_after_max_attempts():
    """Persistindo o 503, desiste com RuntimeError claro (sem traceback)."""
    try:
        _run([FakeError("503 UNAVAILABLE", code=503)])
    except RuntimeError as error:
        assert "card" in str(error) and "instantes" in str(error), error
    else:
        raise AssertionError("esperava RuntimeError após esgotar as tentativas")
    print("desiste apos as tentativas OK")


def test_non_transient_fails_immediately():
    """Erro definitivo (ex.: 400) não fica tentando: falha na primeira."""
    try:
        _run([FakeError("400 INVALID_ARGUMENT", code=400)])
    except RuntimeError:
        pass
    else:
        raise AssertionError("esperava RuntimeError imediato")
    print("erro definitivo: falha imediata OK")


def test_retry_delay_parsing():
    """O retryDelay é lido do erro; ausente vira None."""
    assert interpretation._retry_delay(FakeError('{"retryDelay": "45s"}', code=429)) == 45.0
    assert interpretation._retry_delay(FakeError("sem retryDelay", code=429)) is None
    print("parse do retryDelay OK")


def test_provider_dispatch():
    """_call_provider despacha conforme LLM_PROVIDER."""
    original_provider = interpretation.LLM_PROVIDER
    original_deepseek, original_gemini = interpretation._call_deepseek, interpretation._call_gemini
    chamadas = []
    interpretation._call_deepseek = lambda prompt: chamadas.append("deepseek") or "d"
    interpretation._call_gemini = lambda prompt: chamadas.append("gemini") or "g"
    try:
        interpretation.LLM_PROVIDER = "deepseek"
        assert interpretation._call_provider("x") == "d"
        interpretation.LLM_PROVIDER = "gemini"
        assert interpretation._call_provider("x") == "g"
        assert chamadas == ["deepseek", "gemini"], chamadas
    finally:
        interpretation.LLM_PROVIDER = original_provider
        interpretation._call_deepseek, interpretation._call_gemini = original_deepseek, original_gemini
    print("despacho de provedor OK")


def test_prompt_is_grounded():
    """O prompt inclui o texto do usuário, as fontes e o guard."""
    prompt = interpretation.build_prompt("meu trecho", [HIT])
    assert "meu trecho" in prompt
    assert "Maquiavel, O Principe, pág. 45" in prompt
    assert "Nenhuma conexão relevante identificada" in prompt
    print("prompt grounded OK")


def main():
    test_no_hits_skips_api()
    test_prompt_is_grounded()
    test_retry_delay_parsing()
    test_provider_dispatch()
    test_retries_transient_error()
    test_transient_error_from_openai_shape()
    test_non_transient_fails_immediately()
    test_gives_up_after_max_attempts()
    print("TODOS OS TESTES PASSARAM")


if __name__ == "__main__":
    main()
