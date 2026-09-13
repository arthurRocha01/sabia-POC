"""Testes do interpretation.py — offline: nenhuma chamada de rede é feita.

Rode com:  .venv/bin/python tests/test_interpretation.py

Cobre o retry para erros temporários do serviço (503/429), que foi o que
derrubou uma demonstração real: o modelo respondeu "high demand, try again".
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
    """Erro com .code, imitando o ServerError/ClientError do SDK."""

    def __init__(self, message: str, code: int | None = None):
        super().__init__(message)
        self.code = code


class FakeResponse:
    """Resposta mínima: só o campo .text é usado pelo interpret()."""

    def __init__(self, text: str):
        self.text = text


class StubModels:
    """Sequência de resultados: erro ou resposta, na ordem das chamadas."""

    def __init__(self, script: list[object]):
        self.script = list(script)
        self.calls = 0

    def generate_content(self, **kwargs):  # noqa: ANN003
        item = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        if isinstance(item, Exception):
            raise item
        return item


class StubClient:
    def __init__(self, script: list[object]):
        self.models = StubModels(script)


def _run(script: list[object]) -> tuple[object, StubModels, list[float]]:
    """Roda interpret() com cliente e sleep falsos; devolve saída, stub e esperas."""
    stub = StubClient(script)
    sleeps: list[float] = []
    original_client = interpretation._get_client
    original_sleep = interpretation.time.sleep
    interpretation._get_client = lambda: stub
    interpretation.time.sleep = lambda seconds: sleeps.append(float(seconds))
    try:
        output = interpretation.interpret("trecho do usuário", [HIT])
    finally:
        interpretation._get_client = original_client
        interpretation.time.sleep = original_sleep
    return output, stub.models, sleeps


def test_retries_transient_error():
    """503 transitório: espera e tenta de novo, e devolve o card quando volta."""
    output, models, sleeps = _run([
        FakeError("503 UNAVAILABLE. high demand", 503),
        FakeResponse("MESMO CONCEITO — card ok"),
    ])
    assert output == "MESMO CONCEITO — card ok", output
    assert models.calls == 2, models.calls
    assert sleeps == [interpretation.RETRY_SLEEP], sleeps
    print("retry em erro transitorio OK")


def test_gives_up_after_max_attempts():
    """Persistindo o 503, desiste com RuntimeError claro (sem traceback)."""
    try:
        _run([FakeError("503 UNAVAILABLE", 503)])
    except RuntimeError as error:
        assert "card" in str(error) and "instantes" in str(error), error
    else:
        raise AssertionError("esperava RuntimeError após esgotar as tentativas")
    print("desiste apos as tentativas OK")


def test_non_transient_fails_immediately():
    """Erro definitivo (ex.: 400) não fica tentando: falha na primeira."""
    stub = StubClient([FakeError("400 INVALID_ARGUMENT", 400)])
    sleeps: list[float] = []
    original_client, original_sleep = interpretation._get_client, interpretation.time.sleep
    interpretation._get_client = lambda: stub
    interpretation.time.sleep = lambda seconds: sleeps.append(float(seconds))
    try:
        try:
            interpretation.interpret("trecho", [HIT])
        except RuntimeError:
            pass
        else:
            raise AssertionError("esperava RuntimeError imediato")
    finally:
        interpretation._get_client = original_client
        interpretation.time.sleep = original_sleep
    assert stub.models.calls == 1, stub.models.calls
    assert sleeps == [], sleeps
    print("erro definitivo: falha imediata OK")


def test_no_hits_skips_api():
    """Sem trechos: guard, sem chamada e sem espera."""
    assert interpretation.interpret("texto", []) == interpretation.NO_CONNECTION
    print("guard sem trechos OK")


def main():
    test_no_hits_skips_api()
    test_retries_transient_error()
    test_non_transient_fails_immediately()
    test_gives_up_after_max_attempts()
    print("TODOS OS TESTES PASSARAM")


if __name__ == "__main__":
    main()
