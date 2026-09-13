"""Testes do cli.py e do interpretation.py — offline, sem API key.

Rode com:  .venv/bin/python tests/test_cli.py

O que precisa de rede (interpretação e busca) é substituído por dublês; aqui se
testa o contrato do CLI e a montagem do prompt, não a qualidade do modelo.
"""

import io
import shutil
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import cli  # noqa: E402
import interpretation  # noqa: E402
import retrieval  # noqa: E402
from test_database import _isolate_data  # noqa: E402

SAMPLE_HITS = [
    {
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
    },
    {
        "book_id": "sun-tzu",
        "title": "A Arte da Guerra",
        "author": "Sun Tzu",
        "line": "default",
        "page_start": 12,
        "page_end": 13,
        "label_start": "12",
        "label_end": "13",
        "text": "conhece o inimigo e a ti mesmo",
        "score": 0.77,
    },
]


def test_parser_contract():
    """O parser exige metadados no add e tem os defaults do conectar."""
    parser = cli.build_parser()
    args = parser.parse_args(["conectar", "um trecho"])
    assert args.k == cli.DEFAULT_K and args.de is None and args.linha is None

    args = parser.parse_args(["add", "livro.pdf", "--titulo", "T", "--autor", "A"])
    assert args.linha == "default"

    for invalid in (["add", "livro.pdf"], ["add", "livro.pdf", "--titulo", "T"]):
        try:
            parser.parse_args(invalid)
        except SystemExit:
            continue
        raise AssertionError(f"esperava erro de argumentos para {invalid}")
    print("parser OK")


def test_listar_vazio():
    """Sem livros ingeridos, o listar avisa em vez de sair vazio."""
    out = io.StringIO()
    with redirect_stdout(out):
        code = cli.main(["listar"])
    assert code == 0 and "nenhum livro" in out.getvalue()
    print("listar vazio OK")


def test_conectar_sem_trechos():
    """Sem trechos recuperados, o guard aparece e nenhuma API é chamada."""
    original = retrieval.find_connections
    retrieval.find_connections = lambda *a, **k: []
    try:
        out = io.StringIO()
        with redirect_stdout(out):
            code = cli.main(["conectar", "texto sem conexão"])
        assert code == 0
        assert interpretation.NO_CONNECTION in out.getvalue()
    finally:
        retrieval.find_connections = original
    print("guard sem trechos OK")


def test_conectar_mostra_card_e_trechos():
    """Com trechos, o CLI mostra o card e depois os trechos para conferência."""
    original_retrieval = retrieval.find_connections
    original_interpret = interpretation.interpret
    retrieval.find_connections = lambda *a, **k: SAMPLE_HITS
    interpretation.interpret = lambda text, hits: "CARD DE TESTE"
    try:
        out = io.StringIO()
        with redirect_stdout(out):
            code = cli.main(["conectar", "texto", "--k", "2"])
        printed = out.getvalue()
    finally:
        retrieval.find_connections = original_retrieval
        interpretation.interpret = original_interpret

    assert code == 0
    assert "CARD DE TESTE" in printed
    assert "trechos recuperados" in printed
    assert "Maquiavel, O Principe, pág. 45" in printed
    assert "Sun Tzu, A Arte da Guerra, págs. 12-13" in printed
    print("card + trechos OK")


def test_conectar_mostra_trechos_quando_card_falha():
    """Se o card falhar, os trechos recuperados ainda aparecem (degradação visível)."""
    original_retrieval = retrieval.find_connections
    original_interpret = interpretation.interpret
    retrieval.find_connections = lambda *a, **k: SAMPLE_HITS

    def failing_interpret(text, hits):
        raise RuntimeError("modelo indisponível")

    interpretation.interpret = failing_interpret
    try:
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = cli.main(["conectar", "texto"])
        printed = out.getvalue()
    finally:
        retrieval.find_connections = original_retrieval
        interpretation.interpret = original_interpret

    assert code == 0
    assert "trechos recuperados" in printed
    assert "Maquiavel, O Principe, pág. 45" in printed
    assert "card indisponível" in err.getvalue()
    print("card falhou, trechos ainda exibidos OK")


def test_interpretation_prompt():
    """O prompt é grounded: inclui o texto do usuário e as fontes dos trechos."""
    prompt = interpretation.build_prompt("meu trecho selecionado", SAMPLE_HITS)
    assert "meu trecho selecionado" in prompt
    assert "Maquiavel, O Principe, pág. 45" in prompt
    assert "conhece o inimigo e a ti mesmo" in prompt
    assert "Nenhuma conexão relevante identificada" in prompt  # guard no prompt
    print("prompt grounded OK")


def test_interpretation_sem_trechos_nao_chama_api():
    """Sem trechos, interpret devolve o guard sem tocar na API."""
    assert interpretation.interpret("texto", []) == interpretation.NO_CONNECTION
    print("guard do interpret OK")


def main():
    """Roda os testes (o banco de teste fica num diretório temporário)."""
    tmp = _isolate_data()
    try:
        test_parser_contract()
        test_listar_vazio()
        test_conectar_sem_trechos()
        test_conectar_mostra_card_e_trechos()
        test_conectar_mostra_trechos_quando_card_falha()
        test_interpretation_prompt()
        test_interpretation_sem_trechos_nao_chama_api()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("TODOS OS TESTES PASSARAM")


if __name__ == "__main__":
    main()
