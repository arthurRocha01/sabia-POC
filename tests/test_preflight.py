"""Testes do preflight.py — offline: nenhuma chamada de rede é feita.

Rode com:  .venv/bin/python tests/test_preflight.py

Cobre as checagens locais (condições, custo, tema) e a sondagem real com
embeddings substituídos por vetores falsos.
"""

import shutil
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import database as db  # noqa: E402
import embeddings  # noqa: E402
import ingestion  # noqa: E402
import preflight  # noqa: E402
from test_database import (  # noqa: E402
    DOGS,
    _isolate_data,
    _maquiavel_chunks,
    _save_book,
    _sun_tzu_chunks,
)

SUN_TZU_PDF = PROJECT_ROOT / "resources" / "A-Arte-da-Guerra-de-Sun-Tzu.pdf"


def test_lexical_overlap():
    """Sobreposição alta para textos do mesmo tema, baixa para temas distintos."""
    guerra_a = (
        "estrategia militar exercito inimigo batalha territorio vitoria general "
        "tropas manobra cerco defesa ataque guerra"
    )
    guerra_b = (
        "estrategia militar exercito inimigo batalha terreno vitoria comandante "
        "tropas movimento defesa ataque guerra disciplina"
    )
    culinaria = "receita massa tomate manjericao forno cozinha tempero panela sabor"
    assert preflight.lexical_overlap(guerra_a, guerra_b) > 0.5
    assert preflight.lexical_overlap(guerra_a, culinaria) < 0.1
    print("sobreposicao lexical OK")


def test_analyze_good_pdf_with_empty_corpus():
    """PDF bom, acervo vazio: condições OK e aviso de que não há com o que conectar."""
    report = preflight.analyze_book(SUN_TZU_PDF, title="A Arte da Guerra", author="Sun Tzu")
    assert report["ok"], report["problems"]
    assert report["n_chunks"] > 20
    assert report["n_texts"] == report["n_chunks"]
    assert report["quality"]["letter_ratio"] > 0.6
    assert report["quality"]["control_ratio"] < 0.05
    assert report["quota_fraction"] < 1
    assert report["theme_overlap"] is None
    assert any("acervo vazio" in w for w in report["warnings"])
    print(f"analise de PDF bom OK ({report['n_chunks']} chunks)")


def test_analyze_with_corpus_reports_overlap():
    """Com acervo existente, o indício de tema é calculado."""
    _save_book("sun-tzu", "A Arte da Guerra", "Sun Tzu", _sun_tzu_chunks())
    report = preflight.analyze_book(SUN_TZU_PDF)
    assert report["ok"], report["problems"]
    assert report["theme_overlap"] is not None
    assert 0.0 <= report["theme_overlap"] <= 1.0
    print(f"indicio de tema com acervo OK ({report['theme_overlap']:.0%})")


def test_analyze_reports_problems_without_raising():
    """Problemas viram relatório, não exceção (é o ponto do preflight)."""
    missing = preflight.analyze_book(PROJECT_ROOT / "nao-existe.pdf")
    assert not missing["ok"] and any("não encontrado" in p for p in missing["problems"])

    # Arquivo real, formato não suportado (a checagem de formato só faz sentido
    # depois de o arquivo existir).
    tmp = Path(tempfile.mkdtemp(prefix="projeto-x-fmt-"))
    try:
        wrong = tmp / "livro.txt"
        wrong.write_text("texto qualquer", encoding="utf-8")
        wrong_format = preflight.analyze_book(wrong)
        assert not wrong_format["ok"] and any("formato" in p for p in wrong_format["problems"])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    empty_metadata = preflight.analyze_book(SUN_TZU_PDF, title="  ", author="Sun Tzu")
    assert not empty_metadata["ok"] and any("título vazio" in p for p in empty_metadata["problems"])
    print("problemas reportados sem excecao OK")


def test_quality_gate_surfaces_as_problem():
    """PDF com camada ilegível aparece como problema (e não estoura)."""
    original = ingestion.MIN_LETTER_RATIO
    ingestion.MIN_LETTER_RATIO = 0.99  # força a reprovação do PDF bom
    try:
        report = preflight.analyze_book(SUN_TZU_PDF)
    finally:
        ingestion.MIN_LETTER_RATIO = original
    assert not report["ok"]
    assert any("ilegível" in p for p in report["problems"])
    assert "n_chunks" not in report  # sem métricas quando a extração é recusada
    print("portao de qualidade vira problema OK")


def test_probe_connections_with_fake_embeddings():
    """Sondagem real (com embeddings falsos) encontra o que já está no acervo."""
    embeddings.embed_texts = lambda texts: [DOGS for _ in texts]
    hits, best = preflight.probe_book(SUN_TZU_PDF, samples=3, k=2)
    assert hits, "esperava achados do acervo"
    assert best is not None and best > 0.9
    assert {"query_page", "score", "citation", "text"} <= set(hits[0])
    # A ordenação é do mais próximo para o menos próximo.
    scores = [hit["score"] for hit in hits]
    assert scores == sorted(scores, reverse=True)
    print(f"sondagem real OK ({len(hits)} achados, melhor {best:.3f})")


def test_connection_verdict_thresholds():
    """O veredito traduz o score em leitura útil."""
    assert "fortes" in preflight.connection_verdict(0.80)
    assert "moderadas" in preflight.connection_verdict(0.65)
    assert "fracas" in preflight.connection_verdict(0.40)
    assert "sem dados" in preflight.connection_verdict(None)
    print("veredito de conexao OK")


def main():
    tmp = _isolate_data()
    try:
        test_lexical_overlap()
        test_analyze_good_pdf_with_empty_corpus()
        test_analyze_reports_problems_without_raising()
        test_quality_gate_surfaces_as_problem()
        test_analyze_with_corpus_reports_overlap()
        test_probe_connections_with_fake_embeddings()
        test_connection_verdict_thresholds()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("TODOS OS TESTES PASSARAM")


if __name__ == "__main__":
    main()
