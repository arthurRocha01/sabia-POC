"""Testes do ingestion.py — offline, sem API key e sem chamada de rede.

Rode com:  .venv/bin/python tests/test_ingestion.py

Usa os PDFs reais de resources/ para a extração (que não precisa de rede) e
substitui o embedding por vetores falsos ao testar a orquestração.
"""

import shutil
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pymupdf  # noqa: E402

import database as db  # noqa: E402
import embeddings  # noqa: E402
import ingestion  # noqa: E402
from test_database import _isolate_data  # noqa: E402

SUN_TZU_PDF = PROJECT_ROOT / "resources" / "A-Arte-da-Guerra-de-Sun-Tzu.pdf"
SCANNED_PDF = PROJECT_ROOT / "resources" / "Maquiavel_Nicolau_O_principe_1990.pdf"


def test_clean_page_text():
    """Limpeza: hífen quebrado, número de página na borda e quebras de linha."""
    raw = (
        "45\n"
        "o conheci-\n"
        "mento da guerra\n"
        "\n"
        "é essencial\n"
        "46\n"
    )
    cleaned = ingestion.clean_page_text(raw)
    assert "conhecimento" in cleaned, cleaned
    assert "conheci-" not in cleaned
    assert cleaned.split() == ["o", "conhecimento", "da", "guerra", "é", "essencial"], cleaned
    print("limpeza OK")


def test_extract_chunks_real_pdf():
    """Extração do PDF real: tamanho, páginas coerentes e sobreposição."""
    chunks = ingestion.extract_chunks(SUN_TZU_PDF)
    assert len(chunks) > 20, f"esperava muitos chunks, veio {len(chunks)}"

    for chunk in chunks:
        words = chunk["text"].split()
        assert words, "chunk vazio"
        assert len(words) <= ingestion.CHUNK_WORDS, f"chunk grande: {len(words)}"
        assert 1 <= chunk["page_start"] <= chunk["page_end"] <= 128, chunk
        assert chunk["label_start"] and chunk["label_end"], chunk

    # Sobreposição: o começo de um chunk repete o fim do anterior.
    previous = chunks[0]["text"].split()
    current = chunks[1]["text"].split()
    assert current[: ingestion.OVERLAP_WORDS] == previous[-ingestion.OVERLAP_WORDS:], (
        "a sobreposição entre chunks não confere"
    )
    print(f"extracao OK ({len(chunks)} chunks) + sobreposicao OK")


def test_labels_are_monotonic():
    """Os rótulos de página não podem andar para trás ao longo do livro."""
    chunks = ingestion.extract_chunks(SUN_TZU_PDF)
    numeric = [int(c["page_start"]) for c in chunks]
    assert numeric == sorted(numeric), "page_start fora de ordem"
    print("numeracao monotona OK")


def test_scanned_pdf_is_rejected():
    """PDF sem camada de texto (escaneado) tem que ser recusado com erro claro."""
    try:
        ingestion.extract_chunks(SCANNED_PDF)
    except ValueError as error:
        assert "sem camada de texto" in str(error), error
        print("PDF escaneado recusado OK")
        return
    raise AssertionError("esperava ValueError para PDF escaneado")


def test_blank_pdf_is_rejected():
    """PDF gerado sem texto também é recusado (não grava livro vazio)."""
    tmp = Path(tempfile.mkdtemp(prefix="projeto-x-pdf-"))
    try:
        path = tmp / "vazio.pdf"
        doc = pymupdf.open()
        doc.new_page()
        doc.save(path)
        doc.close()
        try:
            ingestion.extract_chunks(path)
        except ValueError as error:
            assert "sem camada de texto" in str(error), error
            print("PDF vazio recusado OK")
            return
        raise AssertionError("esperava ValueError para PDF sem texto")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_validation():
    """Metadados incompletos e arquivo inválido são recusados antes de gravar."""
    cases = [
        (SUN_TZU_PDF, "", "Sun Tzu"),
        (SUN_TZU_PDF, "A Arte da Guerra", ""),
        (PROJECT_ROOT / "nao-existe.pdf", "X", "Y"),
        (SUN_TZU_PDF.with_suffix(".txt"), "X", "Y"),
    ]
    for path, title, author in cases:
        try:
            ingestion.validate_book_file(path, title=title, author=author)
        except ValueError:
            continue
        raise AssertionError(f"esperava ValueError para {path}, {title!r}, {author!r}")
    # Caso válido passa
    assert ingestion.validate_book_file(SUN_TZU_PDF, title="A Arte da Guerra", author="Sun Tzu")
    print("validacao OK")


def test_slug_and_book_id():
    """Slug e id: estável para o mesmo arquivo, sufixado para título repetido."""
    assert ingestion.slugify("A Arte da Guerra") == "a-arte-da-guerra"
    assert ingestion.slugify("O Príncipe") == "o-principe"

    # Mesmo arquivo, mesmo título -> mesmo id (substitui).
    first = ingestion.make_book_id("A Arte da Guerra", SUN_TZU_PDF)
    assert first == "a-arte-da-guerra"
    db.save_book(
        book_id=first, title="A Arte da Guerra", author="Sun Tzu",
        source_path=str(SUN_TZU_PDF),
        chunks=[{"text": "x", "page_start": 1, "page_end": 1, "embedding": [1.0, 0.0, 0.0]}],
    )
    assert ingestion.make_book_id("A Arte da Guerra", SUN_TZU_PDF) == first

    # Outro arquivo com o mesmo título -> id sufixado, não sobrescreve.
    other = ingestion.make_book_id("A Arte da Guerra", PROJECT_ROOT / "outro.pdf")
    assert other == "a-arte-da-guerra-2", other
    print("slug e book_id OK")


def test_ingest_book_with_fake_embeddings():
    """Orquestração ponta a ponta sem rede: extrai, embeda (fake) e grava."""
    # Substitui o Gemini por vetores falsos e determinísticos.
    embeddings.embed_texts = lambda texts: [[float(len(t) % 7), 1.0, 0.0] for t in texts]

    entry = ingestion.ingest_book(
        SUN_TZU_PDF, title="A Arte da Guerra", author="Sun Tzu"
    )
    assert entry["book_id"] == "a-arte-da-guerra"
    assert entry["n_chunks"] > 20
    stored = db.get_book("a-arte-da-guerra")
    assert stored is not None and stored["n_chunks"] == entry["n_chunks"]

    hits = db.search([1.0, 1.0, 0.0], k=2)
    assert len(hits) == 2
    assert hits[0]["label_start"] and hits[0]["page_start"] >= 1

    # Re-ingerir o mesmo arquivo substitui, não duplica.
    entry2 = ingestion.ingest_book(
        SUN_TZU_PDF, title="A Arte da Guerra", author="Sun Tzu"
    )
    assert len(db.list_books()) == 1
    assert entry2["n_chunks"] == entry["n_chunks"]
    print("ingestao ponta a ponta OK (embedding falso)")


def main():
    """Roda todos os testes com banco isolado."""
    tmp = _isolate_data()
    try:
        test_clean_page_text()
        test_extract_chunks_real_pdf()
        test_labels_are_monotonic()
        test_scanned_pdf_is_rejected()
        test_blank_pdf_is_rejected()
        test_validation()
        test_slug_and_book_id()
        test_ingest_book_with_fake_embeddings()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("TODOS OS TESTES PASSARAM")


if __name__ == "__main__":
    main()
