"""Ingestão de livros: PDF -> chunks com página (índice) e numeração do livro.

Etapas, nesta ordem:
  1. Validação dos metadados e do arquivo (nada é gravado antes de validar).
  2. Extração do texto, página por página, com limpeza.
  3. Determinação da numeração do livro (rótulo) — ANTES da limpeza descartar
     a linha do número de página.
  4. Chunking em janela deslizante com sobreposição.
  5. Orquestração: embeddings + gravação no banco vetorial e no catálogo.

Fronteira importante: nenhuma função daqui chama a API de embeddings, exceto
ingest_book(). Assim, extração/limpeza/chunking são testáveis sem chave.

Numeração de página: cada chunk guarda as DUAS coisas — page_start/page_end
(índice posicional, 1..N, único, serve para localizar a página no PDF) e
label_start/label_end (a numeração do próprio livro, serve para citar).
"""

from __future__ import annotations

import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import pymupdf

import database
import embeddings

CHUNK_WORDS = 120      # tamanho do chunk, em palavras (menor = menos diluição)
OVERLAP_WORDS = 30     # sobreposição de 25% (chunks pequenos pedem mais continuidade)
EDGE_LINES = 3         # linhas de borda onde vive cabeçalho/rodapé
BOILERPLATE_RATIO = 0.5  # linha em >50% das páginas é cabeçalho/rodapé
BOILERPLATE_MAX_CHARS = 60
MIN_PRINTED_COVERAGE = 0.5  # cobertura mínima para aceitar números impressos

# Linha que é só número (arábico) ou numeral romano: candidata a número de página.
_PAGE_NUMBER_RE = re.compile(r"^(?:\d{1,4}|[ivxlcdm]{1,7})$", re.IGNORECASE)
# Palavra quebrada com hífen na virada de linha: "conheci-\nmento" -> "conhecimento".
_HYPHEN_BREAK_RE = re.compile(r"(\w)-\s*\n\s*(\w)")


# ---------------------------------------------------------------------------
# Limpeza de texto
# ---------------------------------------------------------------------------

def _page_lines(raw: str) -> list[str]:
    """Recompõe hífens quebrados e devolve as linhas não vazias já com strip."""
    return [line.strip() for line in _HYPHEN_BREAK_RE.sub(r"\1\2", raw).splitlines()
            if line.strip()]


def clean_page_text(raw: str) -> str:
    """Limpa o texto de UMA página e devolve um fluxo contínuo de palavras.

    Remove o número de página impresso nas bordas (o rótulo já foi capturado
    antes) e colapsa as quebras de linha, que no PDF não significam fim de
    frase. É esta string que vai virar vetor, então aqui é onde ruído custa
    qualidade de busca.
    """
    lines = _page_lines(raw)
    edge = set(range(min(EDGE_LINES, len(lines))))
    edge |= set(range(max(0, len(lines) - EDGE_LINES), len(lines)))
    kept = [
        line for i, line in enumerate(lines)
        if not (i in edge and _PAGE_NUMBER_RE.match(line))
    ]
    return " ".join(kept)


def _drop_repeated_lines(pages: list[list[str]]) -> list[list[str]]:
    """Remove cabeçalho/rodapé que se repete em mais da metade das páginas."""
    if len(pages) < 4:
        return pages
    counter: Counter[str] = Counter()
    for lines in pages:
        counter.update(set(lines))  # conta no máximo 1x por página
    limit = len(pages) * BOILERPLATE_RATIO
    boilerplate = {
        line for line, count in counter.items()
        if count > limit and len(line) < BOILERPLATE_MAX_CHARS
    }
    if not boilerplate:
        return pages
    return [[line for line in lines if line not in boilerplate] for lines in pages]


# ---------------------------------------------------------------------------
# Numeração das páginas (rótulo do livro)
# ---------------------------------------------------------------------------

def _page_text(page: pymupdf.Page) -> str:
    """Texto de uma página (get_text tem sobrecargas ambíguas nos stubs)."""
    return str(page.get_text("text"))


def _detect_printed_labels(doc: pymupdf.Document) -> list[str] | None:
    """Tenta ler o número impresso no topo/rodapé de cada página.

    Retorna os rótulos por página, ou None se a detecção não for confiável
    (pouca cobertura ou sequência não monotônica — sinal de leitura errada).
    Páginas sem número detectado ficam com rótulo vazio.
    """
    candidates: list[str] = []
    for i in range(doc.page_count):
        lines = _page_lines(_page_text(doc[i]))
        edges = lines[:EDGE_LINES] + lines[-EDGE_LINES:]
        inferred = next((line for line in edges if _PAGE_NUMBER_RE.match(line)), "")
        candidates.append(inferred)

    numeric = [c for c in candidates if c.isdigit()]
    if len(numeric) < len(candidates) * MIN_PRINTED_COVERAGE:
        return None
    values = [int(c) for c in numeric]
    if any(later < earlier for earlier, later in zip(values, values[1:])):
        return None  # andou para trás: a detecção pegou a coisa errada
    return candidates


def resolve_page_labels(doc: pymupdf.Document) -> tuple[list[str], str]:
    """Descobre a numeração do livro.

    Ordem de preferência: PageLabels do próprio PDF -> número impresso nas
    páginas -> índice posicional (fallback, marcado como tal). Retorna
    (rótulos por página, origem: "pdf" | "printed" | "positional").
    """
    labels = [doc[i].get_label() or "" for i in range(doc.page_count)]
    if any(labels):
        # Página sem rótulo (ex.: capa) usa o índice posicional como fallback.
        return [label or str(i + 1) for i, label in enumerate(labels)], "pdf"

    printed = _detect_printed_labels(doc)
    if printed is not None:
        return printed, "printed"

    return [str(i + 1) for i in range(doc.page_count)], "positional"


# ---------------------------------------------------------------------------
# Chunks
# ---------------------------------------------------------------------------

def _build_chunks(words: list[tuple[str, int]], labels: list[str]) -> list[dict]:
    """Janela deslizante sobre as palavras; cada chunk sabe de quais páginas vem.

    words: lista de (palavra, índice de página 0-based).
    """
    step = CHUNK_WORDS - OVERLAP_WORDS
    chunks: list[dict] = []
    for start in range(0, len(words), step):
        window = words[start:start + CHUNK_WORDS]
        if not window:
            break
        first_page = window[0][1]
        last_page = window[-1][1]
        chunks.append({
            "text": " ".join(word for word, _ in window),
            "page_start": first_page + 1,
            "page_end": last_page + 1,
            "label_start": labels[first_page] or str(first_page + 1),
            "label_end": labels[last_page] or str(last_page + 1),
        })
        if start + CHUNK_WORDS >= len(words):
            break  # a última janela já alcançou o fim do livro
    return chunks


def extract_chunks(pdf_path: str | Path) -> list[dict]:
    """Extrai os chunks do PDF (sem rede, sem API key).

    Cada chunk: text, page_start, page_end, label_start, label_end.
    Levanta ValueError se o PDF não tiver camada de texto (escaneado).
    """
    doc = pymupdf.open(str(pdf_path))
    try:
        pages = [_page_lines(_page_text(doc[i])) for i in range(doc.page_count)]
        pages = _drop_repeated_lines(pages)
        _labels, source = resolve_page_labels(doc)
        labels = _labels
        page_texts = [clean_page_text("\n".join(lines)) for lines in pages]
    finally:
        doc.close()

    if source == "positional":
        print(
            "aviso: numeração do livro não recuperada (sem PageLabels nem "
            "número impresso) — citando páginas do PDF pelo índice posicional",
            file=sys.stderr,
        )

    words = [
        (word, page)
        for page, text in enumerate(page_texts)
        for word in text.split()
    ]
    if not words:
        raise ValueError(
            f"{pdf_path}: PDF sem camada de texto (escaneado?) — "
            "o POC não faz OCR"
        )

    return _build_chunks(words, labels)


# ---------------------------------------------------------------------------
# Validação e identidade do livro
# ---------------------------------------------------------------------------

def validate_book_file(pdf_path: str | Path, *, title: str, author: str) -> Path:
    """Valida metadados obrigatórios e o arquivo antes de qualquer efeito colateral.

    A ficha do catálogo não pode sair incompleta: sem título e autor a conexão
    entre autores — que é a proposta do produto — não se sustenta.
    """
    if not title or not title.strip():
        raise ValueError("título é obrigatório para ingerir (a ficha não pode sair incompleta)")
    if not author or not author.strip():
        raise ValueError("autor é obrigatório para ingerir (a ficha não pode sair incompleta)")

    # Sempre absoluto: o caminho é a identidade do arquivo para a re-ingestão
    # (um caminho relativo digitado em outro diretório criaria um livro duplicado).
    path = Path(pdf_path).resolve()
    if not path.is_file():
        raise ValueError(f"arquivo não encontrado: {path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"formato ainda não suportado (só PDF): {path.name}")
    return path


def slugify(text: str) -> str:
    """'A Arte da Guerra' -> 'a-arte-da-guerra' (id legível para a CLI)."""
    ascii_only = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", ascii_only.lower()).strip("-") or "livro"


def make_book_id(title: str, source_path: str | Path) -> str:
    """Id estável do livro: slug do título.

    Re-ingestão do MESMO arquivo devolve o mesmo id (substitui). Títulos iguais
    de arquivos diferentes ganham sufixo -2, -3... para não se sobrescreverem.
    """
    base = slugify(title)
    target = Path(source_path).resolve()
    candidate, suffix = base, 1
    while database.book_exists(candidate):
        existing = database.get_book(candidate)
        if existing and Path(existing["source_path"]).resolve() == target:
            return candidate
        suffix += 1
        candidate = f"{base}-{suffix}"
    return candidate


# ---------------------------------------------------------------------------
# Orquestração
# ---------------------------------------------------------------------------

def ingest_book(
    pdf_path: str | Path,
    *,
    title: str,
    author: str,
    line: str = "default",
) -> dict:
    """Ingere um livro: valida, extrai, embeda e grava. Devolve a ficha do catálogo.

    A ordem existe para nunca deixar livro meio ingerido: validação e extração
    acontecem antes de qualquer gravação (e antes de gastar chamada de API).
    """
    path = validate_book_file(pdf_path, title=title, author=author)
    chunks = extract_chunks(path)

    vectors = embeddings.embed_texts([chunk["text"] for chunk in chunks])
    if len(vectors) != len(chunks):
        # Guarda contra o desalinhamento silencioso vetor<->chunk.
        raise RuntimeError(
            f"embeddings desalinhados: {len(vectors)} vetores para {len(chunks)} chunks"
        )
    for chunk, vector in zip(chunks, vectors):
        chunk["embedding"] = vector

    book_id = make_book_id(title, path)
    return database.save_book(
        book_id=book_id,
        title=title,
        author=author,
        source_path=str(path),
        line=line,
        chunks=chunks,
    )
