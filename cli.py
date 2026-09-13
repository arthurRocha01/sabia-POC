"""CLI do POC.

  python cli.py add <caminho.pdf> --titulo "..." --autor "..."
  python cli.py conectar "<texto>" [--de <livro_id>] [--k 3] [--linha default]
  python cli.py listar

Os nomes dos comandos e das opções são interface com o usuário (português);
o código por dentro é em inglês, conforme a convenção do projeto.
"""

from __future__ import annotations

import argparse
import sys

import database
import ingestion
import interpretation
import retrieval

DEFAULT_K = 3


def cmd_add(args: argparse.Namespace) -> int:
    """Ingere um livro (PDF) e registra no catálogo."""
    entry = ingestion.ingest_book(
        args.caminho,
        title=args.titulo,
        author=args.autor,
        line=args.linha,
    )
    print(
        f"ingerido: {entry['book_id']} — {entry['title']} ({entry['author']}), "
        f"{entry['n_chunks']} chunks"
    )
    return 0


def cmd_listar(args: argparse.Namespace) -> int:
    """Lista os livros já ingeridos."""
    books = database.list_books()
    if not books:
        print("nenhum livro ingerido ainda")
        return 0
    for book in books:
        print(
            f"{book['book_id']:<24} {book['n_chunks']:>4} chunks  "
            f"{book['title']} ({book['author']})"
        )
    return 0


def cmd_conectar(args: argparse.Namespace) -> int:
    """Busca conexões em outros livros e apresenta o card de insights."""
    hits = retrieval.find_connections(
        args.texto,
        k=args.k,
        exclude_book_id=args.de,
        line=args.linha,
    )

    if not hits:
        print(interpretation.NO_CONNECTION)
        return 0

    print("=== card de insights ===")
    try:
        print(interpretation.interpret(args.texto, hits))
    except RuntimeError as error:
        # A busca já deu certo: perder só o card não pode esconder as conexões.
        print(f"(card indisponível: {error})", file=sys.stderr)

    # Os trechos crus ficam visíveis para conferência do POC (a citação do card
    # vem daqui; sem isso não dá para auditar a resposta do modelo).
    print("\n=== trechos recuperados ===")
    for hit in hits:
        citation = interpretation.format_citation(hit)
        preview = " ".join(hit["text"].split())[:140]
        print(f"[{hit['score']:.3f}] {citation}\n    {preview}...")
    return 0


EPILOG = """\
exemplos:
  python cli.py add resources/livro.pdf --titulo "Título" --autor "Autor"
  python cli.py listar
  python cli.py conectar "conhecer o inimigo e a si mesmo" --de a-arte-da-guerra --k 3
  python cli.py conectar "o que e a virtu?" --k 5        # consulta livre (Modo II)

detalhes de um comando:  python cli.py <COMANDO> --help
requer GOOGLE_API_KEY no arquivo .env (veja .env.example).\
"""


def build_parser() -> argparse.ArgumentParser:
    """Monta o parser com os três comandos do POC e ajuda detalhada."""
    parser = argparse.ArgumentParser(
        prog="cli.py",
        description=(
            "POC do Projeto X: conexões entre livros de uma mesma linha de\n"
            "aprendizado. Ingere livros, busca trechos de OUTROS autores e\n"
            "apresenta um card de insights (síntese + relação + fontes)."
        ),
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="comando", required=True, metavar="COMANDO")

    add = subparsers.add_parser(
        "add",
        help="ingere um livro PDF no acervo",
        description=(
            "Extrai o texto do PDF, fatia em chunks, gera os embeddings e grava\n"
            "no banco local. Título e autor são obrigatórios: a ficha do catálogo\n"
            "não pode sair incompleta, e a conexão só faz sentido entre autores."
        ),
        epilog='exemplo:\n  python cli.py add resources/livro.pdf --titulo "A Arte da Guerra" --autor "Sun Tzu"',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add.add_argument("caminho", help="caminho do arquivo PDF a ingerir")
    add.add_argument("--titulo", required=True, help="título do livro (obrigatório)")
    add.add_argument("--autor", required=True, help="autor do livro (obrigatório)")
    add.add_argument("--linha", default="default", help="linha de aprendizado (default: default)")
    add.set_defaults(func=cmd_add)

    conectar = subparsers.add_parser(
        "conectar",
        help="busca conexões e gera o card de insights",
        description=(
            "Recebe o trecho selecionado (ou uma pergunta) e devolve as conexões\n"
            "encontradas em outros livros, já interpretadas pelo modelo. Use --de\n"
            "para excluir o livro de origem: sem isso a conexão mais próxima tende\n"
            "a ser o próprio texto lido."
        ),
        epilog=(
            "exemplos:\n"
            '  python cli.py conectar "conhecer o inimigo e a si mesmo" --de a-arte-da-guerra\n'
            '  python cli.py conectar "o que e a virtu?" --k 5'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    conectar.add_argument("texto", help="trecho selecionado ou pergunta de partida")
    conectar.add_argument(
        "--de", default=None, metavar="LIVRO_ID",
        help="livro de origem a EXCLUIR da busca (o que você está lendo)",
    )
    conectar.add_argument(
        "--k", type=int, default=DEFAULT_K,
        help=f"quantas conexões recuperar (default: {DEFAULT_K})",
    )
    conectar.add_argument(
        "--linha", default=None, metavar="LINHA",
        help="restringe a busca a uma linha de aprendizado",
    )
    conectar.set_defaults(func=cmd_conectar)

    listar = subparsers.add_parser(
        "listar",
        help="lista os livros ingeridos",
        description="Mostra os livros do acervo: id, quantidade de chunks, título e autor.",
        epilog="exemplo:\n  python cli.py listar",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    listar.set_defaults(func=cmd_listar)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Ponto de entrada: erros esperados saem como mensagem, não traceback."""
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except ValueError as error:
        print(f"erro: {error}", file=sys.stderr)
        return 2
    except RuntimeError as error:
        print(f"erro: {error}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
