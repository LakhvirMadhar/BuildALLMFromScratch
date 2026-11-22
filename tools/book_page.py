"""Render pages of the book PDF to PNG images so Claude can read them.

Usage:
    poetry run python tools/book_page.py 253
    poetry run python tools/book_page.py 253-256
    poetry run python tools/book_page.py all

Page numbers are the ones printed in the book, not the PDF viewer's index.
Roman-numeral front-matter pages work too (e.g. "xii"). Rendered pages are
cached in .cache/book_pages/page_<label>.png and reused on later calls.
"""

import argparse
import re
import sys
from pathlib import Path

import pymupdf

REPO_ROOT = Path(__file__).resolve().parent.parent
BOOK_PDF = REPO_ROOT.parent / "LLMs-from-scratch-main" / "Build_a_Large_Language_Model_(From_Scrat.pdf"
OUTPUT_DIR = REPO_ROOT / ".cache" / "book_pages"
RENDER_DPI = 130
MAX_PAGES_PER_CALL = 20
ALL_PAGES = "all"


class PageNotFoundError(Exception):
    def __init__(self, page_label: str) -> None:
        self.page_label = page_label
        super().__init__(f"No page labelled {page_label!r} in the book")


def parse_page_labels(page_spec: str) -> list[str]:
    """Turn "253" into ["253"] and "253-256" into ["253", ..., "256"]."""
    if "-" not in page_spec:
        return [page_spec]

    first, last = (int(part) for part in page_spec.split("-", maxsplit=1))
    if last < first:
        raise ValueError(f"Range {page_spec!r} ends before it starts")
    if last - first + 1 > MAX_PAGES_PER_CALL:
        raise ValueError(f"Range {page_spec!r} is over {MAX_PAGES_PER_CALL} pages")
    return [str(number) for number in range(first, last + 1)]


def build_output_path(page_label: str) -> Path:
    """Map a printed label to its PNG path, making odd labels (like the cover's title) filename-safe."""
    safe_label = re.sub(r"[^A-Za-z0-9]+", "_", page_label).strip("_")
    return OUTPUT_DIR / f"page_{safe_label}.png"


def render_page(book: pymupdf.Document, page_index: int, page_label: str) -> Path:
    """Save one page as a PNG, skipping it if it's already cached, and return its path."""
    output_path = build_output_path(page_label)
    if not output_path.exists():
        book[page_index].get_pixmap(dpi=RENDER_DPI).save(output_path)
    return output_path


def render_labelled_page(book: pymupdf.Document, page_label: str) -> Path:
    matching_indexes = book.get_page_numbers(page_label)
    if not matching_indexes:
        raise PageNotFoundError(page_label)
    return render_page(book, matching_indexes[0], page_label)


def render_all_pages(book: pymupdf.Document) -> int:
    """Render every page in the book and return how many there were."""
    page_count: int = book.page_count
    used_labels: set[str] = set()
    for page_index in range(page_count):
        page_label = book[page_index].get_label() or f"pdf_{page_index + 1}"
        # The publisher's back-matter ads all share the label "Promo page"
        if page_label in used_labels:
            page_label = f"{page_label}_pdf_{page_index + 1}"
        used_labels.add(page_label)
        render_page(book, page_index, page_label)
    return page_count


def main() -> None:
    parser = argparse.ArgumentParser(description="Render book pages to PNG for Claude to read.")
    parser.add_argument("pages", help=f'A printed page number like "253", a range like "253-256", or "{ALL_PAGES}"')
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        with pymupdf.open(BOOK_PDF) as book:
            if args.pages == ALL_PAGES:
                page_count = render_all_pages(book)
                print(f"Rendered {page_count} pages to {OUTPUT_DIR}")
                return

            for page_label in parse_page_labels(args.pages):
                print(render_labelled_page(book, page_label))
    except (ValueError, PageNotFoundError) as error:
        sys.exit(str(error))


if __name__ == "__main__":
    main()
