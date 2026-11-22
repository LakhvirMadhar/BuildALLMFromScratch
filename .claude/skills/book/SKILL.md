---
name: book
description: Read and explain pages of "Build a Large Language Model (From Scratch)". Use when the user runs /book <page>, or asks to look at, read, or explain a page or page range of the book ("look at page 253", "help me understand pages 88-90").
argument-hint: <page or range, e.g. 253 or 88-90>
---

# Read a page of the book

The user wants help understanding page(s) `$ARGUMENTS` of the book.

## 1. Open the page images

Every page is pre-rendered as a PNG at `.cache/book_pages/page_<N>.png`, where `<N>` is the page number **printed in the book** (not the PDF index). Front matter uses its roman numeral (`page_xii.png`).

- Open each requested page with the Read tool. For a range like `88-90`, open `page_88.png`, `page_89.png`, `page_90.png`.
- If a file is missing (e.g. `.cache/` was deleted), run `poetry run python tools/book_page.py all` once to regenerate every page, then open them.
- Never Read the PDF itself — on Windows that fails with "pdftoppm is not installed".

## 2. Write the explanation into the lesson doc — not the chat

The user studies from `.md` files, not from chat. The explanation goes in `docs/lessons/`.

- **Find the lesson for these pages.** The page header shows the section number (e.g. "A.2", "3.5"). Lesson docs mirror the user's code: `src/build_llm_from_scratch/chapter_NN_topic/NN_topic.py` ↔ `docs/lessons/chapter_NN_topic/lesson_NN_topic.md` (Appendix A is `chapter_00`). Check `docs/lessons/00-TOC.md` and the user's `src/` folder to find the lesson that covers this section.
- **If the lesson doc exists, add to it.** Put the new material in the right place in its Concepts section, in book order. Don't duplicate what's already there — extend it.
- **If the user has a code file for this section but no doc yet, create the doc** with the matching name, using the Lesson Docs Format in `AGENTS.md` (What We're Doing, Why / Context, Concepts with "Try it" tasks, Your Task), and add its row to `docs/lessons/00-TOC.md`.
- **If there's no code file for this section yet, ask the user** what they're naming it before creating the doc — they name the code files, and docs follow.
- **Large ranges (a whole section or appendix):** start the doc with a short reading plan — which sub-sections to read closely, skim, or skip, and why.

Follow all the tutor rules in `AGENTS.md`. For book pages specifically:

- **Say which section the pages belong to** and what part of the LLM pipeline it's about.
- **Walk through figures explicitly** - figures carry much of this book's meaning. Describe what each box, arrow, and label shows, in order, and name the figure and page.
- **Decode every code listing and formula** before explaining what it accomplishes: each symbol, method, and argument. State tensor shapes and what each dimension means.
- **Anchor to plain Python** - what the same idea looks like with lists and loops.
- **Trace jargon to its origin** if a term is opaque (see "Etymology for Unfamiliar Terms" in `AGENTS.md`).
- **Flag sentences that run onto the next page** rather than guessing how they end.

Don't write the book's code into the user's `src/` files — explaining it is the job. Short illustrative snippets inside the lesson doc are fine.

## 3. Close in chat — briefly

In chat, say only:

- which lesson doc was written or updated, as a clickable link
- one line on what was added
- an offer of the next pages, and the question: did any term on these pages feel like magic?
