# N-i — docx highlighting: the tooling question, answered by test

You asked me to flag immediately if the docx tooling cannot do reliable highlighting at this scale.
**It can, but not via python-docx.** I built the alternative and validated it against a real Word
document rather than asserting it would work.

## Why python-docx is the wrong tool here

python-docx exposes runs (`<w:r>`), and a sentence in Word is almost never one run. Word fragments
runs on spell-check state, revision marks, language attributes, formatting changes and sometimes
nothing at all. Highlighting "the run containing the sentence" therefore misses most real targets,
and highlighting each run that *overlaps* the sentence spills colour onto neighbouring text.

In this very document the manuscript title is stored as **three separate runs**. A run-level
approach would have highlighted a third of it, or a whole paragraph.

## What I built instead

`scripts/p13_docx_highlight.py` unzips the .docx, works on the **concatenated `<w:t>` text of each
paragraph**, locates each target by character offset, then **splits the runs at the match
boundaries** and inserts `<w:highlight w:val="yellow"/>` into the `<w:rPr>` of exactly the pieces
inside the match. It rezips preserving every other part unchanged.

Three properties are asserted in the code, not assumed:

1. **Visible text is byte-identical before and after.** If a single character changes, the tool
   aborts and prints the offset and both contexts. Highlighting must never edit prose.
2. **No target is silently skipped.** Every requested string is either highlighted or listed under
   NOT FOUND, and the exit status is non-zero if any is missing.
3. **A target occurring *n* times is highlighted *n* times**, and the per-target count is reported.

## Validation — run against a real Word document

Five targets, chosen to include the hard cases: a plain sentence; the manuscript title (known to
span runs); a mid-sentence fragment; a phrase occurring many times; and one string that does not
occur at all.

| Check | Result |
|---|---|
| targets requested | 5 |
| targets found | 4 (the fifth is the deliberate absentee) |
| absent target reported, not skipped | ✅ listed under NOT FOUND, exit status 1 |
| spans highlighted | 16 |
| **visible text unchanged** | ✅ 26,798 chars identical |
| zip integrity | ✅ |
| XML well-formed | ✅ |
| required parts present | ✅ `[Content_Types].xml`, `word/document.xml`, `_rels/.rels` |
| **title reassembled across its run split** | ✅ the highlighted fragments concatenate to the title exactly, character for character |
| **pre-existing highlights lost** | ✅ **0** — the document already had 11 yellow runs; splitting one of them preserved its colour on every character |
| characters newly highlighted | 329 |

The pre-existing-highlight check is the one that matters most for N-i: the marked manuscript will be
produced from a document that may already contain highlighting, and a tool that quietly drops it
would be worse than no tool.

Machine-readable record: `results/N-i_highlight_tool_validation.json`.

## What this means for N-i

The marked manuscript is now a **mechanical** step once the .docx arrives: assemble the target list
from `N4_cell_edits_DRAFT.md`, `N2_manuscript_prose_DRAFT.md` and `N-cde_manuscript_edits.md`, run
the tool, and read the NOT FOUND list — which doubles as a check that every edit I specified actually
matches text in the manuscript.

**Still blocked on the manuscript .docx.** Nothing else about N-i is.

### One limitation, stated plainly

The tool highlights **text that already exists**. Edits that *insert new* prose — the new
subsections, the DF40 section, the Code Availability paragraph — must be pasted in first, then
highlighted. The order is: apply all edits to the clean manuscript, save the marked copy from it,
then run the tool over the marked copy with the new text as targets.
