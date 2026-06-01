# Paper Conversion Notes

This directory contains manually cleaned Markdown notes derived from local PDF
and TeX-source ingestion.

Conversion path used here:

- PDFs were downloaded into `papers/pdf/`.
- arXiv TeX sources were downloaded and extracted into `papers/src/`.
- `pdftotext -layout` was used to create raw text sidecars for quick page
  inspection.
- The final Markdown files are hand-cleaned summaries/audits written against
  the TeX source and reference code rather than bulk auto-converted prose.

The cleaned files are intentionally not full-paper reproductions. They preserve
the algorithm, implementation-relevant equations, experimental settings, code
audit notes, and benchmark implications needed for this repository.
