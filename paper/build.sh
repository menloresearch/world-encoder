#!/usr/bin/env bash
# Build the paper and produce an arXiv-ready source zip.
#
# Why this exists: arXiv does not reliably run BibTeX, and main.bbl is a
# gitignored build artifact, so a plain "zip paper/" ships without a compiled
# bibliography and every \cite renders as "undefined". This script runs the
# full latex->bibtex->latex->latex cycle to regenerate main.bbl, then zips the
# exact source set arXiv needs (tex + bbl + bib + sty + figures) into
# arxiv_submission.zip. Re-run it any time citations or figures change.
#
# Usage:  cd paper && ./build.sh
set -euo pipefail
cd "$(dirname "$0")"

echo "== pdflatex (pass 1) =="
pdflatex -interaction=nonstopmode main.tex >/dev/null
echo "== bibtex =="
bibtex main
echo "== pdflatex (pass 2) =="
pdflatex -interaction=nonstopmode main.tex >/dev/null
echo "== pdflatex (pass 3) =="
pdflatex -interaction=nonstopmode main.tex >/dev/null

# Fail loudly if any citation is still unresolved.
if grep -q "Citation.*undefined" main.log; then
  echo "ERROR: undefined citations remain -- check references.bib" >&2
  grep "Citation.*undefined" main.log | head >&2
  exit 1
fi

echo "== zipping arXiv source =="
rm -f arxiv_submission.zip
zip -r arxiv_submission.zip \
  main.tex main.bbl references.bib neurips_2024.sty figures/ \
  -x '*.mmd' >/dev/null

echo
echo "Done. Upload arxiv_submission.zip to arXiv."
echo "Pages: $(pdfinfo main.pdf 2>/dev/null | awk '/^Pages:/{print $2}')  |  Undefined citations: 0"
unzip -l arxiv_submission.zip | tail -n +4 | head -n -2
