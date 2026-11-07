#!/usr/bin/env bash
# Build report/report.pdf.
#
# Two routes, in order of preference:
#   1. pdflatex, if a TeX distribution is installed
#   2. tectonic, a single static binary that fetches its own package bundle and needs no admin rights
#
# Called by `make report`.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO/report" || exit 1

if command -v pdflatex >/dev/null 2>&1; then
  echo "using pdflatex"
  pdflatex -interaction=nonstopmode report.tex >/tmp/tex1.out 2>&1
  pdflatex -interaction=nonstopmode report.tex >/tmp/tex2.out 2>&1   # second pass for pgfplots
  rc=$?
  grep -E "^!" /tmp/tex2.out | head -5
else
  if [ ! -x /tmp/tectonic ]; then
    echo "no pdflatex found; fetching tectonic (no admin rights needed)"
    ok=0
    for url in \
      "https://github.com/tectonic-typesetting/tectonic/releases/download/tectonic%400.15.0/tectonic-0.15.0-x86_64-unknown-linux-musl.tar.gz" \
      "https://github.com/tectonic-typesetting/tectonic/releases/download/tectonic%400.14.1/tectonic-0.14.1-x86_64-unknown-linux-musl.tar.gz"
    do
      if curl -fsSL --max-time 300 -o /tmp/tectonic.tar.gz "$url"; then ok=1; break; fi
    done
    if [ "$ok" -eq 0 ]; then echo "could not fetch tectonic" >&2; exit 1; fi
    tar xzf /tmp/tectonic.tar.gz -C /tmp && chmod +x /tmp/tectonic
  fi
  /tmp/tectonic --keep-logs report.tex >/tmp/tex.out 2>&1
  rc=$?
  grep -Ei "^error" /tmp/tex.out | head -5
fi

echo "exit: $rc"
if [ -f report.pdf ]; then
  echo "report/report.pdf: $(du -h report.pdf | cut -f1)"
else
  echo "report.pdf was not produced" >&2
  exit 1
fi
