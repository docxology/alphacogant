# Regenerate artifacts and render inspected manuscript releases

AlphaCOGANT is standalone. Never run another repository's template stages as a
local gate. Engine-only installation and use need no document toolchain.
Manuscript/visual releases require the contained PDF/HTML rendering below and
actual page/figure inspection. Rendering needs Pandoc, pandoc-crossref and
XeLaTeX; it uses Matplotlib's bundled fonts without installing host tools or fonts. The numbered body chapters include the registered graphical
abstract at the end of `00_abstract.md`, before the introduction. No extra cover
chapter is needed. Raw LaTeX clear-page blocks isolate it in PDF and are ignored
in HTML.

## Regenerate source-owned artifacts and check saved consistency

From the repository root:

```bash
uv sync --locked --extra dev
uv run --locked ruff check scripts src tests
uv run --locked pytest tests/ --cov=alphacogant --cov-fail-under=90 -q
uv run --locked python scripts/run_alphacogant_demo.py
uv run --locked python scripts/y_generate_figures.py
uv run --locked python scripts/z_generate_manuscript_variables.py
uv run --locked python scripts/z_generate_manuscript_variables.py --check
```

Generation writes current variables, registry, manifest and injected Markdown.
`--check` does not rewrite saved artifacts: changed tokens, registry, hashes,
undecodable registered PNGs or stale injected prose fail. This is structural
consistency, not proof of producer execution, empirical calibration or the
correctness of arbitrary literal prose.

## Required contained PDF and HTML rendering for manuscript releases

Write render metadata from the current manuscript configuration:

```bash
uv run --locked python - <<'PY'
import json
from pathlib import Path
import yaml
import matplotlib
c = yaml.safe_load(Path('docs/manuscript/config.yaml').read_text())
metadata = {'title': c['paper']['title'],
            'subtitle': c['paper']['subtitle'] + ' — v' + c['paper']['version'],
            'author': [a['name'] for a in c['authors']],
            'date': c['paper']['date'], 'link-citations': True}
font_dir = Path(matplotlib.get_data_path()) / 'fonts' / 'ttf'
def font_options(family, italic):
    return [f'Path={font_dir}/', 'Extension=.ttf',
            f'UprightFont={family}', f'BoldFont={family}-Bold',
            f'ItalicFont={family}-{italic}', f'BoldItalicFont={family}-Bold{italic}']
metadata.update(mainfont='DejaVuSerif', monofont='DejaVuSansMono',
                mainfontoptions=font_options('DejaVuSerif', 'Italic'),
                monofontoptions=font_options('DejaVuSansMono', 'Oblique'))
Path('output/manuscript/render_metadata.json').write_text(json.dumps(metadata))
PY
pandoc output/manuscript/[0-9][0-9]_*.md \
  --metadata-file=output/manuscript/render_metadata.json \
  --filter=pandoc-crossref --citeproc --number-sections \
  --bibliography=docs/manuscript/references.bib --resource-path=docs:output \
  --pdf-engine=xelatex -V documentclass=extarticle -V fontsize=9pt \
  -V geometry:margin=1.4cm -o output/manuscript/alphacogant.pdf
pandoc output/manuscript/[0-9][0-9]_*.md \
  --metadata-file=output/manuscript/render_metadata.json \
  --filter=pandoc-crossref --citeproc --number-sections \
  --bibliography=docs/manuscript/references.bib --resource-path=docs:output \
  --standalone --embed-resources --math-method=mathml -o output/manuscript/alphacogant.html
```

The resource path resolves the authored `../output/figures/` links inside this
checkout. Keep temporary/cache directories repository-local when running under
an inherited execution governor. For reproducibility, set `SOURCE_DATE_EPOCH`
to a recorded source timestamp and retain tool/runtime versions and raw logs.

## Inspect semantic content and actual rendered pages before publication

- Inspect renderer diagnostics for undefined citations/references, missing glyphs
  and layout overflow; compilation alone is not acceptance.
- Decode every figure and inspect axes, event-region signs, annotations and
  n/seed/concentration/horizon recipes. Inspect the registered graphical abstract
  and its provenance as well as the separate demo figure.
- Inspect actual PDF pages and HTML, including tables, equations, captions and
  the rendered bibliography. Check PDF structure with an available tool such as
  `qpdf --check output/manuscript/alphacogant.pdf`.
- Retain source/runtime/command receipts and output hashes. Regenerate twice when
  claiming deterministic artifacts, and state exactly which bytes matched.
- Historical corporate assertions remain attributed and unverified where the
  primary source is unavailable. A clean artifact manifest does not resolve that.

## Optional external-template integration

A separately configured, authorized template workflow may add its own cover,
bookends and publication-copy stages. Those are not local commands or standalone
acceptance requirements. Do not modify the template or its environment for this
project's release. Copied publication bytes, if used, need separate hash parity.
