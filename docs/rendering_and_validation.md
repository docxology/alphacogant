# Render source-checked PDF/HTML and inspect manuscript evidence

AlphaCOGANT's engine runs with the locked Python environment. For
manuscript and visual releases, use the contained PDF/HTML rendering below and
actual page/figure inspection. Rendering needs existing Pandoc 3.13+,
compatible pandoc-crossref, XeLaTeX and qpdf on POSIX. The producer uses
Matplotlib's bundled fonts; it installs no host tools or fonts. The numbered
body chapters include the registered graphical
abstract at the end of `00_abstract.md`, before the introduction. No extra cover
chapter is needed. Raw LaTeX clear-page blocks isolate it in PDF and are ignored
in HTML.

## Regenerate source-owned artifacts and check saved consistency

From the repository root:

```bash
uv sync --locked --extra dev
uv run --locked ruff check scripts src tests
uv run --locked python scripts/run_alphacogant_demo.py
uv run --locked python scripts/y_generate_figures.py
uv run --locked python scripts/z_generate_manuscript_variables.py
uv run --locked python scripts/z_generate_manuscript_variables.py --check
uv run --locked pytest tests/ --cov=alphacogant --cov-fail-under=90 -q
```

Generation writes current variables, registry, manifest and injected Markdown.
`--check` does not rewrite saved artifacts: changed tokens, registry, hashes,
undecodable registered PNGs or stale injected prose fail. This is structural
consistency, not proof of producer execution, empirical calibration or the
correctness of arbitrary literal prose.

## Produce a fresh contained bundle from checked current sources

After regeneration, run the source-owned thin orchestrator:

```bash
uv run --locked python scripts/render_manuscript.py \
  --destination output/manuscript-release \
  --source-date-epoch "$(git log -1 --format=%ct)"
```

Choose a new destination for every run. Existing directories, lexical output
escapes and symlink components below the project root are refused. Only current
authored numbered chapters are selected, never orphan injected chapters left by
deleted sources. The preflight requires matching package/source versions and
Python-source hashes, current computed tokens, registry, manifest and injected text.

The bundle contains `alphacogant.pdf`, self-contained MathML
`alphacogant.html`, metadata, raw tool diagnostics and (only after success)
`render.json`. The receipt binds consumed source/artifact hashes, the explicit
epoch, Python/library versions, tool binary hashes/versions and bundled-font
hashes. Each HTML figure label must retain its registered PNG bytes and nonempty alt text;
internal references/figure anchors must resolve uniquely. qpdf checks PDF
structure. Emitted unresolved citation/reference or missing-glyph diagnostics
fail; layout diagnostics remain for human review. Failed partial bundles are
retained without a successful receipt.

Temporary/cache paths stay below the bundle; native tools run without shell
escape and with a finite per-tool bound and owned process-group cleanup.
This is a trusted local producer, not an OS sandbox, a task service or an agent
method. Follow inherited admission when a checkout is governed. Source paths
must remain stable; cooperative path checks do not defeat hostile same-UID races.
Disk hashes do not bind previously imported code, and tool/font hashes do not
bind the entire TeX/OS/native dependency closure. Use a fresh process and retain
the actual environment when comparing results. An epoch is recorded context,
not proof of clean Git state, input chronology or empirical calibration.

The separate hosted manuscript job acquires checksum-pinned compatible Pandoc
and crossref binaries on a disposable runner, then regenerates/renders twice.
It records generated-byte equality and actual PDF-page pixel equality separately
from PDF-byte equality, and retains pages, figures and contact sheets for review.
That job does not install tools on contributors' workstations.
The same job exercises an actual unresolved citation in an independent copied
fixture and requires failure without a successful render receipt.

## Inspect semantic content and actual rendered pages before publication

- Inspect renderer diagnostics for undefined citations/references, missing glyphs
  and layout overflow; compilation alone is not acceptance.
- Decode every figure and inspect axes, event-region signs, annotations and
  n/seed/concentration/horizon recipes. Inspect the registered graphical abstract
  and its provenance as well as the separate demo figure.
- Inspect actual PDF pages and HTML, including tables, equations, captions and
  the rendered bibliography. Check PDF structure with an available tool such as
  `qpdf --check output/manuscript-release/alphacogant.pdf`.
- Retain source/runtime/command receipts and output hashes. Regenerate twice when
  claiming deterministic artifacts, and state exactly which bytes matched.
- Historical corporate assertions remain attributed and unverified where the
  primary source is unavailable. A clean artifact manifest does not resolve that.

## Optional external-template verification

Use a separate, disposable AlphaCOGANT checkout for this compatibility check.
The shared validator writes its own reports under `output/`.
The contained renderer above remains the standalone release workflow.

From the AlphaCOGANT checkout:

```bash
project_source="$(pwd -P)"
git clone https://github.com/docxology/template.git ../template-renderer
renderer_root="$(cd ../template-renderer && pwd -P)"
git -C "$renderer_root" checkout --detach 87704feae0a4490ef7dad9c931d268e8c30273c2
(cd "$renderer_root" && uv sync --locked --no-dev)
mkdir -p "$renderer_root/projects/working"
git clone "$project_source" "$renderer_root/projects/working/alphacogant"
cd "$renderer_root/projects/working/alphacogant"
git checkout --detach
uv sync --locked --extra dev
uv run --locked python scripts/run_alphacogant_demo.py
uv run --locked python scripts/y_generate_figures.py
uv run --locked python scripts/z_generate_manuscript_variables.py
uv run --locked python scripts/z_generate_manuscript_variables.py --check

cd "$renderer_root"
uv run --no-sync python scripts/pipeline/stage_03_render.py --project working/alphacogant
uv run --no-sync python scripts/pipeline/stage_04_validate.py --project working/alphacogant
```

Use Pandoc 3.14 with pandoc-crossref 0.3.26 built against that Pandoc
version, plus XeLaTeX and BibTeX. The external stages write
`output/pdf/alphacogant_combined.pdf`, `output/web/index.html`,
`output/reports/validation_report.json`, and
`output/reports/rendered_provenance.json` in the working project.
They use the manuscript's PDF and HTML configuration.

See the pinned [project integration guide](https://github.com/docxology/template/blob/87704feae0a4490ef7dad9c931d268e8c30273c2/projects/README.md#L208-L223),
[manuscript resolver](https://github.com/docxology/template/blob/87704feae0a4490ef7dad9c931d268e8c30273c2/infrastructure/core/project_paths.py#L59-L75),
and [stage commands](https://github.com/docxology/template/blob/87704feae0a4490ef7dad9c931d268e8c30273c2/docs/RUN_GUIDE.md#L563-L574).
