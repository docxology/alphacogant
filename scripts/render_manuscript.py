#!/usr/bin/env python3
"""Render checked manuscript sources to a fresh contained PDF/HTML bundle.

Requires existing Pandoc 3.13+, pandoc-crossref, XeLaTeX and qpdf on POSIX.
Installs nothing. Raw diagnostics remain for human inspection, including failures.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from alphacogant import __version__  # noqa: E402
from alphacogant.io import write_text  # noqa: E402
from alphacogant.profiling import source_fingerprint  # noqa: E402
from alphacogant.rendering import (  # noqa: E402
    check_render_diagnostics,
    html_metrics,
    prepare_render,
    render_destination,
    render_inputs,
)
from alphacogant.tokens.manuscript_artifacts import sha256  # noqa: E402


def _tool(argv: list[str], root: Path, bundle: Path, label: str) -> str:
    process = subprocess.Popen(
        argv, cwd=root, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, start_new_session=True,
    )
    timed_out = False
    try:
        stdout, stderr = process.communicate(timeout=300)
    except subprocess.TimeoutExpired:
        timed_out = True
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        stdout, stderr = process.communicate()
    except BaseException:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.communicate()
        raise
    write_text(stdout.decode("utf-8", errors="replace"), bundle / f"{label}.stdout.log")
    write_text(stderr.decode("utf-8", errors="replace"), bundle / f"{label}.stderr.log")
    if timed_out or process.returncode:
        raise ValueError(f"{label} failed ({'timeout' if timed_out else process.returncode}); "
                         f"logs retained in {bundle.relative_to(root)}.")
    if label.startswith("render-"):
        check_render_diagnostics(
            stdout.decode("utf-8", errors="replace") + "\n"
            + stderr.decode("utf-8", errors="replace")
        )
    return stdout.decode("utf-8", errors="replace").strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--source-date-epoch", type=int, required=True)
    args = parser.parse_args()
    try:
        if os.name != "posix" or args.source_date_epoch < 0:
            raise ValueError("Rendering requires POSIX and a nonnegative source-date epoch.")
        root = PROJECT_ROOT.resolve()
        bundle = render_destination(root, args.destination)
        tools = {name: shutil.which(name) for name in
                 ("pandoc", "pandoc-crossref", "xelatex", "qpdf")}
        missing = [name for name, executable in tools.items() if executable is None]
        if missing:
            raise ValueError(f"Missing tools (not installed automatically): {', '.join(missing)}")
        bundle.mkdir(parents=True, mode=0o700)
        for key in (
            "MPLCONFIGDIR", "XDG_CACHE_HOME", "TEXMFVAR", "TEXMFCONFIG", "TEXMFCACHE", "TMPDIR"
        ):
            path = bundle / ".cache" / key.lower()
            path.mkdir(parents=True)
            os.environ[key] = str(path)
        os.environ["SOURCE_DATE_EPOCH"] = str(args.source_date_epoch)
        import matplotlib
        import numpy as np
        import yaml

        context = prepare_render(root)
        config = context["config"]
        metadata = {"title": config["paper"]["title"],
                    "subtitle": config["paper"]["subtitle"] + " — v" + __version__,
                    "author": [a["name"] for a in config["authors"]],
                    "date": config["paper"]["date"], "link-citations": True}
        font_dir = Path(matplotlib.get_data_path()) / "fonts/ttf"
        fonts = []
        for field, family, italic in (("mainfont", "DejaVuSerif", "Italic"),
                                      ("monofont", "DejaVuSansMono", "Oblique")):
            names = (family, family + "-Bold", family + "-" + italic, family + "-Bold" + italic)
            metadata[field] = family
            metadata[field + "options"] = [f"Path={font_dir}/", "Extension=.ttf", *(
                f"{style}Font={name}" for style, name in zip(
                    ("Upright", "Bold", "Italic", "BoldItalic"), names, strict=True))]
            fonts.extend({"name": name + ".ttf", "sha256": sha256(font_dir / (name + ".ttf"))}
                         for name in names)
        metadata_path = bundle / "metadata.json"
        write_text(json.dumps(metadata, indent=2), metadata_path)
        versions = {name: {"version": _tool([executable, "--version"], root, bundle,
                                           name + "-version").splitlines()[0],
                           "sha256": sha256(Path(executable).resolve())}
                    for name, executable in tools.items()}
        common = [tools["pandoc"], *(p.relative_to(root).as_posix() for p in context["chapters"]),
                  "--metadata-file=" + metadata_path.relative_to(root).as_posix(),
                  "--filter=" + tools["pandoc-crossref"], "--citeproc", "--number-sections",
                  "--bibliography=docs/manuscript/references.bib", "--resource-path=docs:output"]
        commands = []
        for format_, options in (
            ("pdf", ["--pdf-engine=" + tools["xelatex"], "--pdf-engine-opt=-no-shell-escape",
                     "-V", "documentclass=" + config["metadata"]["documentclass"],
                     "-V", "fontsize=" + config["metadata"]["fontsize"],
                     "-V", "geometry:" + config["metadata"]["geometry"]]),
            ("html", ["--standalone", "--embed-resources", "--math-method=mathml"]),
        ):
            command = [*common, *options, "-o",
                       (bundle / f"alphacogant.{format_}").relative_to(root).as_posix()]
            _tool(command, root, bundle, "render-" + format_)
            commands.append(command)
        pdf = bundle / "alphacogant.pdf"
        with pdf.open("rb") as handle:
            if handle.read(5) != b"%PDF-":
                raise ValueError("Renderer did not produce a PDF document.")
        _tool([tools["qpdf"], "--check", str(pdf)], root, bundle, "pdf-check")
        pages = int(_tool([tools["qpdf"], "--show-npages", str(pdf)], root, bundle, "pdf-pages"))
        html = html_metrics(bundle / "alphacogant.html", context["figures"], root)
        if (
            render_inputs(root) != context["inputs"]
            or source_fingerprint() != context["engine_source_sha256"]
        ):
            raise ValueError("Render inputs or engine sources changed during production.")
        artifacts = [{"path": p.name, "sha256": sha256(p), "size_bytes": p.stat().st_size}
                     for p in (pdf, bundle / "alphacogant.html")]
        receipt = {"schema_version": "alphacogant.render.v1", "status": "rendered",
                   "version": __version__, "source_date_epoch": args.source_date_epoch,
                   "engine_source_sha256": context["engine_source_sha256"],
                   "runtime": {"python": sys.version, "numpy": np.__version__,
                               "matplotlib": matplotlib.__version__, "pyyaml": yaml.__version__},
                   "inputs": context["inputs"], "fonts": fonts, "tools": versions,
                   "commands": commands, "pdf_pages": pages, "html": html, "artifacts": artifacts,
                   "inspection": (
                       "Human page/figure, equation, citation and layout review required."
                   ),
                   "limitations": [
                       "Saved consistency does not certify empirical calibration.",
                       "Diagnostics require review; render success is not publication acceptance.",
                       "Hashes bind disk sources, not pre-existing imported module state.",
                       "Binary/font hashes do not bind the complete TeX/OS dependency closure.",
                   ]}
        write_text(json.dumps(receipt, indent=2, sort_keys=True), bundle / "render.json")
        print(json.dumps({"bundle": bundle.relative_to(root).as_posix(), "artifacts": artifacts,
                          "pdf_pages": pages, "html": html, "inspection": receipt["inspection"]}))
        return 0
    except (OSError, ValueError, TypeError, LookupError) as exc:
        print(f"manuscript render: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
