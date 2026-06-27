<!-- AlphaCOGANT manuscript preamble. Section order for the combined render:
00_abstract, 01_introduction, 02_active_inference_mapping, 03_gnn_via_cogant,
04_generative_model_inference, 05_epistemic_and_pragmatic_value,
06_integrity_and_functionality, 07_conclusion, 08_formalisms, 09_limitations,
99_references.
Margins + base font size are set in config.yaml (metadata.geometry / fontsize /
documentclass), NOT here, so geometry is loaded exactly once by pandoc. -->

# LaTeX Preamble

LaTeX packages injected before `\begin{document}` for the combined PDF. Geometry and
base font size live in `config.yaml` (`metadata.geometry`, `metadata.fontsize`,
`metadata.documentclass: extarticle`).

```latex
% Core mathematics
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{amsfonts}

% Layout, floats, graphics
\usepackage{float}
\usepackage{graphicx}

% Tables
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{array}

% Typography
\usepackage{microtype}
\usepackage{xcolor}

% Cross-references and citations
\usepackage{hyperref}
\hypersetup{
    colorlinks=true,
    allcolors=blue
}
\usepackage[capitalise,noabbrev]{cleveref}

% Auto-numbered, hyperlinked citations from references.bib. The natbib package
% options (numbers, square, sort&compress) and biblio-style (unsrtnat, number by
% citation order) are set via pandoc metadata in 00_abstract.md; this fallback
% guarantees numeric square-bracket citations even if that metadata is dropped.
\setcitestyle{numbers,square,comma}
% The manuscript already provides the "References" heading (99_references.md);
% suppress natbib's own duplicate section header so there is exactly one.
\renewcommand{\bibsection}{}

% Unicode-capable fonts (xelatex)
% The manuscript uses Greek/math unicode in prose and math.
% latinmodern-math gives broad math coverage; JuliaMono covers code glyphs.
% If JuliaMono is absent, the guard leaves xelatex on its default mono font.
\usepackage{fontspec}
\IfFontExistsTF{JuliaMono-Regular}{%
  \setmonofont{JuliaMono-Regular}[
    Extension      = .ttf,
    UprightFont    = *,
    BoldFont       = JuliaMono-Bold,
    ItalicFont     = JuliaMono-RegularItalic,
    BoldItalicFont = JuliaMono-BoldItalic,
    Scale          = MatchLowercase,
  ]%
}{}
\setmathfont{latinmodern-math.otf}
```
