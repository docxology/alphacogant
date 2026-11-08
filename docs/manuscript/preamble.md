<!-- AlphaCOGANT manuscript preamble. Section order for the combined render:
00_abstract, 01_introduction, 02_active_inference_mapping, 03_gnn_via_cogant,
04_generative_model_inference, 05_epistemic_and_pragmatic_value,
06_integrity_and_functionality, 07_conclusion, 08_formalisms, 09_limitations,
99_references.
Margins + base font size are set in config.yaml (metadata.geometry / fontsize /
documentclass), NOT here, so geometry is loaded exactly once by pandoc. -->

# Optional template LaTeX preamble; not the contained renderer

This preamble is an optional external-template input, not a manuscript chapter or
a dependency of the contained Pandoc recipe. That recipe supplies its own font and
layout options; see [rendering and validation](../rendering_and_validation.md).
When a template consumes this file, geometry and base font size belong in
`config.yaml` (`metadata.geometry`, `metadata.fontsize`, `metadata.documentclass`),
not duplicate package loads here. The template must supply its chosen citation
and Unicode-math packages for the settings below.

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

% Optional natbib fallback for a template that explicitly loads natbib.
% The contained citeproc renderer does not consume this preamble.
\setcitestyle{numbers,square,comma}
% The manuscript already provides the "References" heading (99_references.md);
% suppress natbib's own duplicate section header so there is exactly one.
\renewcommand{\bibsection}{}

% Unicode-capable fonts (xelatex)
% The manuscript uses Greek/math unicode in prose and math.
% latinmodern-math supplies math glyphs; JuliaMono is an optional local code font.
% If JuliaMono is absent, the guard keeps the default monospace font.
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
