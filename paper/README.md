# Reports and preprint preparation

- `main.tex` / `main.pdf`: full technical report, prepared as a preprint draft.
- `tech_report/tech_report.tex` / `tech_report/tech_report.pdf`: short challenge summary.
- `tech_report/method_figure.*`: original method diagram, also displayed in the repository README.

The reports describe a competition engineering study. They are not presented as a
peer-reviewed workshop paper or an announced arXiv submission. The official
[challenge results](https://opensun3d.github.io/#winners) are linked separately.
The frozen geometry-based stage and final full-network finetuning stage must be
distinguished; the method diagram illustrates the former.

## Build and source package

Run from `paper/`:

```bash
tectonic --keep-intermediates main.tex
```

Run from the repository root:

```bash
python3 paper/package_arxiv.py
```

This creates `output/arxiv/articulate3d-source.tar.gz` containing the long report's
LaTeX, bibliography, generated `.bbl`, and vector figures. Generated archives and
TeX intermediates are not committed. The short report is compiled separately from
`paper/tech_report/` using `tectonic tech_report.tex`.

The package is a preparation artifact, not proof that arXiv's own compiler or
moderation has accepted it. Before submitting, confirm the author affiliation,
review the manuscript's limitations and incomplete final-run provenance, and
choose the arXiv category and distribution license. No arXiv identifier or DOI is
claimed; the repository's `CITATION.cff` currently supplies a software citation.
