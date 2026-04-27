# Stack Overflow ChatGPT Governance

This repository contains replication materials, derived outputs, figures, and manuscript PDFs for the working paper:

> *Staged Public Resolution in the Generative-AI Era: Evidence from Stack Overflow*
>
> Some manuscript files use the longer title *Private AI, Public Answer Work, and Certification at Stack Overflow*.

The project studies how Stack Overflow activity changed during the period when generative AI tools became widely available. It focuses on whether public answer work changed across stages of resolution: early answers, later answers, acceptance, and related role patterns. The analysis uses Stack Overflow question data from 2020-01-01 through 2025-12-31 across 16 technical domains. It does not directly observe private AI use by individual users.

This is a research repository intended to document and reproduce the paper's empirical workflow. It is not a software package. Some paths may need local adjustment depending on where the raw Stack Exchange dump is stored.

## Headline Panel Statistics

- **N = 2,322,009** focal-tag Stack Overflow questions (extended sample, full panel)
- **N = 2,035,885** focal-tag Stack Overflow questions (harmonized integration cut)
- 16 technical domains: `bash`, `excel`, `javascript`, `numpy`, `pandas`, `python`, `regex`, `sql`, `apache-spark`, `android`, `docker`, `firebase`, `kubernetes`, `linux`, `memory-management`, `multithreading`
- Time window: 2020-01-01 through 2025-12-31
- Treatment reference: ChatGPT public release, 2022-11-30
- Numbers reproduce in `processed/extended_sample_results.json` and `processed/harmonized_integration_results.json` under the key `n_question_level_rows`

## Main Contents

```text
.
|-- README.md
|-- LICENSE
|-- scripts/                      Python and PowerShell build/analysis scripts
|-- processed/                    Aggregated panels, model outputs, and checks
|-- figures/                      PNG figures used in the manuscript materials
|-- rendered/
|   `-- who_still_answers_submission_package_latest/
|       |-- 00_Submission_Package_Index.pdf
|       |-- 01_Manuscript.pdf
|       |-- 02_Online_Appendix.pdf
|       |-- 04_Reviewer_Memo.pdf
|       |-- supporting review materials
|       `-- source_markdown/
`-- isr_submission_package/
    |-- 01_Manuscript.pdf
    `-- 02_Online_Appendix.pdf
```

## Data

The pipeline starts from the official Stack Exchange data dump:

<https://archive.org/details/stackexchange>

This repository does not include the raw Stack Exchange archives. The raw dump is large, is distributed by Stack Exchange through Archive.org, and remains under the Stack Exchange Creative Commons license.

The `processed/` directory contains outputs derived from the raw dump, including tag-month panels, coefficient tables, validation summaries, placebo checks, permutation tests, and event-study aggregates. A small number of row-level audit or matched-pair files may also appear where they are needed to document a table. Treat all Stack Exchange derived files as source-data derivatives rather than standalone MIT data.

## Reproducing the Pipeline

1. Download the relevant Stack Exchange dump from Archive.org.
2. Extract the Stack Overflow files, including `Posts.xml`, `Users.xml`, `Comments.xml`, `Votes.xml`, `Tags.xml`, and `PostHistory.xml`.
3. Place the extracted XML or parquet files under a local `raw/` directory.
4. Check the expected paths in the build scripts, especially `scripts/build_stackoverflow_2025_dump_extension.py`.
5. Run the scripts in roughly this order:

```text
scripts/build_stackoverflow_2025_dump_extension.py
scripts/build_question_level_exposure_index_api.py
role-mechanism and closure-ladder builders
scripts/build_extended_sample_results.py
scripts/build_harmonized_integration_table.py
```

The exact run order is partly documented by script names and file dependencies. Outputs in `processed/` should reproduce within normal numerical tolerance when the same raw dump and local configuration are used.

## Important Caveats

The repository preserves the analysis state used for the current manuscript materials. It is not a claim that every script is independent, parameterized, or ready to run from a clean machine without path edits.

The empirical sample reported in the paper covers 2,322,009 focal-tag Stack Overflow questions across these domains: `bash`, `excel`, `javascript`, `numpy`, `pandas`, `python`, `regex`, `sql`, `apache-spark`, `android`, `docker`, `firebase`, `kubernetes`, `linux`, `memory-management`, and `multithreading`.

The paper's interpretation should be read with the manuscript and appendix, not from this README alone. This README is meant to help a reader understand what is in the repository and how to inspect or reproduce it.

## Citation

```bibtex
@unpublished{liu2026staged,
  author = {Liu, Sully},
  title  = {Staged Public Resolution in the Generative-AI Era: Evidence from Stack Overflow},
  year   = {2026},
  note   = {Working paper. Replication: \url{https://github.com/liusulldel/stackoverflow-chatgpt-governance}}
}
```

## License

Code and original manuscript text in this repository are released under the MIT License. See `LICENSE`.

The underlying Stack Exchange data remains under the Stack Exchange Creative Commons license at the source. Reconstructed tables, panels, excerpts, and figures derived from Stack Exchange content may inherit those terms and should be reused with Stack Exchange attribution.
