# Changelog

## [2.5.0](https://github.com/tedil/rna-seq-kallisto-sleuth/compare/v2.4.0...v2.5.0) (2022-03-29)


### Features

* adapt to fgsea updates, configure fgsea precision by minimum achievable p-value ([dcd77ca](https://github.com/tedil/rna-seq-kallisto-sleuth/commit/dcd77ca90ead1213acd0c293d500c18c0e579222))
* extended diffexp tables with gene description ([#51](https://github.com/tedil/rna-seq-kallisto-sleuth/issues/51)) ([09dc9dd](https://github.com/tedil/rna-seq-kallisto-sleuth/commit/09dc9ddce9d1440267baecb191e15c2a5a4874f1))
* generate batch effect corrected matrix output ([#47](https://github.com/tedil/rna-seq-kallisto-sleuth/issues/47)) ([cd3ae35](https://github.com/tedil/rna-seq-kallisto-sleuth/commit/cd3ae3564a65a53736a72758d898e9c78c916b9a))
* join sample expressions into diffexp table ([#52](https://github.com/tedil/rna-seq-kallisto-sleuth/issues/52)) ([123923d](https://github.com/tedil/rna-seq-kallisto-sleuth/commit/123923dc7e3fb4646a3412a3204ae05d7e8fdd6f))


### Bug Fixes

* fixed custom representative transcript handling; various little bug fixes ([#54](https://github.com/tedil/rna-seq-kallisto-sleuth/issues/54)) ([3df522c](https://github.com/tedil/rna-seq-kallisto-sleuth/commit/3df522c75ff6d62ae031ba2738c1f2bde722ee34))
* Only handle canonical column in target mapping if it is actually present. ([c867026](https://github.com/tedil/rna-seq-kallisto-sleuth/commit/c867026d94490bcf02439324ca470cf7cd2e173a))
* use correct path of vega plot template even when running workflow as a module ([68b8817](https://github.com/tedil/rna-seq-kallisto-sleuth/commit/68b8817974063fbb4d1c36cf8515ecdfa7de514c))

## [2.4.0](https://github.com/snakemake-workflows/rna-seq-kallisto-sleuth/compare/v2.3.2...v2.4.0) (2022-03-29)


### Features

* adapt to fgsea updates, configure fgsea precision by minimum achievable p-value ([dcd77ca](https://github.com/snakemake-workflows/rna-seq-kallisto-sleuth/commit/dcd77ca90ead1213acd0c293d500c18c0e579222))
