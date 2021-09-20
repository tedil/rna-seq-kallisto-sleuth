**Volcano plots** computed with sleuth for wald test using the model ``{{ snakemake.params.model["full"] }}``.
The plots display beta values (regression coefficient) on the x-axis vs. significance on the y-axis and has a significance level of {{ snakemake.params.sig_level_volcano }},
which is displayed as a dotted horizontal line.
Significant genes are coloured blue and their tooltips display additional information.
Marginal histograms depict the distribution of beta values (top) and q-values (right).