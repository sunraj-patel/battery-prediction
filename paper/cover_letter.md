Dear Editor,

We submit our manuscript "Publication-aware prediction of battery material properties from literature-extracted data" for consideration as a Full Paper in *Digital Discovery*.

Text-mined materials databases are increasingly used to train property-prediction models, and *Digital Discovery* has published much of the work on building and using them. Records in such databases are not independent: one publication reports many related values, so random splits are widely expected to overstate performance. We test this expectation on the auto-generated battery materials database of Huang and Cole (285,243 records from 42,840 publications). The result is more nuanced than a simple leakage argument predicts, and the reasons are useful to anyone who evaluates models on text-mined data:

- Between 30% and 84% of target variance lies between publications, yet DOI-grouped splitting lowers $R^2$ by only 0.02–0.08. Composition descriptors carry almost no recoverable publication identity, so clustering alone does not produce leakage.
- A different leakage route survives DOI grouping: 16–48% of DOI-grouped test records repeat a (material, value) pair reported by another publication in training. These records are predicted far better than values without such a precedent, and on the latter the achievable $R^2$ falls to about 0.25–0.35 for capacity, voltage and energy, and to zero for conductivity and Coulombic efficiency.
- We adapt the public pipelines of the two previous regression studies on the same database (Jablonka 2026; Zhuang and Barnard 2024) to identical partitions, so that the comparison is on a common basis.

The study follows *Digital Discovery*'s reproducibility expectations. Every excluded column is listed with its reason, the database's warning flags are audited (one encodes publication identity and another is a function of the target), splits are fingerprinted so that results from different machines can be paired, and all numbers in the manuscript are generated from the stored run outputs by a single script. Code, processed features and result tables are available to reviewers at https://github.com/sunraj-patel/battery-prediction and will be archived on Zenodo on acceptance.

The manuscript has not been published and is not under consideration elsewhere. All authors approve the submission. There are no conflicts of interest.

Suggested reviewers (optional): [names, affiliations, e-mail]

Yours sincerely,
Sunraj Patel and Shikhar Misra
[affiliation, corresponding e-mail]
