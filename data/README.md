# Inputs

Place these files in `data/` (or point `GLIOMA_DATA` at a directory containing them).
None are redistributed in this repository; all are public.

| file | what it is | where it comes from |
|---|---|---|
| `41588_2025_2168_MOESM2_ESM.xlsx` | Supplementary Table 1: specimen-level clinical annotation (patient, time point, primary/recurrent label, sample ID) | Supplementary information of Spitzer et al., *Nat Genet* 2025, doi:10.1038/s41588-025-02168-4 |
| `care_wt_state_counts.csv` | malignant cells per sample × state | produced by `scripts/dump_counts.R` from `malignant_meta_data_2025_01_08.RDS` distributed with the CARE-WT code repository (GBM-CARE-WT) |
| `care_wt_synapse_dna_snrna_id_linker.txt` | aliquot identifiers per sample | same CARE-WT repository; used only to document label disagreements |
| `gse326221_custom_pseudobulk.tsv` | per patient × time point × state weights and module scores | GSE326221 (processed tables used by the earlier analysis) |
| `gse326221_ecm_gene_expression.tsv` | 62-gene panel, archived log-scale values per pseudobulk group | as above |
| `gse292130_custom_pseudobulk.tsv`, `gse292130_ecm_gene_expression.tsv` | same two tables for the second six-state cohort | GSE292130 |
| `GSE174554_state_specific_pseudobulk.tsv` | per-state pseudobulk raw counts (genes × sample|state) | GSE174554 processed files |
| `GSE174554_cellstate_sample_summary.tsv` | per-sample cell counts per state and clinical time point | GSE174554 processed files |

Notes

* The 62-gene panel values for the two six-state cohorts are **archived** and were not
  recomputed here; the archived files do not record the normalisation and aggregation
  steps, which is why they are used as a consistency check and not as independent
  replications. This limitation is stated in the manuscript.
* GSE174554 is processed in this repository from raw per-state counts (CPM within
  pseudobulk column, log1p, gene filter), so that path is fully documented.
* No patient-identifiable data are included. The CARE-WT metadata carry cell-level
  sample identifiers only.
