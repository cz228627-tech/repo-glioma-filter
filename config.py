"""Paths and shared constants.

Set the environment variable GLIOMA_DATA to point at a directory containing the
inputs listed in data/README.md. Defaults to ./data next to this file.
"""
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("GLIOMA_DATA", os.path.join(ROOT, "data"))
OUT = os.environ.get("GLIOMA_OUT", os.path.join(ROOT, "outputs"))

# inputs (see data/README.md)
CARE_WT_CLINICAL = os.path.join(DATA, "41588_2025_2168_MOESM2_ESM.xlsx")
CARE_WT_COUNTS = os.path.join(DATA, "care_wt_state_counts.csv")
CARE_WT_LINKER = os.path.join(DATA, "care_wt_synapse_dna_snrna_id_linker.txt")
CARE_EXPR = os.path.join(DATA, "gse326221_ecm_gene_expression.tsv")
CARE_META = os.path.join(DATA, "gse326221_custom_pseudobulk.tsv")
G2_EXPR = os.path.join(DATA, "gse292130_ecm_gene_expression.tsv")
G2_META = os.path.join(DATA, "gse292130_custom_pseudobulk.tsv")
G174_EXPR = os.path.join(DATA, "GSE174554_state_specific_pseudobulk.tsv")
G174_META = os.path.join(DATA, "GSE174554_cellstate_sample_summary.tsv")

MIN_CELLS = 20
THRESHOLDS = [0, 5, 10, 20, 50, 100]
SEED = 20261029

os.makedirs(OUT, exist_ok=True)
