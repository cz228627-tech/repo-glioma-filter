# Extract malignant-cell counts per sample x state from the CARE-WT metadata.
#
# The metadata object is distributed with the CARE consortium code repository
# (GBM-CARE-WT) and is also described in the supplementary material of the
# associated publication. It is not redistributed here.
#
# Usage:
#   Rscript scripts/dump_counts.R "path/to/malignant_meta_data_2025_01_08.RDS" data/care_wt_state_counts.csv

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop("usage: Rscript dump_counts.R <metadata.RDS> <out.csv>")
}
m <- as.data.frame(readRDS(args[1]))
cnt <- as.data.frame(table(m$ID, m$State))
names(cnt) <- c("ID", "State", "n")
cnt$patient <- sub("T[0-9]+$", "", cnt$ID)
cnt$tp <- sub("^P[0-9]+", "", cnt$ID)
write.csv(cnt, args[2], row.names = FALSE)
cat("wrote", args[2], "\n")
