#!/usr/bin/env Rscript
# Convert SingleCellExperiment object from RDS or RData to H5AD format
#
# Usage:
#   Rscript sce_to_h5ad.R --input <input.rds|input.rdata> --output <output.h5ad>
#
# Requirements:
#   - SingleCellExperiment R package (Bioconductor)
#   - zellkonverter R package (Bioconductor)

suppressPackageStartupMessages({
    library(optparse)
})

# Parse command-line arguments
option_list <- list(
    make_option(c("-i", "--input"), type="character", default=NULL,
                help="Input RDS or RData file path containing SingleCellExperiment object", metavar="FILE"),
    make_option(c("-o", "--output"), type="character", default=NULL,
                help="Output H5AD file path", metavar="FILE")
)

opt_parser <- OptionParser(option_list=option_list)
opt <- parse_args(opt_parser)

# Validate arguments
if (is.null(opt$input)) {
    stop("ERROR: --input argument is required")
}

if (is.null(opt$output)) {
    stop("ERROR: --output argument is required")
}

input_file <- opt$input
output_file <- opt$output

# Check if input file exists
if (!file.exists(input_file)) {
    stop(paste("ERROR: Input file does not exist:", input_file))
}

# Try to load SingleCellExperiment package
if (!requireNamespace("SingleCellExperiment", quietly=TRUE)) {
    stop("ERROR: SingleCellExperiment package is not installed. Please install with: BiocManager::install('SingleCellExperiment')")
}

# Try to load zellkonverter package
if (!requireNamespace("zellkonverter", quietly=TRUE)) {
    stop("ERROR: zellkonverter package is not installed. Please install with: BiocManager::install('zellkonverter')")
}

suppressPackageStartupMessages({
    library(SingleCellExperiment)
    library(zellkonverter)
})

# Determine file format and load object
is_rdata <- tolower(sub(".*\\.", "", input_file)) %in% c("rdata", "rda")

# Read and validate SingleCellExperiment object
tryCatch({
    if (is_rdata) {
        # Load RData workspace and find SingleCellExperiment object
        load(input_file)
        obj_names <- ls()
        obj <- NULL
        for (name in obj_names) {
            candidate <- get(name)
            if (is(candidate, "SingleCellExperiment")) {
                obj <- candidate
                break
            }
        }
        if (is.null(obj)) {
            stop("ERROR: RData file does not contain a SingleCellExperiment object")
        }
    } else {
        # Read RDS file
        obj <- readRDS(input_file)
    }
    
    # Check if it's a SingleCellExperiment object
    if (!is(obj, "SingleCellExperiment")) {
        stop(paste("ERROR: File does not contain a SingleCellExperiment object. Detected class:", paste(class(obj), collapse=", ")))
    }
    
    # Check if object is valid (has cells and features)
    if (ncol(obj) == 0 || nrow(obj) == 0) {
        stop("ERROR: SingleCellExperiment object is empty (no cells or features)")
    }
    
}, error=function(e) {
    stop(paste("ERROR: Failed to read or validate SingleCellExperiment object:", conditionMessage(e)))
})

# Convert using zellkonverter
tryCatch({
    # zellkonverter::writeH5AD converts SCE directly to H5AD
    writeH5AD(obj, file=output_file, overwrite=TRUE)
    
    cat("SUCCESS: Converted using zellkonverter\n")
    
}, error=function(e) {
    stop(paste("ERROR: Conversion failed:", conditionMessage(e)))
})

# Verify output file was created
if (!file.exists(output_file)) {
    stop("ERROR: Conversion appeared successful but output file was not created")
}

cat(paste("Conversion completed successfully. Output saved to:", output_file, "\n"))
