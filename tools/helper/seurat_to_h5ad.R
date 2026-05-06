#!/usr/bin/env Rscript
# Convert Seurat object from RDS or RData to H5AD format
#
# Usage:
#   Rscript seurat_to_h5ad.R --input <input.rds|input.rdata> --output <output.h5ad>
#
# Requirements:
#   - Seurat R package
#   - SeuratDisk R package

suppressPackageStartupMessages({
    library(optparse)
})

# Parse command-line arguments
option_list <- list(
    make_option(c("-i", "--input"), type="character", default=NULL,
                help="Input RDS or RData file path containing Seurat object", metavar="FILE"),
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

# Try to load Seurat package
if (!requireNamespace("Seurat", quietly=TRUE)) {
    stop("ERROR: Seurat package is not installed. Please install with: install.packages('Seurat')")
}

suppressPackageStartupMessages({
    library(Seurat)
})

# Determine file format and load object
is_rdata <- tolower(sub(".*\\.", "", input_file)) %in% c("rdata", "rda")

# Read and validate Seurat object
tryCatch({
    if (is_rdata) {
        # Load RData workspace and find Seurat object
        load(input_file)
        obj_names <- ls()
        obj <- NULL
        for (name in obj_names) {
            candidate <- get(name)
            if ("Seurat" %in% class(candidate)) {
                obj <- candidate
                break
            }
        }
        if (is.null(obj)) {
            stop("ERROR: RData file does not contain a Seurat object")
        }
    } else {
        # Read RDS file
        obj <- readRDS(input_file)
    }
    
    # Check if it's a Seurat object
    if (!("Seurat" %in% class(obj))) {
        stop(paste("ERROR: File does not contain a Seurat object. Detected class:", paste(class(obj), collapse=", ")))
    }
    
    # Check if object is valid (has cells and features)
    if (ncol(obj) == 0 || nrow(obj) == 0) {
        stop("ERROR: Seurat object is empty (no cells or features)")
    }
    
}, error=function(e) {
    stop(paste("ERROR: Failed to read or validate Seurat object:", conditionMessage(e)))
})

# Convert using SeuratDisk
if (!requireNamespace("SeuratDisk", quietly=TRUE)) {
    stop("ERROR: SeuratDisk package is not installed. Please install with: install.packages('SeuratDisk')")
}

tryCatch({
    suppressPackageStartupMessages({
        library(SeuratDisk)
    })
    
    # Create temporary H5Seurat file
    temp_h5seurat <- tempfile(fileext=".h5seurat")
    on.exit(unlink(temp_h5seurat, force=TRUE), add=TRUE)
    
    # Save as H5Seurat
    SaveH5Seurat(obj, filename=temp_h5seurat, overwrite=TRUE)
    
    # Convert to H5AD
    Convert(temp_h5seurat, dest=output_file, overwrite=TRUE)
    
    cat("SUCCESS: Converted using SeuratDisk\n")
    
}, error=function(e) {
    stop(paste("ERROR: Conversion failed:", conditionMessage(e)))
})

# Verify output file was created
if (!file.exists(output_file)) {
    stop("ERROR: Conversion appeared successful but output file was not created")
}

cat(paste("Conversion completed successfully. Output saved to:", output_file, "\n"))
