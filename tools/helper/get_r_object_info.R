#!/usr/bin/env Rscript
# Get R object type and information in RDS or RData file
#
# Usage:
#   Rscript get_r_object_info.R --input <input.rds|input.rdata>
#
# Arguments:
#   --input: Input RDS or RData file path (required)
#
# Output:
#   Prints JSON to stdout with object information:
#   - For RDS: {"<filename>": {"type": "...", "info": {...}}}
#   - For RData: {"object1": {"type": "...", "info": {...}}, "object2": {...}}
#   
#   Info fields included for each type:
#   - "vector": type, length, mode
#   - "matrix": type, n_row, n_col, mode
#   - "dataframe": type, n_row, n_col, columns
#   - "list": type, length, names (if named list)
#   - "seurat": type, n_obs (number of cells), n_var (number of features), samples (if available)
#   - "sce": type, n_obs (number of cells), n_var (number of features), samples (if available)
#   - "unknown": type only

suppressPackageStartupMessages({
    library(optparse)
    library(jsonlite)
})

# Parse command-line arguments
option_list <- list(
    make_option(c("-i", "--input"), type="character", default=NULL,
                help="Input RDS or RData file path", metavar="FILE")
)

opt_parser <- OptionParser(option_list=option_list)
opt <- parse_args(opt_parser)

# Validate arguments
if (is.null(opt$input)) {
    cat("ERROR: Input file path is required\n")
    quit(status=1)
}

input_file <- opt$input

# Check if input file exists
if (!file.exists(input_file)) {
    cat("ERROR: Input file does not exist\n")
    quit(status=1)
}

# Determine file format
file_ext <- tolower(sub(".*\\.", "", input_file))
is_rdata <- file_ext %in% c("rdata", "rda")

# Function to detect object type
detect_type <- function(obj) {
    # Check for Seurat (must check first as Seurat objects can have other classes)
    if ("Seurat" %in% class(obj)) {
        return("seurat")
    }
    
    # Check for SingleCellExperiment
    if (requireNamespace("SingleCellExperiment", quietly=TRUE)) {
        if (methods::is(obj, "SingleCellExperiment")) {
            return("sce")
        }
    }
    
    # Get primary class
    obj_class <- class(obj)[1]
    
    # Check standard R types
    if (obj_class == "matrix") {
        return("matrix")
    } else if (obj_class == "data.frame") {
        return("dataframe")
    } else if (obj_class == "list") {
        return("list")
    } else if (obj_class %in% c("numeric", "character", "logical", "integer", "complex")) {
        return("vector")
    }
    
    # Unknown type
    return("unknown")
}

# Function to get object info as a list
get_object_info_list <- function(obj, obj_type) {
    info <- list(type = obj_type)
    
    if (obj_type == "vector") {
        info$length <- length(obj)
        info$mode <- mode(obj)
    } else if (obj_type == "matrix") {
        info$n_row <- nrow(obj)
        info$n_col <- ncol(obj)
        info$mode <- mode(obj)
    } else if (obj_type == "dataframe") {
        info$n_row <- nrow(obj)
        info$n_col <- ncol(obj)
        info$columns <- colnames(obj)
    } else if (obj_type == "list") {
        info$length <- length(obj)
        if (!is.null(names(obj))) {
            info$names <- names(obj)
        }
    } else if (obj_type %in% c("seurat", "sce")) {
        # Seurat and SCE objects: ncol = cells (n_obs), nrow = features (n_var)
        tryCatch({
            info$n_obs <- ncol(obj)
            info$n_var <- nrow(obj)
            # Try to get number of unique samples if available
            tryCatch({
                if (obj_type == "seurat") {
                    # Seurat: use meta.data - check multiple common column names
                    meta_data <- obj@meta.data
                    sample_col <- NULL
                    if ("sample" %in% colnames(meta_data)) {
                        sample_col <- meta_data$sample
                    } else if ("Sample" %in% colnames(meta_data)) {
                        sample_col <- meta_data$Sample
                    } else if ("sample_id" %in% colnames(meta_data)) {
                        sample_col <- meta_data$sample_id
                    } else if ("orig.ident" %in% colnames(meta_data)) {
                        # orig.ident is often used for sample identity in Seurat
                        sample_col <- meta_data$orig.ident
                    } else if ("batch" %in% colnames(meta_data)) {
                        # batch is sometimes used as sample identifier
                        sample_col <- meta_data$batch
                    }
                    if (!is.null(sample_col)) {
                        info$samples <- length(unique(sample_col))
                    }
                } else if (obj_type == "sce") {
                    # SCE: use colData - check multiple common column names
                    if (requireNamespace("SingleCellExperiment", quietly=TRUE)) {
                        col_data <- colData(obj)
                        sample_col <- NULL
                        if ("sample" %in% colnames(col_data)) {
                            sample_col <- col_data$sample
                        } else if ("Sample" %in% colnames(col_data)) {
                            sample_col <- col_data$Sample
                        } else if ("sample_id" %in% colnames(col_data)) {
                            sample_col <- col_data$sample_id
                        } else if ("batch" %in% colnames(col_data)) {
                            sample_col <- col_data$batch
                        }
                        if (!is.null(sample_col)) {
                            info$samples <- length(unique(sample_col))
                        }
                    }
                }
            }, error=function(e) {
                # If metadata access fails, skip samples
            })
        }, error=function(e) {
            # If extraction fails, just include type
        })
    }
    
    return(info)
}

# Get base filename without extension for RDS files
get_base_name <- function(file_path) {
    basename_no_ext <- sub("\\.[^.]*$", "", basename(file_path))
    return(basename_no_ext)
}

# Try to get object type and information
tryCatch({
    if (is_rdata) {
        # Load RData workspace - get info for ALL objects
        env <- new.env()
        tryCatch({
            load(input_file, envir=env)
        }, error=function(e) {
            cat("ERROR: Failed to load RData file:", conditionMessage(e), "\n")
            quit(status=1)
        })
        obj_names <- ls(envir=env)
        
        if (length(obj_names) == 0) {
            cat("{}")
            quit(status=0)
        }
        
        result <- list()
        for (name in obj_names) {
            tryCatch({
                obj <- get(name, envir=env)
                obj_type <- detect_type(obj)
                result[[name]] <- get_object_info_list(obj, obj_type)
            }, error=function(e) {
                result[[name]] <<- list(type = "unknown", error = conditionMessage(e))
            })
        }
        
        cat(toJSON(result, auto_unbox=TRUE, pretty=FALSE))
        quit(status=0)
    } else {
        # RDS file - single object, use filename as key
        tryCatch({
            obj <- readRDS(input_file)
        }, error=function(e) {
            cat("ERROR: Failed to read RDS file:", conditionMessage(e), "\n")
            quit(status=1)
        })
        obj_type <- detect_type(obj)
        result <- list()
        key_name <- get_base_name(input_file)
        result[[key_name]] <- get_object_info_list(obj, obj_type)
        
        cat(toJSON(result, auto_unbox=TRUE, pretty=FALSE))
        quit(status=0)
    }
    
}, error=function(e) {
    cat("ERROR: Unexpected error:", conditionMessage(e), "\n")
    quit(status=1)
})
