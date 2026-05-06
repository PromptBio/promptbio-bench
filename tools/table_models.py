"""Data models for table alignment and comparison operations (e.g., column mapping, row ID detection, alignment plans, comparison plans)."""

from typing import Dict, Optional, List, Literal, TypeAlias, Any
from pydantic import BaseModel, Field

# Type alias for simple column mapping: maps reference column names to candidate column names
# None value means no match found for that reference column
ColumnMappingDict: TypeAlias = Dict[str, Optional[str]]


class IndexMapping(BaseModel):
    """Structured result for a mapping from source keys to target keys (e.g. row index alignment, entity mapping)."""
    mapping: Dict[str, Optional[str]] = Field(
        description="Map each source key to the matching target key, or null if no good match exists."
    )
    reasoning: Optional[str] = Field(
        default=None,
        description="Brief explanation of how source items were matched to targets."
    )
    confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Confidence score between 0.0 and 1.0 for the overall mapping.",
    )


class ColumnMapEntry(BaseModel):
    """Entry for a single column mapping with per-mapping confidence and reasoning."""
    ref_column: str = Field(
        description="The reference column name being mapped."
    )
    alt_column: Optional[str] = Field(
        description="The corresponding candidate column name, or null if no good match exists."
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score between 0.0 and 1.0 for this specific mapping."
    )
    reasoning: str = Field(
        description="Brief explanation of why this mapping was chosen (or why no match exists)."
    )


class ColumnMapping(BaseModel):
    """Structured result for mapping reference columns to candidate columns with per-mapping details."""
    mapping: Dict[str, ColumnMapEntry] = Field(
        description="Map each reference column to its corresponding candidate column entry (with alt_column, confidence, and reasoning)."
    )
    overall_confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Overall confidence score between 0.0 and 1.0 for the entire mapping."
    )
    overall_reasoning: Optional[str] = Field(
        default=None,
        description="Brief explanation of the overall mapping strategy and any patterns observed."
    )


class RowIDColumns(BaseModel):
    """Structured result for identifying row identifier columns."""
    id_columns: List[str] = Field(
        description="List of column names that should be used as row identifiers (e.g., ['gene_id'], ['sample_id', 'replicate'], ['id']). Empty list if no suitable ID columns found."
    )
    reasoning: Optional[str] = Field(
        default=None,
        description="Brief explanation of why these columns were chosen as identifiers."
    )
    confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Confidence score between 0.0 and 1.0 for the ID column selection.",
    )


class AlignPlan(BaseModel):
    """Configuration for structural alignment of dataframes.
    
    This configuration guides how to align two dataframes structurally:
    1. Table presentation alignment (transpose/pivot detection)
    2. Column alignment (semantic matching, case-insensitive, intersection)
    3. Row alignment (ID column detection or index-based matching)
    4. Sorting (if needed)
    """
    # ========== TABLE PRESENTATION ==========
    needs_transpose: bool = Field(
        default=False,
        description="Whether the candidate table should be transposed to match reference structure. True if tables appear to be transposed versions of each other (e.g., ref has genes as rows, alt has genes as columns)."
    )
    
    # ========== ALIGNMENT OPERATIONS ==========
    align_rows: bool = Field(
        default=True,
        description="Whether to align rows before comparison. Usually true, but false if tables are already aligned or row alignment is not meaningful."
    )
    align_columns: bool = Field(
        default=True,
        description="Whether to align columns before comparison. Usually true, but false if column names already match exactly."
    )
    
    # ========== COLUMN MAPPING ==========
    column_mapping: Optional[ColumnMappingDict] = Field(
        default=None,
        description="Mapping from reference column names to candidate column names (ref_col -> alt_col). "
                    "None value for a ref column means no match found. "
                    "If the entire field is None, column alignment will auto-detect via LLM fallback."
    )
    
    # ========== ROW IDENTIFIERS ==========
    id_columns: Optional[List[str]] = Field(
        default=None,
        description="Column name(s) to use as row identifiers for row alignment (use reference column names). "
                    "Empty list means use positional/index-based alignment. "
                    "None means auto-detect via LLM fallback."
    )
    
    # ========== SORTING ==========
    row_order_sensitive: bool = Field(
        description="Whether row order matters for comparison. True for ranked lists (e.g., top N results), false for sets/unordered data (e.g., gene lists)."
    )
    column_order_sensitive: bool = Field(
        default=False,
        description="Whether column order matters for comparison. Usually false, but true if column order is semantically meaningful."
    )
    pre_sort: bool = Field(
        default=False,
        description="Whether to sort dataframes before comparison. True if row_order_sensitive is true but data might be unsorted, or if sorting by a key column would help alignment."
    )
    sort_by: Optional[str] = Field(
        default=None,
        description="Column name(s) to sort by (comma-separated if multiple). Only used if pre_sort is true. Should be a column that exists in both tables after alignment."
    )
    
    # ========== PERMUTATION DETECTION ==========
    check_permutation: bool = Field(
        description="Whether to check if dataframes are permutations of each other. True if row_order_sensitive is false (order doesn't matter, so check if same data in different order)."
    )
    
    # ========== METADATA ==========
    reasoning: str = Field(
        description="Brief explanation of the alignment strategy, including key decisions made (e.g., why certain columns were mapped, why specific ID columns were chosen, why transpose was needed)."
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score between 0.0 and 1.0 for the alignment plan. Higher values indicate more certainty (e.g., exact column name matches, clear ID columns)."
    )


class ColumnCompareConfig(BaseModel):
    """Configuration for a single column's comparison strategy.
    
    Specifies how to compare a specific column based on its data type.
    
    Aggregation Model:
    All cell-by-cell comparisons (exact, overlap, rae) use unified averaging:
    1. Compute a metric series (0/1 for exact, Jaccard values for overlap, RAE values for rae)
    2. If cell_tolerance is set: threshold values to 0/1, then average
    3. If cell_tolerance is None: average raw metric values
    4. Summary strategies (pearson, spearman) compute a single correlation value (no aggregation needed)
    
    This unified approach:
    - threshold = 0 → exact matching
    - threshold = small value → tolerance-based matching
    - threshold = inf/None → raw average of metric values
    """
    name: str = Field(
        description="Column name (reference column name after alignment, excluding ID columns)."
    )
    data_type: Literal["categorical", "numerical"] = Field(
        description="Data type of the column: 'categorical' (object, string, category) or 'numerical' (int, float, numeric)."
    )
    strategy: Literal["exact", "overlap", "pearson", "spearman", "rae", "ignore"] = Field(
        description="Comparison strategy for this column: "
                    "For categorical: "
                    "'exact' = cell-by-cell exact match (0/1 series), average gives similarity. "
                    "'overlap' = cell-by-cell Jaccard similarity (can be thresholded via cell_threshold or averaged raw). "
                    "For numerical: "
                    "'pearson' = Pearson correlation (summary statistic, focuses on linear relationship). "
                    "'spearman' = Spearman correlation (summary statistic, focuses on monotonic/rank relationship). "
                    "'rae' = cell-by-cell relative absolute error (can be thresholded via cell_tolerance or averaged raw). "
                    "'ignore' = skip this column."
    )
    cell_delimiter: Optional[str] = Field(
        default=None,
        description="Optional delimiter string for parsing cell values when strategy is 'overlap' for categorical columns. "
                    "Common delimiters: ',' (comma), ';' (semicolon), '|' (pipe), '\\t' (tab). "
                    "None means no parsing (treat cell as single value). "
                    "Only used for categorical columns with 'overlap' strategy."
    )
    cell_tolerance: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Optional tolerance for RAE comparison strategy (numerical columns). "
                    "When set: RAE values <= tolerance are counted as matching. "
                    "When None: raw RAE values are averaged and converted to similarity. "
                    "For 'rae': threshold on RAE (e.g., 0.01 = RAE values <= 0.01 count as matching). "
                    "For other strategies: not applicable."
    )
    cell_threshold: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Optional threshold for Jaccard similarity in 'overlap' strategy (categorical columns). "
                    "When set: pairs with Jaccard similarity >= threshold are counted as matching. "
                    "When None: raw Jaccard similarity values are averaged. "
                    "For 'overlap': threshold on Jaccard similarity (e.g., 0.8 = pairs with similarity >= 0.8 count as matching). "
                    "For other strategies: not applicable."
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score between 0.0 and 1.0 for the comparison strategy. Higher values indicate more certainty in the decisions made (e.g., clear data patterns, unambiguous task requirements)."
    )
    reasoning: str = Field(
        description="Brief explanation of the comparison strategy, including key decisions made (e.g., why certain columns are prioritized, why specific tolerance was chosen, why this comparison focus was selected)."
    )


class ComparisonPlan(BaseModel):
    """Configuration for comparison strategy after dataframes are aligned.
    
    This configuration guides how to compare values in aligned dataframes:
    1. Which columns to compare
    2. Per-column comparison strategies with data types and tolerances
    """
    # ========== COLUMN-SPECIFIC COMPARISON ==========
    column_configs: Optional[List[ColumnCompareConfig]] = Field(
        default=None,
        description="List of column configurations specifying which columns to compare and how. "
                    "Each ColumnCompareConfig specifies: column name, data type (categorical/numerical), "
                    "comparison strategy, and optional cell delimiter for parsing. "
                    "If None, compare all common columns except ID columns with inferred strategies."
    )
    
    # ========== METADATA ==========
    overall_reasoning: str = Field(
        description="Brief explanation of the comparison strategy, including key decisions made (e.g., why certain columns are prioritized, why specific tolerance was chosen, why this comparison focus was selected)."
    )
    overall_confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score between 0.0 and 1.0 for the comparison plan. Higher values indicate more certainty in the decisions made (e.g., clear data patterns, unambiguous task requirements)."
    )


class TableMeta(BaseModel):
    """Tabular file parsing metadata detected by LLM or heuristic fallback."""
    delimiter: Optional[str] = Field(
        default=None,
        description="Column delimiter character. Common values: ',' for CSV, '\\t' for TSV, "
                    "';' for semicolon-separated, '|' for pipe-separated. None if cannot be determined."
    )
    comment_char: Optional[str] = Field(
        default=None,
        description="Single character that marks comment/metadata lines to skip during parsing. "
                    "Common values: '#' (most common in bioinformatics), '%'. "
                    "None if no comment lines exist in the file."
    )
    has_header: bool = Field(
        default=True,
        description="Whether the first non-comment, non-blank line is a column header row "
                    "containing column names (e.g., 'gene_id', 'expression', 'pvalue'). "
                    "True if it contains descriptive names, False if it is actual data values."
    )
    skip_rows: int = Field(
        default=0,
        description="Number of non-comment metadata lines at the top of the file to skip. "
                    "Do NOT count comment lines here (they are handled by comment_char). "
                    "Only count lines like 'File: output.csv' or blank lines that don't start "
                    "with the comment character. 0 if no such lines exist."
    )
    reasoning: str = Field(
        default="",
        description="Brief explanation of why these parsing parameters were chosen."
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0, le=1.0,
        description="Confidence score (0.0-1.0) in the detected parsing parameters."
    )


class ColumnSimilarity(BaseModel):
    """Standardized result for column similarity calculations."""
    similarity: Optional[float] = Field(
        description="Similarity score between 0.0 and 1.0, or None if calculation failed"
    )
    strategy: str = Field(
        description="Comparison strategy used (e.g., 'exact', 'overlap', 'pearson', 'spearman', 'rae')"
    )
    matching_count: Optional[int] = Field(
        default=None,
        description="Number of cells that matched within tolerance (for exact/rae strategies)"
    )
    total_count: Optional[int] = Field(
        default=None,
        description="Total number of cells compared"
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message or reason for edge cases (e.g., 'Both empty', 'One empty', 'All values invalid')"
    )
    details: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata (e.g., tolerance, correlation, delimiter, valid_pairs, total_pairs)"
    )

