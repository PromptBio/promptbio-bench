import json
import logging
from pathlib import Path
from typing import Optional

from utils.agents import create_llm_agent, invoke_structured_agent
from utils.models import EvalFile, StrategyRecommendation, StrategyResult

logger = logging.getLogger(__name__)


class StrategyRecommender:
    """Class to handle strategy recommendations for tasks."""

    def __init__(self, tool_schema_path: Path, model: str = "gpt-5.4"):
        """
        Initialize the StrategyRecommender.

        Args:
            tool_schema_path: Path to tools/tool_schema.json
            model: LLM model to use for recommendations
        """
        self.model = model
        if not Path(tool_schema_path).exists():
            raise FileNotFoundError(f"Tool schema not found at {tool_schema_path}")
        with open(tool_schema_path) as f:
            self.tool_schema = json.load(f)
        self._param_type_map: dict[str, dict[str, dict[str, str]]] = {
            fmt_name: {
                strat_name: {
                    p["name"]: p["type"]
                    for p in strat_data.get("parameters", [])
                    if "name" in p and "type" in p
                }
                for strat_name, strat_data in fmt_data.get("strategies", {}).items()
            }
            for fmt_name, fmt_data in self.tool_schema.get("formats", {}).items()
        }
        self.system_prompt = self._build_system_prompt()

    def _build_system_prompt(self) -> str:
        return """\
You are an expert in bioinformatics and data science. Your task is to recommend the most appropriate comparison strategy and parameters for evaluating a bioinformatics task output.

## Step 1 — Parse the eval_guideline first

The eval_guideline is written by a domain expert. Read it to understand what must match and how precisely — it informs your strategy choice but does not mechanically determine it:
- **What must match**: "sequence must match" → record-level comparison (exact/approximate). "per-class metrics must match" → aggregate comparison (summary). "plot must show X" → semantic.
- **Acceptable variation**: "row and column order may differ" → approximate handles this. "case and line-wrap may differ" → exact with ignore_case.
- **Tolerances**: Numeric tolerances ("+/-0.10", "1% relative absolute error") tell you how precisely values must agree. The eval_guideline is passed to the comparison engine directly, but also use tolerances to set numeric strategy parameters where applicable (e.g., `tolerance` for rdata/anndata/bed approximate strategies). A very loose tolerance can be a secondary signal of a stochastic task, but confirm with the task question before switching to summary.

## Step 2 — Check for stochasticity

If the task involves any of the following, use 'summary' regardless of file format (unless the guideline explicitly says record-level matching is expected):
- Random train/test split, "70% training", holdout set
- Neural network training, gradient descent, model fitting
- Bootstrap, permutation, Monte Carlo sampling
- Stochastic algorithm (k-means random init, t-SNE, diffusion maps)
- Random seed required for reproducibility

**Exception**: If the task fixes a random seed AND the guideline says "must match exactly", use exact or approximate instead.

## Step 3 — Apply format defaults (when guideline and stochasticity give no override)

| Format | Default strategy | Override when |
|---|---|---|
| csv / tsv / gct / table | approximate | Stochastic task → summary; bit-identical requirement → exact |
| txt / image / script | semantic | Structured text (BLAST tabular) → approximate or exact; pure numeric scalar → numeric |
| fasta / fastq | exact | Assembly polishing or consensus → approximate; metagenomics or random subsampling → summary |
| bam / sam / cram | summary | Coverage uniformity task → coverage; variant-level goal → variant |
| bed / bedgraph / narrowPeak | approximate | Score-pattern task → correlation; total coverage goal → overlap |
| vcf / bcf | summary | Identical caller + params with exact requirement → exact |
| anndata / h5ad / loom | summary | Always summary for single-cell (clustering is inherently stochastic) |
| bigwig / wig | summary | Identical deterministic pipeline → approximate or exact |
| tbi / bai / csi / crai | functional | Always; set indexed_file from the input files list |
| pdb / cif | exact | Structure prediction task → approximate |

## Step 4 — Set parameters precisely

**anndata summary**: Set `reduction` to the embedding the task produces (X_umap for UMAP, X_pca for PCA, X_tsne for t-SNE). Set `cluster_col` to the algorithm name if mentioned (leiden, louvain, kmeans); leave null for auto-detection.

**fasta exact / approximate**: Set `by_name=true` when the task produces a multi-FASTA with meaningful sequence IDs (transcript IDs, gene names, contig names). Leave `by_name=false` (default) for single-sequence outputs or content-based matching. Set `single_line=true` only when the task explicitly converts multiline → single-line FASTA; leave false otherwise.

**tbi / bai functional**: Set `indexed_file` to the corresponding BAM/CRAM/VCF file from the task input files list.

**bam variant**: Only use when the task goal is variant-level functional equivalence; `ref_fasta` is required — infer its path from the task input files list.

**bed correlation**: Set `correlation_method=spearman` for rank-based score comparisons; pearson for linear signal comparisons.

**rdata / anndata approximate**: Set `tolerance` from the eval_guideline numeric precision (e.g., "+/-0.001" → `tolerance=0.001`).

**bed approximate**: Set `tolerance` (positional, in bp) if the eval_guideline specifies a coordinate slack (e.g., "+/-1 bp" → `tolerance=1`).

## Output format

Respond with:
1. strategy — exactly one of the listed strategy names for this format
2. parameters — a dict of overrides (omit parameters you are leaving at default)
3. reasoning — 1-2 sentences: state the deciding factor (guideline signal, stochasticity, or format default) and why the chosen strategy is appropriate
4. confidence — 0.0–1.0; use lower values when eval_guideline is absent or ambiguous"""

    def _build_user_prompt(self, question: str, input_files: Optional[list[str]], eval_file: EvalFile) -> str:
        sections = [f"Task question: {question}"]

        if input_files:
            sections.append("Task input files:\n" + "\n".join(f"  - {f}" for f in input_files))

        eval_file_section = f"Output file to evaluate:\n  Reference answer: {eval_file.reference_file} (format: {eval_file.file_format})"
        if eval_file.candidate_file:
            eval_file_section += f"\n  Candidate output: {eval_file.candidate_file}"
        if eval_file.eval_guideline:
            eval_file_section += f"\n  Evaluation guideline: {eval_file.eval_guideline}"
        sections.append(eval_file_section)

        fmt_entry = self.tool_schema.get("formats", {}).get(eval_file.file_format.lower())
        if fmt_entry is None:
            sections.append(f"(No schema entry found for format '{eval_file.file_format}')")
        else:
            default = fmt_entry.get("default_strategy", "")
            schema_lines = [f"Available strategies for '{eval_file.file_format}' (default: {default}):"]
            for strat_name, strat_data in fmt_entry.get("strategies", {}).items():
                marker = " [DEFAULT]" if strat_name == default else ""
                schema_lines.append(f"  {strat_name}{marker}: {strat_data.get('when_to_use', '')}")
                if params := strat_data.get("parameters", []):
                    schema_lines.append("    Parameters: " + " | ".join(
                        f"{p['name']}:{p['type']}={'REQUIRED' if p.get('required') else repr(p.get('default'))}"
                        + (f", one of {p['enum']}" if "enum" in p else "")
                        + f" — {p['description']}"
                        for p in params
                    ))
            sections.append("\n".join(schema_lines))

        return "\n\n".join(sections)

    def _coerce_params(self, params: dict, file_format: str, strategy: str) -> dict:
        """Coerce LLM-returned parameter values to their schema-declared types."""
        strat_types = self._param_type_map.get(file_format, {}).get(strategy, {})
        if not strat_types:
            return params
        coerced = {}
        for key, value in params.items():
            t = strat_types.get(key)
            try:
                if t == "bool" and not isinstance(value, bool):
                    coerced[key] = value.lower() in ("true", "1", "yes") if isinstance(value, str) else bool(value)
                elif t == "int" and not isinstance(value, int):
                    coerced[key] = int(value)
                elif t == "float" and not isinstance(value, (int, float)):
                    coerced[key] = float(value)
                elif t == "list" and not isinstance(value, list):
                    coerced[key] = json.loads(value) if isinstance(value, str) else list(value)
                else:
                    coerced[key] = value
            except Exception:
                logger.warning("Could not coerce param %r value %r to %r; keeping original", key, value, t)
                coerced[key] = value
        return coerced

    def recommend(self, question: str, input_files: Optional[list[str]] = None, eval_files: list[EvalFile] = None, verbose: bool = False) -> list[StrategyResult]:
        """
        Recommend a comparison strategy for each candidate file of a task.

        Args:
            question: The task question / description.
            input_files: All input files provided to the agent for this task.
                These are task-level context shared across all eval files.
                When provided, the LLM can infer source relationships (e.g. which
                input a .tbi indexes) and set parameters like indexed_file.
            eval_files: One entry per candidate file to evaluate.
                Set EvalFile.eval_guideline for per-file evaluation guidance.
            verbose: Print per-file results to stdout.

        Returns:
            One StrategyResult per eval file, in the same order as eval_files.
            StrategyResult.error is None on success and set to the error message on
            LLM failure; all other fields retain their default values in that case.

        Raises:
            Exception: If the LLM agent cannot be created (unrecoverable).
        """
        if verbose:
            print(f"Requesting strategy recommendation for {len(eval_files)} eval file(s).")

        agent = create_llm_agent(
            model=self.model,
            system_prompt=self.system_prompt,
            response_format=StrategyRecommendation,
            temperature=0.0,
        )

        results = []
        for eval_file in eval_files:
            user_prompt = self._build_user_prompt(question, input_files, eval_file)

            try:
                recommendation = invoke_structured_agent(agent, user_prompt, StrategyRecommendation)
                coerced_params = self._coerce_params(recommendation.parameters, eval_file.file_format, recommendation.strategy)
                result = StrategyResult(**eval_file.model_dump(exclude={"strategy", "parameters"}), **{**recommendation.model_dump(), "parameters": coerced_params})
            except Exception as e:
                if verbose:
                    print(f"  Warning: LLM call failed for {eval_file.file_format} ({eval_file.reference_file}): {e}")
                result = StrategyResult(**eval_file.model_dump(), error=str(e))

            results.append(result)
            if verbose:
                status = f"error: {result.error}" if result.error else f"{result.strategy} (confidence={result.confidence:.2f})"
                print(f"  {eval_file.file_format} ({eval_file.reference_file}): {status}")

        return results
