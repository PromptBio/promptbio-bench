from pathlib import Path
from typing import List, Optional, Union

from utils.agents import create_llm_agent
from utils.models import FileMappingResult


def match_files_with_llm(
    reference_files: List[str],
    candidate_files: List[str],
    question: Optional[str] = None,
    input_files: Optional[Union[str, List[str]]] = None,
    model: str = "gpt-5.4"
) -> Optional[FileMappingResult]:
    """
    Use LLM to match candidate files to reference answer files.

    Args:
        reference_files: List of reference answer file paths
        candidate_files: List of candidate file paths
        question: Optional task question/description
        input_files: Optional input file paths; accepts a string or list of strings
        model: LLM model to use

    Returns:
        FileMappingResult with mappings, or None if error
    """
    if not reference_files or not candidate_files:
        return None

    if isinstance(input_files, list):
        input_files = "; ".join(input_files)

    system_prompt = """You are an expert in bioinformatics file analysis and matching. Your task is to match candidate output files to reference answer files based on:

1. **File naming patterns**: Similar base names, suffixes, or extensions
2. **File format (inferred from path)**: Use the file path to infer format (e.g. .fasta, .bam, .vcf.gz, .csv, .tsv, .png). Prefer matching reference and candidate files that have the same or compatible format/extension
3. **Task context**: What the task is trying to accomplish
4. **Input files**: Which input files generated which outputs
5. **Biological conventions**: Standard naming conventions in bioinformatics

MATCHING RULES:
- Each reference file should be mapped to AT MOST ONE candidate file
- If a candidate file doesn't match any reference file well, set candidate_file to None
- If there are multiple plausible matches, choose the one with highest confidence
- Consider that agent tools might:
  * Use different naming conventions
  * Add timestamps or additional metadata to filenames
  * Create intermediate files that don't correspond to reference outputs
  * Generate files in different orders

CONFIDENCE SCORING:
- 0.9-1.0: Strong match (same or very similar name, format from path agrees, clear correspondence)
- 0.7-0.9: Good match (similar naming pattern, format from path agrees, plausible correspondence)
- 0.5-0.7: Moderate match (format agrees or compatible, plausible correspondence)
- 0.3-0.5: Weak match (uncertain or format mismatch, may need verification)
- 0.0-0.3: Very uncertain match

Provide clear reasoning for each mapping decision, including the format you inferred from each path when relevant."""

    # Format the file lists
    ref_files_formatted = "\n".join([f"  - {f}" for f in reference_files])
    candidate_files_formatted = "\n".join([f"  - {f}" for f in candidate_files])

    optional_sections = ""
    if question:
        optional_sections += f"Task/Question: {question}\n\n"
    if input_files:
        optional_sections += f"Input Files: {input_files}\n\n"

    user_prompt = f"""{optional_sections}Reference Answer Files:
{ref_files_formatted}

Agent-Generated Files:
{candidate_files_formatted}

Please match each reference file to the most appropriate candidate file. If no good match exists for a reference file, set candidate_file to null."""

    try:
        agent = create_llm_agent(
            model=model,
            system_prompt=system_prompt,
            response_format=FileMappingResult,
            temperature=0.0
        )
        
        result = agent.invoke({"messages": [{"role": "user", "content": user_prompt}]})
        data = result.get("structured_response", result) if isinstance(result, dict) else result
        
        if isinstance(data, FileMappingResult):
            return data
        elif isinstance(data, dict):
            return FileMappingResult.model_validate(data)
        else:
            return None
            
    except Exception as e:
        print(f"Error matching files: {str(e)}")
        return None


def match_output_files(
    reference_dir: Optional[Path] = None,
    candidate_dir: Optional[Path] = None,
    reference_files: Optional[List[str]] = None,
    candidate_files: Optional[List[str]] = None,
    question: Optional[str] = None,
    input_files: Optional[Union[str, List[str]]] = None,
    ignore_files: Union[str, List[str]] = ["docker_log.txt", "data_description.json"],
    model: str = "gpt-5.4",
) -> Optional[FileMappingResult]:
    """
    Match candidate files against reference answer files using LLM.

    Each of reference and candidate can be provided as either an explicit file list
    or a directory to scan.

    Args:
        reference_dir: Directory containing reference answer files
        candidate_dir: Directory containing candidate (agent-generated) files
        reference_files: Explicit list of reference file paths (alternative to reference_dir)
        candidate_files: Explicit list of candidate file paths (alternative to candidate_dir)
        question: Optional task question/description for LLM context
        input_files: Optional input file paths; accepts a semicolon-separated
                     string or a list of strings
        ignore_files: File names to exclude when scanning candidate_dir; accepts
                      a single filename or a list of filenames. Defaults to known
                      tool artifacts (docker_log.txt, data_description.json)
        model: LLM model to use

    Returns:
        FileMappingResult with mappings, or None if matching failed
    """
    if reference_dir is None and reference_files is None:
        raise ValueError("Either reference_dir or reference_files must be provided")
    if reference_dir is not None and reference_files is not None:
        raise ValueError("Provide only one of reference_dir or reference_files, not both")
    if candidate_dir is None and candidate_files is None:
        raise ValueError("Either candidate_dir or candidate_files must be provided")
    if candidate_dir is not None and candidate_files is not None:
        raise ValueError("Provide only one of candidate_dir or candidate_files, not both")

    excluded = {ignore_files} if isinstance(ignore_files, str) else set(ignore_files)

    if reference_files is None:
        reference_files = [str(p) for p in sorted(Path(reference_dir).iterdir()) if p.is_file()]

    if candidate_files is None:
        candidate_files = [
            str(p) for p in sorted(Path(candidate_dir).iterdir())
            if p.is_file() and p.name not in excluded
        ]

    return match_files_with_llm(
        reference_files=reference_files,
        candidate_files=candidate_files,
        question=question,
        input_files=input_files,
        model=model,
    )

