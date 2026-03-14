"""
preprocessing package
=====================
File loading + 5-stage preprocessing pipeline for Llama-3 8B fine-tuning.

Public API:
    from preprocessing.file_loader  import load_file, load_directory, LoadedDocument
    from preprocessing.preprocessor import PreprocessingPipeline, ProcessedChunk, PipelineStats
"""
from preprocessing.file_loader  import load_file, load_directory, LoadedDocument
from preprocessing.preprocessor import PreprocessingPipeline, ProcessedChunk, PipelineStats

__all__ = [
    "load_file", "load_directory", "LoadedDocument",
    "PreprocessingPipeline", "ProcessedChunk", "PipelineStats",
]
