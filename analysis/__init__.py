"""
analysis package
================
Statistical tests (chi-square, t-test, ANOVA) + 8 numbered charts.

Public API:
    from analysis.statistical_analysis import DatasetAnalyser, StatisticalTests
    from analysis.statistical_analysis import ChartGenerator, AnalysisReport, TestResult
"""
from analysis.statistical_analysis import (
    DatasetAnalyser, StatisticalTests,
    ChartGenerator, AnalysisReport, TestResult,
)

__all__ = [
    "DatasetAnalyser", "StatisticalTests",
    "ChartGenerator", "AnalysisReport", "TestResult",
]
