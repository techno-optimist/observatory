"""
Corpus Pipeline - Large-Scale Text Processing for the Cultural Soliton Observatory.

Addresses peer review concern: N=15-30 samples insufficient for reliable inference.
This module enables processing N=10,000+ samples efficiently for:
- Distribution estimation
- Invariance testing
- Statistical power for publication-grade claims

Key features:
- Efficient batch processing with parallel workers
- Memory-efficient streaming for large files
- Corpus-level statistics with proper uncertainty quantification
- Stratified sampling for controlled experiments
- Corpus comparison with effect sizes
- Synthetic corpus generation for power analysis

Usage:
    pipeline = CorpusPipeline(batch_size=100, n_workers=4)
    stats = pipeline.process_file("large_corpus.jsonl")
    print(f"N={stats.n_samples}, CBR mean={stats.cbr_distribution.mean:.3f}")
"""

from __future__ import annotations

import json
import logging
import random
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from functools import partial
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Tuple,
    Union,
)

import numpy as np
from scipy import stats as scipy_stats

from .hierarchical_coordinates import (
    HierarchicalCoordinate,
    extract_hierarchical_coordinate,
    reduce_to_3d,
)
from .legibility_analyzer import (
    LegibilityRegime,
    compute_legibility_sync,
)
from .cbr_thermometer import (
    CBRThermometer,
    CBRReading,
)
from .academic_statistics import (
    EffectSize,
    cohens_d,
    hedges_g,
    bootstrap_ci,
    BootstrapEstimate,
    fisher_rao_distance,
    jensen_shannon_distance,
)


logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes for Pipeline Results
# =============================================================================

@dataclass
class ProcessedSample:
    """Single processed text sample with all extracted features."""

    text: str
    text_id: str
    coordinate: HierarchicalCoordinate
    cbr_reading: CBRReading
    legibility: float
    regime: LegibilityRegime

    # Derived metrics
    core_3d: Tuple[float, float, float] = field(default_factory=lambda: (0.0, 0.0, 0.0))
    kernel_state: int = 0

    # Metadata
    source: Optional[str] = None
    timestamp: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "text_id": self.text_id,
            "text_preview": self.text[:100] + "..." if len(self.text) > 100 else self.text,
            "coordinate": self.coordinate.to_dict(),
            "cbr_temperature": self.cbr_reading.temperature,
            "cbr_signal_strength": self.cbr_reading.signal_strength,
            "legibility": self.legibility,
            "regime": self.regime.value,
            "core_3d": {
                "agency": self.core_3d[0],
                "justice": self.core_3d[1],
                "belonging": self.core_3d[2],
            },
            "kernel_state": self.kernel_state,
            "kernel_label": self.cbr_reading.kernel_label,
            "source": self.source,
        }


@dataclass
class TestResult:
    """Statistical test result."""

    statistic: float
    p_value: float
    test_name: str
    is_significant: bool
    alpha: float = 0.05
    interpretation: str = ""

    def to_dict(self) -> dict:
        return {
            "statistic": self.statistic,
            "p_value": self.p_value,
            "test_name": self.test_name,
            "is_significant": self.is_significant,
            "alpha": self.alpha,
            "interpretation": self.interpretation,
        }


@dataclass
class Distribution:
    """Statistical distribution summary."""

    mean: float
    std: float
    variance: float
    min_val: float
    max_val: float
    median: float
    percentiles: Dict[int, float] = field(default_factory=dict)  # {5, 25, 50, 75, 95}
    skewness: float = 0.0
    kurtosis: float = 0.0
    n: int = 0

    @classmethod
    def from_array(cls, values: np.ndarray) -> "Distribution":
        """Compute distribution statistics from array."""
        if len(values) == 0:
            return cls(
                mean=0.0, std=0.0, variance=0.0,
                min_val=0.0, max_val=0.0, median=0.0,
                percentiles={}, n=0
            )

        return cls(
            mean=float(np.mean(values)),
            std=float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            variance=float(np.var(values, ddof=1)) if len(values) > 1 else 0.0,
            min_val=float(np.min(values)),
            max_val=float(np.max(values)),
            median=float(np.median(values)),
            percentiles={
                5: float(np.percentile(values, 5)),
                25: float(np.percentile(values, 25)),
                50: float(np.percentile(values, 50)),
                75: float(np.percentile(values, 75)),
                95: float(np.percentile(values, 95)),
            },
            skewness=float(scipy_stats.skew(values)) if len(values) > 2 else 0.0,
            kurtosis=float(scipy_stats.kurtosis(values)) if len(values) > 3 else 0.0,
            n=len(values),
        )

    def to_dict(self) -> dict:
        return {
            "mean": self.mean,
            "std": self.std,
            "variance": self.variance,
            "min": self.min_val,
            "max": self.max_val,
            "median": self.median,
            "percentiles": self.percentiles,
            "skewness": self.skewness,
            "kurtosis": self.kurtosis,
            "n": self.n,
        }


@dataclass
class CorpusStats:
    """Comprehensive corpus-level statistics."""

    n_samples: int

    # CBR distribution
    cbr_distribution: Distribution

    # Regime distribution (counts per regime)
    regime_distribution: Dict[str, int] = field(default_factory=dict)

    # 8-state kernel distribution
    kernel_distribution: List[int] = field(default_factory=lambda: [0] * 8)
    kernel_entropy: float = 0.0

    # Cross-sample variance in 3D space
    cross_sample_variance: float = 0.0

    # Coordinate distributions
    agency_distribution: Optional[Distribution] = None
    justice_distribution: Optional[Distribution] = None
    belonging_distribution: Optional[Distribution] = None

    # Legibility distribution
    legibility_distribution: Optional[Distribution] = None

    # Statistical tests
    normality_test: Optional[TestResult] = None
    stationarity_test: Optional[TestResult] = None
    uniformity_test: Optional[TestResult] = None

    # Processing metadata
    processing_time_seconds: float = 0.0
    samples_per_second: float = 0.0

    # Sample storage (optional, for stratified sampling)
    samples: List[ProcessedSample] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "n_samples": self.n_samples,
            "cbr_distribution": self.cbr_distribution.to_dict(),
            "regime_distribution": self.regime_distribution,
            "kernel_distribution": self.kernel_distribution,
            "kernel_entropy": self.kernel_entropy,
            "cross_sample_variance": self.cross_sample_variance,
            "agency_distribution": self.agency_distribution.to_dict() if self.agency_distribution else None,
            "justice_distribution": self.justice_distribution.to_dict() if self.justice_distribution else None,
            "belonging_distribution": self.belonging_distribution.to_dict() if self.belonging_distribution else None,
            "legibility_distribution": self.legibility_distribution.to_dict() if self.legibility_distribution else None,
            "normality_test": self.normality_test.to_dict() if self.normality_test else None,
            "stationarity_test": self.stationarity_test.to_dict() if self.stationarity_test else None,
            "uniformity_test": self.uniformity_test.to_dict() if self.uniformity_test else None,
            "processing_time_seconds": self.processing_time_seconds,
            "samples_per_second": self.samples_per_second,
        }

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"Corpus Statistics (N={self.n_samples})",
            "=" * 50,
            "",
            "CBR Distribution:",
            f"  Mean: {self.cbr_distribution.mean:.4f}",
            f"  Std:  {self.cbr_distribution.std:.4f}",
            f"  Range: [{self.cbr_distribution.min_val:.4f}, {self.cbr_distribution.max_val:.4f}]",
            "",
            "Regime Distribution:",
        ]

        total = sum(self.regime_distribution.values())
        for regime, count in sorted(self.regime_distribution.items()):
            pct = 100 * count / total if total > 0 else 0
            lines.append(f"  {regime}: {count} ({pct:.1f}%)")

        lines.extend([
            "",
            f"Kernel Entropy: {self.kernel_entropy:.4f} bits",
            f"Cross-Sample Variance: {self.cross_sample_variance:.6f}",
            "",
        ])

        if self.normality_test:
            lines.append(f"Normality Test ({self.normality_test.test_name}):")
            lines.append(f"  p-value: {self.normality_test.p_value:.4e}")
            lines.append(f"  Interpretation: {self.normality_test.interpretation}")

        lines.extend([
            "",
            f"Processing: {self.samples_per_second:.1f} samples/sec",
        ])

        return "\n".join(lines)


@dataclass
class ComparisonResult:
    """Result of comparing two corpora."""

    corpus1_n: int
    corpus2_n: int

    # Distribution tests
    cbr_ks_test: TestResult
    regime_chi2_test: TestResult
    kernel_chi2_test: TestResult

    # Effect sizes per dimension
    cbr_effect_size: EffectSize
    agency_effect_size: Optional[EffectSize] = None
    justice_effect_size: Optional[EffectSize] = None
    belonging_effect_size: Optional[EffectSize] = None

    # Distribution distances
    cbr_js_distance: float = 0.0
    kernel_js_distance: float = 0.0
    regime_js_distance: float = 0.0

    # Overall interpretation
    interpretation: str = ""

    def to_dict(self) -> dict:
        return {
            "corpus1_n": self.corpus1_n,
            "corpus2_n": self.corpus2_n,
            "cbr_ks_test": self.cbr_ks_test.to_dict(),
            "regime_chi2_test": self.regime_chi2_test.to_dict(),
            "kernel_chi2_test": self.kernel_chi2_test.to_dict(),
            "cbr_effect_size": self.cbr_effect_size.to_dict(),
            "agency_effect_size": self.agency_effect_size.to_dict() if self.agency_effect_size else None,
            "justice_effect_size": self.justice_effect_size.to_dict() if self.justice_effect_size else None,
            "belonging_effect_size": self.belonging_effect_size.to_dict() if self.belonging_effect_size else None,
            "cbr_js_distance": self.cbr_js_distance,
            "kernel_js_distance": self.kernel_js_distance,
            "regime_js_distance": self.regime_js_distance,
            "interpretation": self.interpretation,
        }


# =============================================================================
# Core Processing Functions (for parallel execution)
# =============================================================================

def _process_single_text(
    text: str,
    text_id: str = "",
    source: str = None,
) -> ProcessedSample:
    """
    Process a single text sample.

    This function is designed to be called in parallel workers.
    """
    # Create fresh thermometer for thread safety
    thermometer = CBRThermometer(window_size=1)

    # Extract coordinate
    coordinate = extract_hierarchical_coordinate(text)

    # Get CBR reading
    reading = thermometer.measure(text)

    # Get legibility
    legibility_result = compute_legibility_sync(text)
    legibility = legibility_result.get("score", 0.5) if isinstance(legibility_result, dict) else 0.5

    # Classify regime
    if legibility >= 0.85:
        regime = LegibilityRegime.NATURAL
    elif legibility >= 0.60:
        regime = LegibilityRegime.TECHNICAL
    elif legibility >= 0.30:
        regime = LegibilityRegime.COMPRESSED
    else:
        regime = LegibilityRegime.OPAQUE

    # Get 3D coordinates
    core_3d = reduce_to_3d(coordinate)

    return ProcessedSample(
        text=text,
        text_id=text_id or str(hash(text)),
        coordinate=coordinate,
        cbr_reading=reading,
        legibility=legibility,
        regime=regime,
        core_3d=core_3d,
        kernel_state=reading.kernel_state,
        source=source,
        timestamp=time.time(),
    )


def _process_batch(
    texts_with_ids: List[Tuple[str, str, Optional[str]]],
) -> List[ProcessedSample]:
    """Process a batch of texts. Each item is (text, text_id, source)."""
    results = []
    for text, text_id, source in texts_with_ids:
        try:
            sample = _process_single_text(text, text_id, source)
            results.append(sample)
        except Exception as e:
            logger.warning(f"Failed to process text {text_id}: {e}")
    return results


# =============================================================================
# Corpus Pipeline Class
# =============================================================================

class CorpusPipeline:
    """
    Efficient batch processing pipeline for large text corpora.

    Addresses the N=10,000+ requirement for statistical validity.

    Usage:
        pipeline = CorpusPipeline(batch_size=100, n_workers=4)

        # Process iterator
        for sample in pipeline.process_texts(text_iterator):
            print(sample.cbr_reading.temperature)

        # Process file with statistics
        stats = pipeline.process_file("corpus.jsonl")
        print(stats.summary())
    """

    def __init__(
        self,
        batch_size: int = 100,
        n_workers: int = 4,
        store_samples: bool = True,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ):
        """
        Initialize the corpus pipeline.

        Args:
            batch_size: Number of texts per processing batch
            n_workers: Number of parallel workers
            store_samples: Whether to store processed samples in CorpusStats
            progress_callback: Optional callback(processed, total) for progress
        """
        self.batch_size = batch_size
        self.n_workers = n_workers
        self.store_samples = store_samples
        self.progress_callback = progress_callback

        # Processing statistics
        self._total_processed = 0
        self._total_time = 0.0

    def process_texts(
        self,
        texts: Iterator[str],
        source: str = None,
    ) -> Iterator[ProcessedSample]:
        """
        Process texts from an iterator with parallel batch processing.

        Yields ProcessedSample objects as they complete.
        Memory-efficient: only keeps one batch in memory at a time.

        Args:
            texts: Iterator of text strings
            source: Optional source identifier for all texts

        Yields:
            ProcessedSample for each text
        """
        batch = []
        batch_id = 0

        for i, text in enumerate(texts):
            text_id = f"{source or 'text'}_{i}"
            batch.append((text, text_id, source))

            if len(batch) >= self.batch_size:
                yield from self._process_batch_parallel(batch)
                batch = []
                batch_id += 1

        # Process remaining texts
        if batch:
            yield from self._process_batch_parallel(batch)

    def _process_batch_parallel(
        self,
        batch: List[Tuple[str, str, Optional[str]]],
    ) -> Iterator[ProcessedSample]:
        """Process a batch using parallel workers."""
        start_time = time.time()

        if self.n_workers <= 1:
            # Single-threaded processing
            results = _process_batch(batch)
        else:
            # Split batch among workers
            chunk_size = max(1, len(batch) // self.n_workers)
            chunks = [
                batch[i:i + chunk_size]
                for i in range(0, len(batch), chunk_size)
            ]

            results = []
            with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
                futures = [
                    executor.submit(_process_batch, chunk)
                    for chunk in chunks
                ]

                for future in as_completed(futures):
                    try:
                        chunk_results = future.result()
                        results.extend(chunk_results)
                    except Exception as e:
                        logger.error(f"Worker failed: {e}")

        elapsed = time.time() - start_time
        self._total_processed += len(results)
        self._total_time += elapsed

        if self.progress_callback:
            self.progress_callback(self._total_processed, -1)

        yield from results

    def process_file(
        self,
        path: str,
        format: str = "jsonl",
        text_field: str = "text",
        limit: Optional[int] = None,
    ) -> CorpusStats:
        """
        Process a large file without loading all into memory.

        Supports:
        - jsonl: JSON lines, one object per line
        - json: JSON array of objects
        - txt: Plain text, one text per line
        - csv: CSV with text column

        Args:
            path: Path to the file
            format: File format (jsonl, json, txt, csv)
            text_field: Field name containing text (for jsonl/json/csv)
            limit: Optional limit on number of samples

        Returns:
            CorpusStats with comprehensive statistics
        """
        start_time = time.time()

        # Create appropriate loader
        if format == "jsonl":
            loader = load_jsonl(path, text_field)
        elif format == "json":
            loader = self._load_json(path, text_field)
        elif format == "txt":
            loader = load_txt(path)
        elif format == "csv":
            loader = load_csv(path, text_field)
        else:
            raise ValueError(f"Unknown format: {format}")

        # Apply limit if specified
        if limit:
            loader = self._limit_iterator(loader, limit)

        # Process and collect samples
        samples = list(self.process_texts(loader, source=path))

        # Compute statistics
        stats = self._compute_corpus_stats(samples, start_time)

        return stats

    def process_directory(
        self,
        path: str,
        pattern: str = "*.txt",
        format: str = "txt",
        text_field: str = "text",
    ) -> CorpusStats:
        """
        Process all matching files in a directory.

        Args:
            path: Directory path
            pattern: Glob pattern for files (e.g., "*.txt", "*.jsonl")
            format: File format (same as process_file)
            text_field: Field name for text (for structured formats)

        Returns:
            Combined CorpusStats for all files
        """
        start_time = time.time()
        dir_path = Path(path)

        if not dir_path.is_dir():
            raise ValueError(f"Not a directory: {path}")

        all_samples = []

        for file_path in sorted(dir_path.glob(pattern)):
            logger.info(f"Processing {file_path}")

            if format == "txt":
                loader = load_txt(str(file_path))
            elif format == "jsonl":
                loader = load_jsonl(str(file_path), text_field)
            elif format == "csv":
                loader = load_csv(str(file_path), text_field)
            else:
                raise ValueError(f"Unknown format: {format}")

            samples = list(self.process_texts(loader, source=str(file_path)))
            all_samples.extend(samples)

        return self._compute_corpus_stats(all_samples, start_time)

    def _compute_corpus_stats(
        self,
        samples: List[ProcessedSample],
        start_time: float,
    ) -> CorpusStats:
        """Compute comprehensive statistics from processed samples."""
        elapsed = time.time() - start_time
        n = len(samples)

        if n == 0:
            return CorpusStats(
                n_samples=0,
                cbr_distribution=Distribution.from_array(np.array([])),
                processing_time_seconds=elapsed,
                samples_per_second=0.0,
            )

        # Extract arrays
        cbr_temps = np.array([s.cbr_reading.temperature for s in samples])
        legibilities = np.array([s.legibility for s in samples])
        kernel_states = np.array([s.kernel_state for s in samples])

        agencies = np.array([s.core_3d[0] for s in samples])
        justices = np.array([s.core_3d[1] for s in samples])
        belongings = np.array([s.core_3d[2] for s in samples])

        # CBR distribution
        cbr_dist = Distribution.from_array(cbr_temps)

        # Regime distribution
        regime_dist = defaultdict(int)
        for s in samples:
            regime_dist[s.regime.value] += 1

        # Kernel distribution (8 states: 0-7)
        kernel_dist = [0] * 8
        for state in kernel_states:
            kernel_dist[state] += 1

        # Kernel entropy
        kernel_probs = np.array(kernel_dist) / n
        kernel_probs = kernel_probs[kernel_probs > 0]
        kernel_entropy = float(-np.sum(kernel_probs * np.log2(kernel_probs)))

        # Cross-sample variance in 3D space
        coords_3d = np.column_stack([agencies, justices, belongings])
        cross_variance = float(np.mean(np.var(coords_3d, axis=0)))

        # Coordinate distributions
        agency_dist = Distribution.from_array(agencies)
        justice_dist = Distribution.from_array(justices)
        belonging_dist = Distribution.from_array(belongings)
        legibility_dist = Distribution.from_array(legibilities)

        # Statistical tests
        normality_test = self._test_normality(cbr_temps)
        stationarity_test = self._test_stationarity(cbr_temps)
        uniformity_test = self._test_uniformity(kernel_dist)

        return CorpusStats(
            n_samples=n,
            cbr_distribution=cbr_dist,
            regime_distribution=dict(regime_dist),
            kernel_distribution=kernel_dist,
            kernel_entropy=kernel_entropy,
            cross_sample_variance=cross_variance,
            agency_distribution=agency_dist,
            justice_distribution=justice_dist,
            belonging_distribution=belonging_dist,
            legibility_distribution=legibility_dist,
            normality_test=normality_test,
            stationarity_test=stationarity_test,
            uniformity_test=uniformity_test,
            processing_time_seconds=elapsed,
            samples_per_second=n / elapsed if elapsed > 0 else 0.0,
            samples=samples if self.store_samples else [],
        )

    def _test_normality(self, values: np.ndarray) -> TestResult:
        """Test if CBR distribution is normal using Shapiro-Wilk or D'Agostino."""
        if len(values) < 8:
            return TestResult(
                statistic=0.0,
                p_value=1.0,
                test_name="insufficient_data",
                is_significant=False,
                interpretation="Insufficient data for normality test",
            )

        # Use D'Agostino-Pearson for larger samples, Shapiro-Wilk for smaller
        if len(values) > 5000:
            # Subsample for Shapiro-Wilk (limit is 5000)
            sample = np.random.choice(values, 5000, replace=False)
        else:
            sample = values

        try:
            stat, p_value = scipy_stats.shapiro(sample)
            test_name = "shapiro_wilk"
        except Exception:
            stat, p_value = scipy_stats.normaltest(values)
            test_name = "dagostino_pearson"

        is_sig = p_value < 0.05
        interp = (
            "Distribution significantly deviates from normal (p < 0.05)"
            if is_sig else
            "Distribution consistent with normal (p >= 0.05)"
        )

        return TestResult(
            statistic=float(stat),
            p_value=float(p_value),
            test_name=test_name,
            is_significant=is_sig,
            interpretation=interp,
        )

    def _test_stationarity(self, values: np.ndarray) -> TestResult:
        """Test if time series is stationary using augmented Dickey-Fuller."""
        if len(values) < 20:
            return TestResult(
                statistic=0.0,
                p_value=1.0,
                test_name="insufficient_data",
                is_significant=False,
                interpretation="Insufficient data for stationarity test",
            )

        try:
            # Simple runs test for stationarity
            # Count runs (sequences of values above/below median)
            median = np.median(values)
            above = values > median
            runs = 1 + np.sum(above[1:] != above[:-1])

            n_above = np.sum(above)
            n_below = len(values) - n_above

            # Expected runs under null
            expected = 1 + 2 * n_above * n_below / len(values)
            var = (2 * n_above * n_below * (2 * n_above * n_below - len(values))) / \
                  (len(values) ** 2 * (len(values) - 1))

            if var > 0:
                z = (runs - expected) / np.sqrt(var)
                p_value = 2 * (1 - scipy_stats.norm.cdf(abs(z)))
            else:
                z = 0.0
                p_value = 1.0

            is_sig = p_value < 0.05
            interp = (
                "Series shows significant non-stationarity (p < 0.05)"
                if is_sig else
                "Series consistent with stationarity (p >= 0.05)"
            )

            return TestResult(
                statistic=float(z),
                p_value=float(p_value),
                test_name="runs_test",
                is_significant=is_sig,
                interpretation=interp,
            )
        except Exception as e:
            return TestResult(
                statistic=0.0,
                p_value=1.0,
                test_name="runs_test_failed",
                is_significant=False,
                interpretation=f"Stationarity test failed: {e}",
            )

    def _test_uniformity(self, kernel_counts: List[int]) -> TestResult:
        """Test if kernel states are uniformly distributed."""
        observed = np.array(kernel_counts)
        n = np.sum(observed)

        if n == 0:
            return TestResult(
                statistic=0.0,
                p_value=1.0,
                test_name="chi2_uniformity",
                is_significant=False,
                interpretation="No samples to test",
            )

        expected = np.full(8, n / 8)

        # Chi-square test
        chi2, p_value = scipy_stats.chisquare(observed, expected)

        is_sig = p_value < 0.05
        interp = (
            "Kernel distribution significantly non-uniform (p < 0.05)"
            if is_sig else
            "Kernel distribution consistent with uniform (p >= 0.05)"
        )

        return TestResult(
            statistic=float(chi2),
            p_value=float(p_value),
            test_name="chi2_uniformity",
            is_significant=is_sig,
            interpretation=interp,
        )

    def _load_json(self, path: str, text_field: str) -> Iterator[str]:
        """Load texts from JSON array file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            for item in data:
                if isinstance(item, str):
                    yield item
                elif isinstance(item, dict) and text_field in item:
                    yield item[text_field]

    def _limit_iterator(self, iterator: Iterator, limit: int) -> Iterator:
        """Limit an iterator to a maximum number of items."""
        for i, item in enumerate(iterator):
            if i >= limit:
                break
            yield item


# =============================================================================
# Standard Corpus Loaders
# =============================================================================

def load_csv(path: str, text_column: str = "text") -> Iterator[str]:
    """
    Load texts from a CSV file.

    Memory-efficient: streams rows without loading entire file.

    Args:
        path: Path to CSV file
        text_column: Name of column containing text

    Yields:
        Text strings from the specified column
    """
    import csv

    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        if text_column not in reader.fieldnames:
            raise ValueError(f"Column '{text_column}' not found. Available: {reader.fieldnames}")

        for row in reader:
            text = row.get(text_column, "")
            if text and text.strip():
                yield text.strip()


def load_jsonl(path: str, text_field: str = "text") -> Iterator[str]:
    """
    Load texts from a JSON Lines file.

    Memory-efficient: streams lines without loading entire file.

    Args:
        path: Path to JSONL file
        text_field: Name of field containing text in each JSON object

    Yields:
        Text strings from the specified field
    """
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
                if isinstance(obj, str):
                    yield obj
                elif isinstance(obj, dict):
                    text = obj.get(text_field, "")
                    if text:
                        yield text
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping malformed JSON at line {line_num}: {e}")


def load_txt(path: str, min_length: int = 10) -> Iterator[str]:
    """
    Load texts from a plain text file (one text per line).

    Args:
        path: Path to text file
        min_length: Minimum character length to include

    Yields:
        Non-empty text lines
    """
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if len(text) >= min_length:
                yield text


def load_huggingface(
    dataset_name: str,
    split: str = "train",
    text_field: str = "text",
    streaming: bool = True,
) -> Iterator[str]:
    """
    Load texts from a HuggingFace dataset.

    Requires: pip install datasets

    Args:
        dataset_name: HuggingFace dataset identifier (e.g., "wikitext", "bookcorpus")
        split: Dataset split ("train", "test", "validation")
        text_field: Field name containing text
        streaming: Whether to stream (recommended for large datasets)

    Yields:
        Text strings from the dataset
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("HuggingFace datasets not installed. Run: pip install datasets")

    dataset = load_dataset(dataset_name, split=split, streaming=streaming)

    for item in dataset:
        text = item.get(text_field, "")
        if text and text.strip():
            yield text.strip()


# =============================================================================
# Stratified Sampling
# =============================================================================

def stratified_sample(
    corpus: CorpusStats,
    n_samples: int,
    strata: str = "regime",
) -> List[ProcessedSample]:
    """
    Sample from corpus while maintaining distribution of strata.

    Useful for creating balanced test sets or reducing dataset size
    while preserving distribution characteristics.

    Args:
        corpus: CorpusStats with stored samples
        n_samples: Target number of samples
        strata: Stratification variable ("regime", "kernel_state", "source")

    Returns:
        List of sampled ProcessedSample objects
    """
    if not corpus.samples:
        raise ValueError("Corpus has no stored samples. Set store_samples=True in pipeline.")

    if n_samples >= len(corpus.samples):
        return corpus.samples.copy()

    # Group samples by stratum
    groups: Dict[str, List[ProcessedSample]] = defaultdict(list)

    for sample in corpus.samples:
        if strata == "regime":
            key = sample.regime.value
        elif strata == "kernel_state":
            key = str(sample.kernel_state)
        elif strata == "source":
            key = sample.source or "unknown"
        else:
            raise ValueError(f"Unknown stratum: {strata}")

        groups[key].append(sample)

    # Compute sampling proportions
    total = len(corpus.samples)
    proportions = {k: len(v) / total for k, v in groups.items()}

    # Sample from each group
    result = []
    remaining = n_samples

    for key, samples in groups.items():
        # Number to sample from this group
        n_from_group = int(proportions[key] * n_samples)
        n_from_group = min(n_from_group, len(samples), remaining)

        if n_from_group > 0:
            selected = random.sample(samples, n_from_group)
            result.extend(selected)
            remaining -= n_from_group

    # Fill remaining with random samples
    if remaining > 0:
        all_remaining = [s for s in corpus.samples if s not in result]
        if all_remaining:
            extra = random.sample(all_remaining, min(remaining, len(all_remaining)))
            result.extend(extra)

    return result


# =============================================================================
# Corpus Comparison
# =============================================================================

def compare_corpora(
    corpus1: CorpusStats,
    corpus2: CorpusStats,
    alpha: float = 0.05,
) -> ComparisonResult:
    """
    Compare two corpora with statistical tests and effect sizes.

    Useful for:
    - Comparing pre/post intervention
    - Comparing different sources
    - Validating distribution claims

    Args:
        corpus1: First corpus statistics
        corpus2: Second corpus statistics
        alpha: Significance level for tests

    Returns:
        ComparisonResult with tests, effect sizes, and distances
    """
    # Extract sample values if available
    if corpus1.samples and corpus2.samples:
        cbr1 = np.array([s.cbr_reading.temperature for s in corpus1.samples])
        cbr2 = np.array([s.cbr_reading.temperature for s in corpus2.samples])

        agency1 = np.array([s.core_3d[0] for s in corpus1.samples])
        agency2 = np.array([s.core_3d[0] for s in corpus2.samples])

        justice1 = np.array([s.core_3d[1] for s in corpus1.samples])
        justice2 = np.array([s.core_3d[1] for s in corpus2.samples])

        belonging1 = np.array([s.core_3d[2] for s in corpus1.samples])
        belonging2 = np.array([s.core_3d[2] for s in corpus2.samples])
    else:
        # Generate synthetic samples from distributions (less accurate)
        n1 = corpus1.n_samples
        n2 = corpus2.n_samples

        cbr1 = np.random.normal(
            corpus1.cbr_distribution.mean,
            corpus1.cbr_distribution.std,
            n1
        )
        cbr2 = np.random.normal(
            corpus2.cbr_distribution.mean,
            corpus2.cbr_distribution.std,
            n2
        )

        agency1 = agency2 = justice1 = justice2 = belonging1 = belonging2 = None

    # KS test for CBR distributions
    ks_stat, ks_p = scipy_stats.ks_2samp(cbr1, cbr2)
    cbr_ks_test = TestResult(
        statistic=float(ks_stat),
        p_value=float(ks_p),
        test_name="kolmogorov_smirnov",
        is_significant=ks_p < alpha,
        alpha=alpha,
        interpretation=(
            "CBR distributions significantly different"
            if ks_p < alpha else
            "CBR distributions not significantly different"
        ),
    )

    # Chi-square test for regime distribution
    regimes = ["natural", "technical", "compressed", "opaque"]
    regime_obs1 = [corpus1.regime_distribution.get(r, 0) for r in regimes]
    regime_obs2 = [corpus2.regime_distribution.get(r, 0) for r in regimes]

    # Combine into contingency table
    contingency = np.array([regime_obs1, regime_obs2])

    # Filter out columns with all zeros
    nonzero_cols = np.any(contingency > 0, axis=0)
    contingency = contingency[:, nonzero_cols]

    if contingency.shape[1] > 1:
        chi2, chi2_p, _, _ = scipy_stats.chi2_contingency(contingency)
    else:
        chi2, chi2_p = 0.0, 1.0

    regime_chi2_test = TestResult(
        statistic=float(chi2),
        p_value=float(chi2_p),
        test_name="chi2_contingency",
        is_significant=chi2_p < alpha,
        alpha=alpha,
        interpretation=(
            "Regime distributions significantly different"
            if chi2_p < alpha else
            "Regime distributions not significantly different"
        ),
    )

    # Chi-square test for kernel distribution
    kernel1 = np.array(corpus1.kernel_distribution) + 0.5  # Add smoothing
    kernel2 = np.array(corpus2.kernel_distribution) + 0.5

    kernel_contingency = np.array([kernel1, kernel2])
    chi2_k, chi2_p_k, _, _ = scipy_stats.chi2_contingency(kernel_contingency)

    kernel_chi2_test = TestResult(
        statistic=float(chi2_k),
        p_value=float(chi2_p_k),
        test_name="chi2_contingency",
        is_significant=chi2_p_k < alpha,
        alpha=alpha,
        interpretation=(
            "Kernel distributions significantly different"
            if chi2_p_k < alpha else
            "Kernel distributions not significantly different"
        ),
    )

    # Effect sizes
    cbr_effect = hedges_g(cbr1, cbr2) if len(cbr1) + len(cbr2) < 50 else cohens_d(cbr1, cbr2)

    agency_effect = None
    justice_effect = None
    belonging_effect = None

    if agency1 is not None and agency2 is not None:
        agency_effect = cohens_d(agency1, agency2)
        justice_effect = cohens_d(justice1, justice2)
        belonging_effect = cohens_d(belonging1, belonging2)

    # Jensen-Shannon distances
    # CBR: create histograms
    all_cbr = np.concatenate([cbr1, cbr2])
    bins = np.linspace(np.min(all_cbr), np.max(all_cbr), 20)
    hist1, _ = np.histogram(cbr1, bins=bins, density=True)
    hist2, _ = np.histogram(cbr2, bins=bins, density=True)
    hist1 = hist1 + 1e-10
    hist2 = hist2 + 1e-10
    hist1 = hist1 / hist1.sum()
    hist2 = hist2 / hist2.sum()
    cbr_js = jensen_shannon_distance(hist1, hist2)

    # Kernel JS distance
    kernel1_norm = kernel1 / kernel1.sum()
    kernel2_norm = kernel2 / kernel2.sum()
    kernel_js = jensen_shannon_distance(kernel1_norm, kernel2_norm)

    # Regime JS distance
    regime1_norm = np.array(regime_obs1) + 0.5
    regime2_norm = np.array(regime_obs2) + 0.5
    regime1_norm = regime1_norm / regime1_norm.sum()
    regime2_norm = regime2_norm / regime2_norm.sum()
    regime_js = jensen_shannon_distance(regime1_norm, regime2_norm)

    # Overall interpretation
    n_sig_tests = sum([
        cbr_ks_test.is_significant,
        regime_chi2_test.is_significant,
        kernel_chi2_test.is_significant,
    ])

    if n_sig_tests == 0:
        interpretation = "Corpora are statistically similar across all dimensions."
    elif n_sig_tests == 1:
        interpretation = "Corpora show minor differences (1/3 tests significant)."
    elif n_sig_tests == 2:
        interpretation = "Corpora show moderate differences (2/3 tests significant)."
    else:
        interpretation = "Corpora are substantially different (all tests significant)."

    # Add effect size interpretation
    if abs(cbr_effect.d) >= 0.8:
        interpretation += f" CBR effect is large (d={cbr_effect.d:.2f})."
    elif abs(cbr_effect.d) >= 0.5:
        interpretation += f" CBR effect is medium (d={cbr_effect.d:.2f})."
    elif abs(cbr_effect.d) >= 0.2:
        interpretation += f" CBR effect is small (d={cbr_effect.d:.2f})."

    return ComparisonResult(
        corpus1_n=corpus1.n_samples,
        corpus2_n=corpus2.n_samples,
        cbr_ks_test=cbr_ks_test,
        regime_chi2_test=regime_chi2_test,
        kernel_chi2_test=kernel_chi2_test,
        cbr_effect_size=cbr_effect,
        agency_effect_size=agency_effect,
        justice_effect_size=justice_effect,
        belonging_effect_size=belonging_effect,
        cbr_js_distance=float(cbr_js),
        kernel_js_distance=float(kernel_js),
        regime_js_distance=float(regime_js),
        interpretation=interpretation,
    )


# =============================================================================
# Synthetic Corpus Generation
# =============================================================================

# Template texts for each regime
REGIME_TEMPLATES = {
    "natural": [
        "I think we should work together on this project because it benefits everyone.",
        "The team decided to share the resources fairly among all members.",
        "We believe in treating each person with dignity and respect.",
        "Our community came together to help those who needed support.",
        "Everyone deserves a fair chance to participate in the decision.",
        "I feel strongly that we need to listen to each other more.",
        "The family gathered to celebrate the achievement together.",
        "People from different backgrounds united for a common cause.",
        "She explained her perspective clearly so others could understand.",
        "We made sure the process was transparent for all involved.",
    ],
    "technical": [
        "The algorithm optimizes the objective function using gradient descent.",
        "System architecture implements microservices with REST API endpoints.",
        "Protocol specification defines message format and error handling.",
        "Analysis indicates significant correlation between variables X and Y.",
        "Implementation requires consideration of edge cases and boundary conditions.",
        "The framework provides abstraction layers for database operations.",
        "Metrics demonstrate improvement in latency and throughput performance.",
        "Documentation specifies interface contracts and dependency requirements.",
        "Configuration parameters control runtime behavior and resource allocation.",
        "Testing methodology includes unit, integration, and end-to-end scenarios.",
    ],
    "compressed": [
        "opt cfg: batch=100, lr=0.001, epochs=50",
        "req: auth, validate, process, respond",
        "status: OK | error: null | latency: 12ms",
        "params={k:v for k,v in defaults.items()}",
        "input -> transform -> aggregate -> output",
        "check: A && B || !C => result",
        "flow: init > load > process > save",
        "spec: type=string, min=1, max=255",
        "action: GET /api/v2/resource?filter=active",
        "result: {success: true, count: 42}",
    ],
    "opaque": [
        "xkcd://proto.sync.node.0x7f",
        "{{$recursive.expand($ctx)}}",
        "~~~[signal:undefined]~~~",
        "@@@###$$$%%%^^^&&&***",
        "null.ref.cascade.terminal",
        "..........!!!!!!!??????",
        ">>>>>>>>>>||||||<<<<<<",
        "0000000000011111111111",
        "????????????????????????",
        "~~~~~~~~~~~~------------",
    ],
}

# Kernel state templates (based on presence/absence of agency, justice, belonging)
KERNEL_TEMPLATES = {
    0: [  # ANOMIE: 000 - no agency, no justice, no belonging
        "Things happen without anyone deciding.",
        "Events occur in sequence without pattern.",
        "The process continues indefinitely.",
    ],
    1: [  # OPPRESSION: 001 - no agency, no justice, belonging
        "We are all affected by forces beyond our control.",
        "Our group faces circumstances none of us chose.",
        "Together we experience what happens to us.",
    ],
    2: [  # NEGLECT: 010 - no agency, justice, no belonging
        "Fair procedures determine the outcome automatically.",
        "The rules apply equally without anyone deciding.",
        "Due process ensures proper distribution.",
    ],
    3: [  # DEPENDENCE: 011 - no agency, justice, belonging
        "Our community relies on fair systems to protect us.",
        "Together we trust that proper procedures will help.",
        "We as a group depend on just institutions.",
    ],
    4: [  # ALIENATION: 100 - agency, no justice, no belonging
        "I make my own choices regardless of fairness.",
        "She decided her path without concern for others.",
        "They acted on their own terms alone.",
    ],
    5: [  # EXPLOITATION: 101 - agency, no justice, belonging
        "We decided together, though not fairly for all.",
        "Our group chose this path, excluding others.",
        "I acted for my people, not for justice.",
    ],
    6: [  # AUTONOMY: 110 - agency, justice, no belonging
        "I chose fairly, on my own principles.",
        "She made a just decision independently.",
        "They acted with integrity, alone.",
    ],
    7: [  # COORDINATION: 111 - agency, justice, belonging
        "We decided together to ensure fairness for everyone.",
        "Our community chose a just path that includes all.",
        "I acted with my group to create equity for all.",
    ],
}


def generate_test_corpus(
    n_samples: int,
    regime_distribution: Optional[Dict[str, float]] = None,
    kernel_distribution: Optional[List[float]] = None,
    seed: Optional[int] = None,
) -> List[str]:
    """
    Generate synthetic texts for testing and power analysis.

    Useful for:
    - Testing pipeline at scale
    - Power analysis for study design
    - Validating statistical methods
    - Creating baseline comparisons

    Args:
        n_samples: Number of texts to generate
        regime_distribution: Target distribution as {regime: proportion}
                           Default: uniform across regimes
        kernel_distribution: Target distribution as list of 8 proportions (sums to 1)
                           Default: uniform across kernel states
        seed: Random seed for reproducibility

    Returns:
        List of generated text strings
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # Default to uniform distributions
    if regime_distribution is None:
        regime_distribution = {
            "natural": 0.25,
            "technical": 0.25,
            "compressed": 0.25,
            "opaque": 0.25,
        }

    if kernel_distribution is None:
        kernel_distribution = [0.125] * 8

    # Normalize distributions
    regime_total = sum(regime_distribution.values())
    regime_distribution = {k: v / regime_total for k, v in regime_distribution.items()}

    kernel_total = sum(kernel_distribution)
    kernel_distribution = [k / kernel_total for k in kernel_distribution]

    texts = []

    for _ in range(n_samples):
        # Sample regime
        regime = np.random.choice(
            list(regime_distribution.keys()),
            p=list(regime_distribution.values())
        )

        # Sample kernel state
        kernel_state = np.random.choice(8, p=kernel_distribution)

        # Generate text combining regime style and kernel content
        # Use regime template as base
        base_templates = REGIME_TEMPLATES.get(regime, REGIME_TEMPLATES["natural"])
        base_text = random.choice(base_templates)

        # Add kernel-specific content
        kernel_templates = KERNEL_TEMPLATES.get(kernel_state, KERNEL_TEMPLATES[7])
        kernel_text = random.choice(kernel_templates)

        # Combine with variation
        if regime in ["compressed", "opaque"]:
            # Keep compressed/opaque texts shorter
            text = base_text
        else:
            # Combine for natural/technical
            if random.random() < 0.5:
                text = f"{base_text} {kernel_text}"
            else:
                text = f"{kernel_text} {base_text}"

        # Add random variation
        text = _add_text_variation(text)

        texts.append(text)

    return texts


def _add_text_variation(text: str) -> str:
    """Add random variation to generated text."""
    variations = [
        lambda t: t,  # No change
        lambda t: t.replace(".", "!"),  # Excitement
        lambda t: t.replace(".", "..."),  # Trailing off
        lambda t: t.lower(),  # Lowercase
        lambda t: t.upper(),  # Uppercase
        lambda t: " ".join(t.split()),  # Normalize spaces
    ]

    variation = random.choice(variations)
    return variation(text)


def generate_power_analysis_corpus(
    effect_sizes: List[float] = [0.2, 0.5, 0.8],
    sample_sizes: List[int] = [100, 500, 1000, 5000],
    n_simulations: int = 100,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Generate corpora for statistical power analysis.

    Helps determine required sample size for detecting effects.

    Args:
        effect_sizes: Target effect sizes (Cohen's d) to test
        sample_sizes: Sample sizes to evaluate
        n_simulations: Number of simulations per condition
        seed: Random seed

    Returns:
        Dictionary with power estimates for each effect size and sample size
    """
    if seed is not None:
        np.random.seed(seed)

    results = {
        "effect_sizes": effect_sizes,
        "sample_sizes": sample_sizes,
        "power_matrix": [],  # [effect_size][sample_size] = power
    }

    for d in effect_sizes:
        power_row = []

        for n in sample_sizes:
            # Simulate detecting effect of size d with sample size n
            detections = 0

            for _ in range(n_simulations):
                # Generate two groups with known effect
                group1 = np.random.normal(0, 1, n)
                group2 = np.random.normal(d, 1, n)  # Shifted by effect size

                # Test for significant difference
                _, p_value = scipy_stats.ttest_ind(group1, group2)

                if p_value < 0.05:
                    detections += 1

            power = detections / n_simulations
            power_row.append(power)

        results["power_matrix"].append(power_row)

    # Generate recommendations
    recommendations = []
    for i, d in enumerate(effect_sizes):
        for j, n in enumerate(sample_sizes):
            power = results["power_matrix"][i][j]
            if power >= 0.8:
                recommendations.append({
                    "effect_size": d,
                    "min_sample_size": n,
                    "power": power,
                })
                break

    results["recommendations"] = recommendations

    return results


# =============================================================================
# Example Usage and Testing
# =============================================================================

def example_usage():
    """
    Demonstrate corpus pipeline with 1000+ synthetic samples.

    This example shows:
    1. Generating synthetic test corpus
    2. Processing with parallel pipeline
    3. Computing full statistics
    4. Comparing two corpora
    """
    print("=" * 70)
    print("CORPUS PIPELINE DEMONSTRATION")
    print("Processing N=1000+ samples for statistical validity")
    print("=" * 70)
    print()

    # 1. Generate synthetic corpora
    print("1. Generating synthetic corpora...")

    # Corpus 1: Balanced distribution
    corpus1_texts = generate_test_corpus(
        n_samples=1200,
        regime_distribution={
            "natural": 0.40,
            "technical": 0.35,
            "compressed": 0.15,
            "opaque": 0.10,
        },
        kernel_distribution=[0.05, 0.10, 0.10, 0.15, 0.10, 0.15, 0.15, 0.20],
        seed=42,
    )
    print(f"   Generated {len(corpus1_texts)} texts for Corpus 1 (balanced)")

    # Corpus 2: Skewed distribution (more technical/compressed)
    corpus2_texts = generate_test_corpus(
        n_samples=1000,
        regime_distribution={
            "natural": 0.20,
            "technical": 0.45,
            "compressed": 0.25,
            "opaque": 0.10,
        },
        kernel_distribution=[0.10, 0.10, 0.15, 0.10, 0.15, 0.15, 0.15, 0.10],
        seed=123,
    )
    print(f"   Generated {len(corpus2_texts)} texts for Corpus 2 (technical-heavy)")
    print()

    # 2. Process with pipeline
    print("2. Processing corpora with parallel pipeline...")

    pipeline = CorpusPipeline(
        batch_size=100,
        n_workers=4,
        store_samples=True,
        progress_callback=lambda p, t: print(f"\r   Processed: {p}", end="", flush=True)
    )

    # Process corpus 1
    print("\n   Processing Corpus 1...")
    stats1 = pipeline._compute_corpus_stats(
        list(pipeline.process_texts(iter(corpus1_texts), source="corpus1")),
        time.time()
    )
    print(f"\n   Corpus 1: {stats1.n_samples} samples in {stats1.processing_time_seconds:.2f}s")

    # Reset for corpus 2
    pipeline._total_processed = 0
    pipeline._total_time = 0.0

    # Process corpus 2
    print("   Processing Corpus 2...")
    stats2 = pipeline._compute_corpus_stats(
        list(pipeline.process_texts(iter(corpus2_texts), source="corpus2")),
        time.time()
    )
    print(f"\n   Corpus 2: {stats2.n_samples} samples in {stats2.processing_time_seconds:.2f}s")
    print()

    # 3. Display statistics
    print("3. Corpus Statistics:")
    print()
    print("CORPUS 1 (Balanced Distribution)")
    print("-" * 40)
    print(stats1.summary())
    print()

    print("CORPUS 2 (Technical-Heavy Distribution)")
    print("-" * 40)
    print(stats2.summary())
    print()

    # 4. Compare corpora
    print("4. Corpus Comparison:")
    print("-" * 40)

    comparison = compare_corpora(stats1, stats2)

    print(f"Sample sizes: N1={comparison.corpus1_n}, N2={comparison.corpus2_n}")
    print()
    print("Statistical Tests:")
    print(f"  CBR KS Test: stat={comparison.cbr_ks_test.statistic:.4f}, p={comparison.cbr_ks_test.p_value:.4e}")
    print(f"    {comparison.cbr_ks_test.interpretation}")
    print(f"  Regime Chi2: stat={comparison.regime_chi2_test.statistic:.4f}, p={comparison.regime_chi2_test.p_value:.4e}")
    print(f"    {comparison.regime_chi2_test.interpretation}")
    print(f"  Kernel Chi2: stat={comparison.kernel_chi2_test.statistic:.4f}, p={comparison.kernel_chi2_test.p_value:.4e}")
    print(f"    {comparison.kernel_chi2_test.interpretation}")
    print()
    print("Effect Sizes:")
    print(f"  CBR: d={comparison.cbr_effect_size.d:.4f} ({comparison.cbr_effect_size.feature_classification})")
    print(f"       95% CI: [{comparison.cbr_effect_size.confidence_interval[0]:.4f}, {comparison.cbr_effect_size.confidence_interval[1]:.4f}]")
    if comparison.agency_effect_size:
        print(f"  Agency: d={comparison.agency_effect_size.d:.4f}")
        print(f"  Justice: d={comparison.justice_effect_size.d:.4f}")
        print(f"  Belonging: d={comparison.belonging_effect_size.d:.4f}")
    print()
    print("Distribution Distances (Jensen-Shannon):")
    print(f"  CBR: {comparison.cbr_js_distance:.4f}")
    print(f"  Kernel: {comparison.kernel_js_distance:.4f}")
    print(f"  Regime: {comparison.regime_js_distance:.4f}")
    print()
    print("Overall:")
    print(f"  {comparison.interpretation}")
    print()

    # 5. Stratified sampling demonstration
    print("5. Stratified Sampling:")
    print("-" * 40)

    # Get 200 samples maintaining regime distribution
    stratified = stratified_sample(stats1, n_samples=200, strata="regime")

    # Compute regime distribution of sample
    sample_regimes = defaultdict(int)
    for s in stratified:
        sample_regimes[s.regime.value] += 1

    print(f"   Original corpus: {stats1.n_samples} samples")
    print(f"   Stratified sample: {len(stratified)} samples")
    print()
    print("   Regime distribution comparison:")
    for regime in ["natural", "technical", "compressed", "opaque"]:
        orig_pct = 100 * stats1.regime_distribution.get(regime, 0) / stats1.n_samples
        samp_pct = 100 * sample_regimes.get(regime, 0) / len(stratified)
        print(f"     {regime}: Original={orig_pct:.1f}%, Sample={samp_pct:.1f}%")
    print()

    # 6. Power analysis
    print("6. Power Analysis (for study design):")
    print("-" * 40)

    power_results = generate_power_analysis_corpus(
        effect_sizes=[0.2, 0.5, 0.8],
        sample_sizes=[50, 100, 200, 500, 1000],
        n_simulations=50,  # Use fewer simulations for demo
        seed=42,
    )

    print("   Statistical power to detect effects:")
    print()
    print("   Sample Size |  d=0.2  |  d=0.5  |  d=0.8")
    print("   " + "-" * 45)

    for j, n in enumerate(power_results["sample_sizes"]):
        powers = [power_results["power_matrix"][i][j] for i in range(3)]
        print(f"   {n:>10} | {powers[0]:>6.2f} | {powers[1]:>6.2f} | {powers[2]:>6.2f}")

    print()
    print("   Recommendations (power >= 0.80):")
    for rec in power_results["recommendations"]:
        print(f"     Effect d={rec['effect_size']}: min N={rec['min_sample_size']} (power={rec['power']:.2f})")

    print()
    print("=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)

    return {
        "corpus1_stats": stats1,
        "corpus2_stats": stats2,
        "comparison": comparison,
        "power_analysis": power_results,
    }


if __name__ == "__main__":
    # Run example when executed directly
    results = example_usage()
