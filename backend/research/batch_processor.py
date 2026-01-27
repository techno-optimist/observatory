"""
Batch Processor for Cultural Soliton Observatory.

Scalable processing infrastructure for N=10,000+ samples.
Handles parallel extraction, memory-efficient streaming, and progress tracking.

Author: Cultural Soliton Observatory Team
Version: 2.0.0
"""

import asyncio
import logging
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Tuple,
    Union,
)
import json
import csv

import numpy as np

from .telescope import Telescope, TelescopeConfig, ObservationResult
from .hierarchical_coordinates import HierarchicalCoordinate
from .cbr_thermometer import CBRThermometer, measure_cbr_batch

logger = logging.getLogger(__name__)


class ProcessingStrategy(Enum):
    """Processing strategy for batch operations."""
    SEQUENTIAL = "sequential"      # Simple loop, lowest memory
    THREADED = "threaded"          # Thread pool for I/O bound
    MULTIPROCESS = "multiprocess"  # Process pool for CPU bound
    STREAMING = "streaming"        # Generator-based, constant memory


@dataclass
class BatchConfig:
    """Configuration for batch processing."""
    batch_size: int = 100
    max_workers: int = 4
    strategy: ProcessingStrategy = ProcessingStrategy.THREADED
    extraction_method: str = "regex"

    # Memory management
    max_memory_mb: int = 1000
    checkpoint_interval: int = 1000
    checkpoint_path: Optional[str] = None

    # Progress tracking
    progress_callback: Optional[Callable[[int, int, float], None]] = None
    log_interval: int = 100


@dataclass
class BatchResult:
    """Result of batch processing."""
    total_processed: int
    successful: int
    failed: int
    processing_time_seconds: float
    samples_per_second: float

    # Aggregate statistics
    mean_agency: float = 0.0
    mean_justice: float = 0.0
    mean_belonging: float = 0.0
    std_agency: float = 0.0
    std_justice: float = 0.0
    std_belonging: float = 0.0

    # CBR statistics
    mean_temperature: float = 0.0
    mean_signal_strength: float = 0.0
    phase_distribution: Dict[str, int] = field(default_factory=dict)
    kernel_distribution: Dict[str, int] = field(default_factory=dict)

    # Per-sample results (optional)
    observations: List[ObservationResult] = field(default_factory=list)
    errors: List[Tuple[int, str]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": {
                "total_processed": self.total_processed,
                "successful": self.successful,
                "failed": self.failed,
                "success_rate": self.successful / max(self.total_processed, 1),
                "processing_time_seconds": self.processing_time_seconds,
                "samples_per_second": self.samples_per_second,
            },
            "coordinate_statistics": {
                "agency": {"mean": self.mean_agency, "std": self.std_agency},
                "justice": {"mean": self.mean_justice, "std": self.std_justice},
                "belonging": {"mean": self.mean_belonging, "std": self.std_belonging},
            },
            "cbr_statistics": {
                "mean_temperature": self.mean_temperature,
                "mean_signal_strength": self.mean_signal_strength,
                "phase_distribution": self.phase_distribution,
                "kernel_distribution": self.kernel_distribution,
            },
            "errors": self.errors[:10] if self.errors else [],
        }


class BatchProcessor:
    """
    Scalable batch processor for the Cultural Soliton Observatory.

    Handles N=10,000+ samples with:
    - Multiple processing strategies (sequential, threaded, multiprocess, streaming)
    - Memory-efficient streaming for very large datasets
    - Checkpointing for long-running jobs
    - Progress tracking and ETA estimation

    Example:
        processor = BatchProcessor(config=BatchConfig(batch_size=500))

        # Process from list
        result = processor.process(texts)
        print(f"Processed {result.total_processed} in {result.processing_time_seconds:.1f}s")

        # Stream from file (memory efficient)
        for batch_result in processor.stream_from_file("large_corpus.jsonl"):
            print(f"Batch done: {batch_result.successful} samples")
    """

    def __init__(self, config: Optional[BatchConfig] = None):
        self.config = config or BatchConfig()
        self.telescope = Telescope(extraction_method=self.config.extraction_method)
        self._checkpoint_data: Dict[str, Any] = {}

    def process(
        self,
        texts: List[str],
        return_observations: bool = False,
    ) -> BatchResult:
        """
        Process a list of texts.

        Args:
            texts: List of texts to process
            return_observations: If True, include per-sample ObservationResults

        Returns:
            BatchResult with statistics and optionally per-sample results
        """
        start_time = time.time()
        n_total = len(texts)

        if n_total == 0:
            return BatchResult(
                total_processed=0,
                successful=0,
                failed=0,
                processing_time_seconds=0.0,
                samples_per_second=0.0,
            )

        logger.info(f"Starting batch processing of {n_total} samples with strategy={self.config.strategy.value}")

        # Choose strategy
        if self.config.strategy == ProcessingStrategy.SEQUENTIAL:
            observations, errors = self._process_sequential(texts)
        elif self.config.strategy == ProcessingStrategy.THREADED:
            observations, errors = self._process_threaded(texts)
        elif self.config.strategy == ProcessingStrategy.MULTIPROCESS:
            observations, errors = self._process_multiprocess(texts)
        else:
            observations, errors = self._process_sequential(texts)

        processing_time = time.time() - start_time

        # Compute statistics
        result = self._compute_statistics(
            observations=observations,
            errors=errors,
            processing_time=processing_time,
            return_observations=return_observations,
        )

        logger.info(f"Batch complete: {result.successful}/{result.total_processed} successful "
                   f"in {processing_time:.1f}s ({result.samples_per_second:.1f} samples/sec)")

        return result

    def _process_sequential(
        self,
        texts: List[str],
    ) -> Tuple[List[ObservationResult], List[Tuple[int, str]]]:
        """Process texts sequentially."""
        observations = []
        errors = []

        for i, text in enumerate(texts):
            try:
                obs = self.telescope.observe(text)
                observations.append(obs)
            except Exception as e:
                errors.append((i, str(e)))
                logger.debug(f"Error processing text {i}: {e}")

            if self.config.progress_callback and (i + 1) % self.config.log_interval == 0:
                self.config.progress_callback(i + 1, len(texts), (i + 1) / len(texts))

            if self.config.checkpoint_path and (i + 1) % self.config.checkpoint_interval == 0:
                self._save_checkpoint(i + 1, observations, errors)

        return observations, errors

    def _process_threaded(
        self,
        texts: List[str],
    ) -> Tuple[List[ObservationResult], List[Tuple[int, str]]]:
        """Process texts using thread pool."""
        observations = [None] * len(texts)
        errors = []
        completed = [0]

        def process_one(args: Tuple[int, str]) -> Tuple[int, Optional[ObservationResult], Optional[str]]:
            idx, text = args
            try:
                obs = self.telescope.observe(text)
                return (idx, obs, None)
            except Exception as e:
                return (idx, None, str(e))

        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = executor.map(process_one, enumerate(texts), chunksize=self.config.batch_size)

            for idx, obs, error in futures:
                if obs is not None:
                    observations[idx] = obs
                else:
                    errors.append((idx, error))

                completed[0] += 1
                if self.config.progress_callback and completed[0] % self.config.log_interval == 0:
                    self.config.progress_callback(completed[0], len(texts), completed[0] / len(texts))

        return [o for o in observations if o is not None], errors

    def _process_multiprocess(
        self,
        texts: List[str],
    ) -> Tuple[List[ObservationResult], List[Tuple[int, str]]]:
        """Process texts using process pool (CPU-bound optimization)."""
        # For multiprocessing, we need to use picklable functions
        # Process in batches and aggregate
        observations = []
        errors = []

        # Split into batches
        batches = [
            texts[i:i + self.config.batch_size]
            for i in range(0, len(texts), self.config.batch_size)
        ]

        with ProcessPoolExecutor(max_workers=self.config.max_workers) as executor:
            batch_results = list(executor.map(_process_batch_standalone, batches))

            offset = 0
            for batch_obs, batch_errors in batch_results:
                observations.extend(batch_obs)
                errors.extend([(idx + offset, err) for idx, err in batch_errors])
                offset += self.config.batch_size

                if self.config.progress_callback:
                    self.config.progress_callback(len(observations), len(texts), len(observations) / len(texts))

        return observations, errors

    def stream_process(
        self,
        texts: Iterator[str],
        total_hint: Optional[int] = None,
    ) -> Iterator[ObservationResult]:
        """
        Stream process texts with constant memory usage.

        Args:
            texts: Iterator of texts
            total_hint: Optional hint for total count (for progress)

        Yields:
            ObservationResult for each text
        """
        processed = 0

        for text in texts:
            try:
                yield self.telescope.observe(text)
            except Exception as e:
                logger.debug(f"Error in stream processing: {e}")
                continue

            processed += 1
            if self.config.progress_callback and processed % self.config.log_interval == 0:
                progress = processed / total_hint if total_hint else 0
                self.config.progress_callback(processed, total_hint or 0, progress)

    def stream_from_file(
        self,
        filepath: Union[str, Path],
        text_field: str = "text",
    ) -> Iterator[BatchResult]:
        """
        Stream process from a JSONL or CSV file.

        Yields BatchResult for each batch (memory efficient).

        Args:
            filepath: Path to JSONL or CSV file
            text_field: Field name containing text

        Yields:
            BatchResult for each processed batch
        """
        filepath = Path(filepath)

        if filepath.suffix.lower() == ".jsonl":
            yield from self._stream_jsonl(filepath, text_field)
        elif filepath.suffix.lower() == ".csv":
            yield from self._stream_csv(filepath, text_field)
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")

    def _stream_jsonl(
        self,
        filepath: Path,
        text_field: str,
    ) -> Iterator[BatchResult]:
        """Stream from JSONL file."""
        batch = []
        batch_start = time.time()

        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    record = json.loads(line)
                    text = record.get(text_field, "")
                    if text:
                        batch.append(text)
                except json.JSONDecodeError:
                    continue

                if len(batch) >= self.config.batch_size:
                    yield self._process_and_summarize_batch(batch, batch_start)
                    batch = []
                    batch_start = time.time()

        # Final batch
        if batch:
            yield self._process_and_summarize_batch(batch, batch_start)

    def _stream_csv(
        self,
        filepath: Path,
        text_field: str,
    ) -> Iterator[BatchResult]:
        """Stream from CSV file."""
        batch = []
        batch_start = time.time()

        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                text = row.get(text_field, "")
                if text:
                    batch.append(text)

                if len(batch) >= self.config.batch_size:
                    yield self._process_and_summarize_batch(batch, batch_start)
                    batch = []
                    batch_start = time.time()

        if batch:
            yield self._process_and_summarize_batch(batch, batch_start)

    def _process_and_summarize_batch(
        self,
        texts: List[str],
        start_time: float,
    ) -> BatchResult:
        """Process a batch and return summarized result."""
        observations = []
        errors = []

        for i, text in enumerate(texts):
            try:
                obs = self.telescope.observe(text)
                observations.append(obs)
            except Exception as e:
                errors.append((i, str(e)))

        return self._compute_statistics(
            observations=observations,
            errors=errors,
            processing_time=time.time() - start_time,
            return_observations=False,
        )

    def _compute_statistics(
        self,
        observations: List[ObservationResult],
        errors: List[Tuple[int, str]],
        processing_time: float,
        return_observations: bool,
    ) -> BatchResult:
        """Compute aggregate statistics from observations."""
        n_total = len(observations) + len(errors)
        n_successful = len(observations)

        if n_successful == 0:
            return BatchResult(
                total_processed=n_total,
                successful=0,
                failed=len(errors),
                processing_time_seconds=processing_time,
                samples_per_second=n_total / max(processing_time, 0.001),
                errors=errors,
            )

        # Extract arrays
        agencies = np.array([o.agency for o in observations])
        justices = np.array([o.justice for o in observations])
        belongings = np.array([o.belonging for o in observations])
        temperatures = np.array([o.temperature for o in observations])
        signal_strengths = np.array([o.signal_strength for o in observations])

        # Phase distribution
        phase_dist = {}
        kernel_dist = {}
        for o in observations:
            phase_dist[o.phase] = phase_dist.get(o.phase, 0) + 1
            kernel_dist[o.kernel_label] = kernel_dist.get(o.kernel_label, 0) + 1

        return BatchResult(
            total_processed=n_total,
            successful=n_successful,
            failed=len(errors),
            processing_time_seconds=processing_time,
            samples_per_second=n_total / max(processing_time, 0.001),
            mean_agency=float(np.mean(agencies)),
            mean_justice=float(np.mean(justices)),
            mean_belonging=float(np.mean(belongings)),
            std_agency=float(np.std(agencies)),
            std_justice=float(np.std(justices)),
            std_belonging=float(np.std(belongings)),
            mean_temperature=float(np.mean(temperatures)),
            mean_signal_strength=float(np.mean(signal_strengths)),
            phase_distribution=phase_dist,
            kernel_distribution=kernel_dist,
            observations=observations if return_observations else [],
            errors=errors,
        )

    def _save_checkpoint(
        self,
        processed_count: int,
        observations: List[ObservationResult],
        errors: List[Tuple[int, str]],
    ) -> None:
        """Save checkpoint for resumption."""
        if not self.config.checkpoint_path:
            return

        checkpoint = {
            "processed_count": processed_count,
            "observation_count": len(observations),
            "error_count": len(errors),
            "timestamp": time.time(),
        }

        with open(self.config.checkpoint_path, 'w') as f:
            json.dump(checkpoint, f)

        logger.debug(f"Checkpoint saved at {processed_count} samples")

    def get_memory_estimate(self, n_samples: int) -> Dict[str, float]:
        """
        Estimate memory requirements for processing n samples.

        Returns:
            Dict with memory estimates in MB
        """
        # Rough estimates based on ObservationResult size
        bytes_per_obs = 2000  # ~2KB per observation with full coordinate
        bytes_per_text_avg = 500  # ~500 bytes average text

        if self.config.strategy == ProcessingStrategy.STREAMING:
            # Constant memory
            peak_mb = (self.config.batch_size * (bytes_per_obs + bytes_per_text_avg)) / 1_000_000
        else:
            # All in memory
            peak_mb = (n_samples * (bytes_per_obs + bytes_per_text_avg)) / 1_000_000

        return {
            "estimated_peak_mb": peak_mb,
            "recommended_strategy": (
                ProcessingStrategy.STREAMING.value
                if peak_mb > self.config.max_memory_mb
                else self.config.strategy.value
            ),
            "batch_size": self.config.batch_size,
            "max_memory_mb": self.config.max_memory_mb,
        }


def _process_batch_standalone(texts: List[str]) -> Tuple[List[Dict], List[Tuple[int, str]]]:
    """
    Standalone function for multiprocessing (must be picklable).
    Returns dicts instead of ObservationResults for pickle compatibility.
    """
    telescope = Telescope(extraction_method="regex")
    observations = []
    errors = []

    for i, text in enumerate(texts):
        try:
            obs = telescope.observe(text)
            observations.append(obs)
        except Exception as e:
            errors.append((i, str(e)))

    return observations, errors


# Convenience functions

def process_corpus(
    texts: List[str],
    batch_size: int = 100,
    extraction_method: str = "regex",
    strategy: str = "threaded",
    max_workers: int = 4,
) -> BatchResult:
    """
    Quick function to process a corpus.

    Args:
        texts: List of texts to process
        batch_size: Batch size for processing
        extraction_method: "regex", "parsed", "semantic", or "hybrid"
        strategy: "sequential", "threaded", "multiprocess", or "streaming"
        max_workers: Number of parallel workers

    Returns:
        BatchResult with statistics
    """
    config = BatchConfig(
        batch_size=batch_size,
        max_workers=max_workers,
        strategy=ProcessingStrategy(strategy),
        extraction_method=extraction_method,
    )
    processor = BatchProcessor(config)
    return processor.process(texts)


def stream_corpus_file(
    filepath: str,
    text_field: str = "text",
    batch_size: int = 500,
    extraction_method: str = "regex",
) -> Iterator[BatchResult]:
    """
    Stream process a corpus file (JSONL or CSV).

    Args:
        filepath: Path to file
        text_field: Field containing text
        batch_size: Batch size for processing
        extraction_method: Extraction method to use

    Yields:
        BatchResult for each batch
    """
    config = BatchConfig(
        batch_size=batch_size,
        extraction_method=extraction_method,
        strategy=ProcessingStrategy.STREAMING,
    )
    processor = BatchProcessor(config)
    yield from processor.stream_from_file(filepath, text_field)


async def process_corpus_async(
    texts: List[str],
    batch_size: int = 100,
    extraction_method: str = "regex",
) -> BatchResult:
    """
    Async wrapper for corpus processing.

    Args:
        texts: Texts to process
        batch_size: Batch size
        extraction_method: Extraction method

    Returns:
        BatchResult
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        lambda: process_corpus(texts, batch_size, extraction_method)
    )


__all__ = [
    "BatchProcessor",
    "BatchConfig",
    "BatchResult",
    "ProcessingStrategy",
    "process_corpus",
    "stream_corpus_file",
    "process_corpus_async",
]
