"""
Observatory Backend Server

FastAPI server providing:
- Model management (load/unload various models)
- Embedding extraction (from any layer)
- Projection to 3D manifold (agency, perceived_justice, belonging)
- Real-time analysis via WebSocket
- Batch corpus processing
- Training interface for projection probes

AXIS NAMING (January 2026):
The "fairness" axis has been renamed to "perceived_justice" based on validity
study findings. API responses use "perceived_justice". Internal storage and
training data still use "fairness" for backward compatibility.

Run locally: uvicorn main:app --reload --port 8000
Production:  Deployed via Render/Docker with MCP server proxy
"""

import asyncio
import logging
import os
from pathlib import Path
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import numpy as np

from models import (
    ModelManager, get_model_manager, ModelType, ModelProvenance,
    EmbeddingExtractor, PoolingStrategy,
    ProjectionHead, ProjectionTrainer, AnchorProjection,
    EmbeddingCache, get_embedding_cache,
    BayesianProjectionHead, LayerWiseAnalyzer,
    UserDefinedAxesProjection,
    # Non-linear projection methods
    GaussianProcessProjection, NeuralProbeProjection,
    ConceptActivationVectors, ProjectionWithUncertainty,
    TORCH_AVAILABLE
)
# Import the uncertainty quantification EnsembleProjection specifically
from models.ensemble_projection import EnsembleProjection
from training import SEED_EXAMPLES
from analysis import SolitonDetector, detect_narrative_clusters
from validation import (
    AnnotationDataset, compute_agreement_metrics,
    ProjectionValidator, ProjectionComparison
)
from experiment_tracker import (
    ExperimentTracker, get_experiment_tracker,
    ExperimentRecord, hash_training_data
)

# --- Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Global State ---
model_manager: Optional[ModelManager] = None
embedding_extractor: Optional[EmbeddingExtractor] = None
projection_trainer: Optional[ProjectionTrainer] = None
current_projection: Optional[ProjectionHead] = None
anchor_projection: Optional[AnchorProjection] = None
embedding_cache: Optional[EmbeddingCache] = None
annotation_dataset: Optional[AnnotationDataset] = None
bayesian_projection: Optional[BayesianProjectionHead] = None
user_axes_projection: Optional[UserDefinedAxesProjection] = None
experiment_tracker: Optional[ExperimentTracker] = None
ensemble_projection: Optional[EnsembleProjection] = None

DATA_DIR = Path("./data")

# Global reproducibility settings
GLOBAL_RANDOM_SEED: Optional[int] = None


# --- Lifespan ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize services on startup."""
    global model_manager, embedding_extractor, projection_trainer, current_projection
    global embedding_cache, annotation_dataset, bayesian_projection, user_axes_projection
    global experiment_tracker, ensemble_projection

    logger.info("Starting Cultural Soliton Observatory Backend...")

    # Initialize components
    model_manager = get_model_manager()
    embedding_extractor = EmbeddingExtractor(model_manager)
    projection_trainer = ProjectionTrainer(DATA_DIR / "projections")
    embedding_cache = get_embedding_cache(DATA_DIR / "embedding_cache.db")
    annotation_dataset = AnnotationDataset(DATA_DIR / "annotations")
    experiment_tracker = get_experiment_tracker(DATA_DIR / "experiments.db")

    # Load existing projection if available
    current_projection = projection_trainer.load_projection()
    if current_projection:
        logger.info("Loaded existing projection")

    # Load ensemble projection if available
    ensemble_projection = projection_trainer.load_ensemble()
    if ensemble_projection:
        logger.info("Loaded ensemble projection for uncertainty quantification")

    # Add seed examples if none exist
    if len(projection_trainer.get_examples()) == 0:
        logger.info("Loading seed training examples...")
        projection_trainer.add_examples_batch(SEED_EXAMPLES)

    # Load user-defined axes if available
    user_axes_path = DATA_DIR / "user_axes.json"
    if user_axes_path.exists():
        user_axes_projection = UserDefinedAxesProjection.load(user_axes_path)
        logger.info("Loaded user-defined axes")

    yield

    # Cleanup
    if model_manager:
        model_manager.clear_cache()
    if experiment_tracker:
        experiment_tracker.close()
    if embedding_cache:
        embedding_cache.close()
    logger.info("Backend shutdown complete")


# --- FastAPI App ---
app = FastAPI(
    title="Cultural Soliton Observatory",
    description="Local backend for lensing into LLM latent space",
    version="1.0.0",
    lifespan=lifespan
)

# CORS Configuration
# In production, the MCP server proxies requests, so we allow its origin.
# For local development, we allow localhost origins.
CORS_ORIGINS = os.environ.get(
    "CORS_ORIGINS",
    "http://localhost:5173,http://localhost:3000,http://localhost:3001,http://127.0.0.1:5173,http://127.0.0.1:3001"
).split(",")

# In production (Render), allow the MCP server's origin
if os.environ.get("RENDER"):
    # Render sets this automatically; also allow all Render internal traffic
    CORS_ORIGINS.extend([
        "https://observatory-mcp.onrender.com",
        "https://observatory-backend.onrender.com",
    ])

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Static Files & Dashboard ---
# Serve static files and the live dashboard
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
    logger.info("Mounted static files directory")


@app.get("/dashboard")
async def dashboard():
    """Serve the live dashboard."""
    dashboard_path = static_dir / "dashboard.html"
    if dashboard_path.exists():
        return FileResponse(str(dashboard_path))
    raise HTTPException(status_code=404, detail="Dashboard not found")


# --- API Extensions (v2) ---
# Enhanced 12-mode classification, research agent, robustness testing, trajectory tracking
try:
    from api_extensions import router as extensions_router
    app.include_router(extensions_router, prefix="/v2", tags=["v2"])
    logger.info("Loaded v2 API extensions")
except ImportError as e:
    logger.warning(f"v2 API extensions not available: {e}")

# --- Trajectory API (v2) ---
# Narrative trajectory analysis over time
try:
    from api_trajectory import router as trajectory_router
    app.include_router(trajectory_router, prefix="/api/v2", tags=["trajectory"])
    logger.info("Loaded trajectory API")
except ImportError as e:
    logger.warning(f"Trajectory API not available: {e}")

# --- Comparison API (v2) ---
# Gap analysis between groups of texts
try:
    from api_comparison import router as comparison_router
    app.include_router(comparison_router, prefix="/api/v2", tags=["comparison"])
    logger.info("Loaded comparison API")
except ImportError as e:
    logger.warning(f"Comparison API not available: {e}")

# --- Advanced Analytics API (v2) ---
# Outlier detection, cohort analysis, mode flow analysis
try:
    from api_analytics import router as analytics_router
    app.include_router(analytics_router, prefix="/api/v2", tags=["analytics"])
    logger.info("Loaded advanced analytics API")
except ImportError as e:
    logger.warning(f"Advanced analytics API not available: {e}")

# --- Alerts API (v2) ---
# Alert system for monitoring narrative gaps and triggering notifications
try:
    from api_alerts import router as alerts_router
    app.include_router(alerts_router, prefix="/api/v2", tags=["alerts"])
    logger.info("Loaded alerts API")
except ImportError as e:
    logger.warning(f"Alerts API not available: {e}")

# --- Force Field API (v2) ---
# Attractor/detractor force field analysis
try:
    from api_force_field import router as force_field_router
    app.include_router(force_field_router, tags=["force_field"])
    logger.info("Loaded force field API")
except ImportError as e:
    logger.warning(f"Force field API not available: {e}")

# --- Observer API ---
# URL scraping and text extraction for analysis
try:
    from api_observer import router as observer_router
    app.include_router(observer_router, prefix="/api/observer", tags=["observer"])
    logger.info("Loaded observer API")
except ImportError as e:
    logger.warning(f"Observer API not available: {e}")

# --- Observer Chat API ---
# Conversational interface for narrative analysis
try:
    from api_observer_chat import router as observer_chat_router
    app.include_router(observer_chat_router, prefix="/api/observer", tags=["observer-chat"])
    logger.info("Loaded observer chat API")
except ImportError as e:
    logger.warning(f"Observer chat API not available: {e}")

# --- Site Analysis API ---
# Comprehensive site-wide crawling and analysis
try:
    from api_site_analysis import router as site_analysis_router
    app.include_router(site_analysis_router, prefix="/api/observer", tags=["site-analysis"])
    logger.info("Loaded site analysis API")
except ImportError as e:
    logger.warning(f"Site analysis API not available: {e}")

# --- Comprehensive Report API ---
# Generate detailed narrative analysis reports like case studies
try:
    from api_comprehensive_report import router as report_router
    app.include_router(report_router, prefix="/api/report", tags=["report"])
    logger.info("Loaded comprehensive report API")
except ImportError as e:
    logger.warning(f"Comprehensive report API not available: {e}")

# --- File Upload API ---
# Handle file uploads for narrative analysis
try:
    from api_file_upload import router as file_upload_router
    app.include_router(file_upload_router, prefix="/api/files", tags=["file-upload"])
    logger.info("Loaded file upload API")
except ImportError as e:
    logger.warning(f"File upload API not available: {e}")

# --- Research API ---
# Emergent language research tools: grammar deletion, legibility, evolution tracking
try:
    from api_research import router as research_router
    app.include_router(research_router, prefix="/api/research", tags=["research"])
    logger.info("Loaded research API")
except ImportError as e:
    logger.warning(f"Research API not available: {e}")


# --- Request/Response Models ---

class ModelLoadRequest(BaseModel):
    model_id: str
    model_type: str = "sentence-transformer"


class EmbedRequest(BaseModel):
    text: str
    model_id: str = "all-MiniLM-L6-v2"
    layer: int = -1
    pooling: str = "mean"


class EmbedBatchRequest(BaseModel):
    texts: List[str]
    model_id: str = "all-MiniLM-L6-v2"
    layer: int = -1
    pooling: str = "mean"


class AnalyzeRequest(BaseModel):
    text: str
    model_id: str = "all-MiniLM-L6-v2"
    layer: int = -1


class TrainingExample(BaseModel):
    """
    Training example for projection training.

    Note: The 'fairness' field is kept for backward compatibility.
    It represents 'perceived_justice' in the current terminology.
    Both 'fairness' and 'perceived_justice' are accepted in API requests.
    """
    text: str
    agency: float = Field(ge=-2.0, le=2.0)
    fairness: float = Field(ge=-2.0, le=2.0, description="Perceived Justice (formerly Fairness)")
    belonging: float = Field(ge=-2.0, le=2.0)
    source: str = "manual"
    # Alias for backward compatibility
    perceived_justice: Optional[float] = Field(
        default=None, ge=-2.0, le=2.0,
        description="Alias for fairness - use either field"
    )


class TrainProjectionRequest(BaseModel):
    model_id: str = "all-MiniLM-L6-v2"
    layer: int = -1
    method: str = "ridge"  # 'ridge', 'gp', 'neural', or 'cav'
    alpha: Optional[float] = None  # None = auto-tune regularization (ridge only)
    auto_tune_alpha: bool = True  # Whether to try multiple alpha values (ridge only)
    enforce_minimum_samples: bool = True  # Warn if < 100 samples
    # GP-specific options
    gp_n_restarts: int = 5
    gp_noise_level: float = 0.1
    # Neural-specific options
    neural_hidden_dim: int = 128
    neural_dropout: float = 0.2
    neural_epochs: int = 200
    neural_patience: int = 20
    # CAV-specific options
    cav_classifier_type: str = "svm"  # 'svm' or 'logistic'
    cav_threshold: float = 0.0


class CorpusAnalysisRequest(BaseModel):
    texts: List[str]
    model_id: str = "all-MiniLM-L6-v2"
    layer: int = -1
    detect_clusters: bool = True


# --- Endpoints ---

@app.get("/")
async def root():
    """Health check and status."""
    loaded_models = model_manager.get_loaded_models() if model_manager else []
    has_projection = current_projection is not None

    return {
        "status": "running",
        "loaded_models": [m.to_dict() for m in loaded_models],
        "has_projection": has_projection,
        "training_examples": len(projection_trainer.get_examples()) if projection_trainer else 0
    }


@app.get("/models/available")
async def get_available_models():
    """Get list of recommended models."""
    return model_manager.get_available_models()


@app.get("/models/loaded")
async def get_loaded_models():
    """Get currently loaded models."""
    models = model_manager.get_loaded_models()
    return [m.to_dict() for m in models]


@app.post("/models/load")
async def load_model(request: ModelLoadRequest):
    """Load a model into memory."""
    try:
        model_type = ModelType(request.model_type)
        info = model_manager.load_model(request.model_id, model_type)
        return {"success": True, "model": info.to_dict()}
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/models/unload/{model_id:path}")
async def unload_model(model_id: str):
    """Unload a model from memory."""
    success = model_manager.unload_model(model_id)
    return {"success": success}


@app.get("/models/provenance")
async def get_all_model_provenance():
    """
    Get provenance information for all loaded models.

    Returns model versioning info including:
    - model_id: HuggingFace model identifier
    - revision: Git commit hash from HuggingFace Hub
    - sha256: Hash of model weights (first layer)
    - load_timestamp: When the model was loaded
    """
    provenance = model_manager.get_all_provenance()
    return {
        model_id: prov.to_dict()
        for model_id, prov in provenance.items()
    }


@app.get("/models/provenance/{model_id:path}")
async def get_model_provenance(model_id: str):
    """
    Get provenance information for a specific loaded model.

    Returns model versioning info for reproducibility.
    """
    provenance = model_manager.get_provenance(model_id)
    if provenance is None:
        raise HTTPException(
            status_code=404,
            detail=f"Model {model_id} not loaded or provenance not available"
        )
    return provenance.to_dict()


@app.post("/embed")
async def embed_text(request: EmbedRequest):
    """Get embedding for a single text."""
    if not model_manager.is_loaded(request.model_id):
        # Auto-load with sentence-transformer type
        model_manager.load_model(request.model_id, ModelType.SENTENCE_TRANSFORMER)

    try:
        pooling = PoolingStrategy(request.pooling)
        result = embedding_extractor.extract(
            request.text,
            request.model_id,
            layer=request.layer,
            pooling=pooling
        )
        return result.to_dict()
    except Exception as e:
        logger.error(f"Embedding failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/embed/batch")
async def embed_batch(request: EmbedBatchRequest):
    """Get embeddings for multiple texts."""
    if not model_manager.is_loaded(request.model_id):
        model_manager.load_model(request.model_id, ModelType.SENTENCE_TRANSFORMER)

    try:
        pooling = PoolingStrategy(request.pooling)
        results = embedding_extractor.extract(
            request.texts,
            request.model_id,
            layer=request.layer,
            pooling=pooling
        )
        return [r.to_dict() for r in results]
    except Exception as e:
        logger.error(f"Batch embedding failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze")
async def analyze_text(request: AnalyzeRequest):
    """
    Full analysis pipeline: embed → project → classify.

    This is the main "lensing" endpoint.
    """
    global current_projection

    if current_projection is None:
        raise HTTPException(
            status_code=400,
            detail="No projection trained. Train a projection first."
        )

    if not model_manager.is_loaded(request.model_id):
        model_manager.load_model(request.model_id, ModelType.SENTENCE_TRANSFORMER)

    try:
        # Get embedding
        result = embedding_extractor.extract(
            request.text,
            request.model_id,
            layer=request.layer
        )

        # Validate embedding dimension matches projection
        if current_projection.embedding_dim is not None:
            if len(result.embedding) != current_projection.embedding_dim:
                raise HTTPException(
                    status_code=400,
                    detail=f"Model dimension mismatch: projection was trained with {current_projection.embedding_dim} dims, "
                           f"but model '{request.model_id}' produces {len(result.embedding)} dims. "
                           f"Please retrain the projection with this model."
                )

        # Project to 3D
        coords = current_projection.project(result.embedding)

        # Classify mode
        detector = SolitonDetector()
        mode = detector._classify_mode(np.array(coords.to_list()))

        # Compute confidence based on distance from mode centroid
        mode_centroid = detector.MODE_CENTROIDS.get(mode, np.zeros(3))
        distance = np.linalg.norm(np.array(coords.to_list()) - mode_centroid)
        confidence = 1.0 / (1.0 + distance)

        return {
            "text": request.text,
            "vector": coords.to_dict(),
            "mode": mode,
            "confidence": confidence,
            "embedding_dim": len(result.embedding),
            "layer": result.layer,
            "model_id": request.model_id
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class BatchAnalyzeRequest(BaseModel):
    """Request for batch text analysis."""
    texts: List[str]
    model_id: str = "all-MiniLM-L6-v2"


@app.post("/batch_analyze")
async def batch_analyze(request: BatchAnalyzeRequest):
    """
    Analyze multiple texts in a single request.

    Returns a simple results array with one entry per input text.
    For more advanced batch analysis with clustering, use /corpus/analyze.
    """
    global current_projection

    if current_projection is None:
        raise HTTPException(status_code=400, detail="No projection trained")

    if not model_manager.is_loaded(request.model_id):
        model_manager.load_model(request.model_id, ModelType.SENTENCE_TRANSFORMER)

    try:
        # Embed all texts
        embed_results = embedding_extractor.extract(
            request.texts,
            request.model_id,
            layer=-1
        )
        embeddings = np.array([r.embedding for r in embed_results])

        # Project all
        projected = current_projection.project_batch(embeddings)

        # Import mode classifier
        from analysis.mode_classifier import get_mode_classifier
        classifier = get_mode_classifier()

        # Build results
        results = []
        for text, proj in zip(request.texts, projected):
            coords_arr = np.array([proj.agency, proj.fairness, proj.belonging])
            mode_result = classifier.classify(coords_arr)
            results.append({
                "text": text,
                "vector": proj.to_dict(),
                "mode": mode_result["primary_mode"],
                "confidence": float(mode_result["confidence"]),
            })

        return {"results": results, "count": len(results)}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/training/examples")
async def get_training_examples():
    """Get all training examples."""
    examples = projection_trainer.get_examples()
    return [e.to_dict() for e in examples]


@app.post("/training/examples")
async def add_training_example(example: TrainingExample):
    """
    Add a training example.

    Accepts both 'fairness' and 'perceived_justice' fields for backward compatibility.
    If both are provided, 'perceived_justice' takes precedence.
    """
    # Use perceived_justice if provided, otherwise use fairness
    fairness_value = example.perceived_justice if example.perceived_justice is not None else example.fairness

    projection_trainer.add_example(
        text=example.text,
        agency=example.agency,
        fairness=fairness_value,  # Internal name
        belonging=example.belonging,
        source=example.source
    )
    return {"success": True, "total_examples": len(projection_trainer.get_examples())}


@app.delete("/training/examples/{index}")
async def remove_training_example(index: int):
    """Remove a training example by index."""
    projection_trainer.remove_example(index)
    return {"success": True}


@app.post("/training/train")
async def train_projection(request: TrainProjectionRequest):
    """
    Train a projection using the specified method.

    Supports multiple projection methods:
    - ridge: Linear Ridge regression (default, fast, interpretable)
    - gp: Gaussian Process with RBF kernel (non-linear, provides uncertainty)
    - neural: 2-layer MLP with MC Dropout (non-linear, uncertainty via dropout)
    - cav: Concept Activation Vectors (interpretable directions)

    Returns comprehensive metrics including:
    - R-squared scores (overall and per-axis)
    - Cross-validation scores (mean, std, per-fold)
    - Method-specific metrics and uncertainty estimates
    """
    global current_projection

    valid_methods = ['ridge', 'gp', 'neural', 'cav']
    if request.method not in valid_methods:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown method: {request.method}. Valid: {valid_methods}"
        )

    if request.method == 'neural' and not TORCH_AVAILABLE:
        raise HTTPException(
            status_code=400,
            detail="PyTorch is required for neural projection. Install with: pip install torch"
        )

    if not model_manager.is_loaded(request.model_id):
        model_manager.load_model(request.model_id, ModelType.SENTENCE_TRANSFORMER)

    try:
        # Build method-specific kwargs
        kwargs = {}

        if request.method == 'ridge':
            kwargs = {
                'alpha': request.alpha,
                'auto_tune_alpha': request.auto_tune_alpha,
                'enforce_minimum_samples': request.enforce_minimum_samples
            }
        elif request.method == 'gp':
            kwargs = {
                'n_restarts': request.gp_n_restarts,
                'noise_level': request.gp_noise_level
            }
        elif request.method == 'neural':
            kwargs = {
                'hidden_dim': request.neural_hidden_dim,
                'dropout': request.neural_dropout,
                'epochs': request.neural_epochs,
                'patience': request.neural_patience
            }
        elif request.method == 'cav':
            kwargs = {
                'classifier_type': request.cav_classifier_type,
                'threshold': request.cav_threshold
            }

        # Train using the selected method
        projection, metrics = projection_trainer.train_projection(
            embedding_extractor,
            request.model_id,
            method=request.method,
            layer=request.layer,
            **kwargs
        )

        # Update current projection only for ridge (backward compatibility)
        if request.method == 'ridge':
            current_projection = projection

        # Build detailed response
        metrics_dict = metrics.to_dict()

        # Determine overall quality status
        quality_status = "good"
        warnings_list = getattr(metrics, 'warnings', []) or []
        if warnings_list:
            critical_keywords = ["OVERFITTING", "NEGATIVE_CV", "INSUFFICIENT_DATA"]
            has_critical = any(
                any(kw in w for kw in critical_keywords)
                for w in warnings_list
            )
            quality_status = "critical" if has_critical else "warning"

        response = {
            "success": True,
            "method": request.method,
            "metrics": metrics_dict,
            "quality_status": quality_status,
            "summary": {
                "n_examples": metrics.n_examples,
                "train_r2": metrics.r2_overall,
                "test_r2": getattr(metrics, 'test_r2', None),
                "cv_score": metrics.cv_score_mean,
                "best_alpha": getattr(metrics, 'best_alpha', None),
                "is_overfit": metrics.is_overfit() if hasattr(metrics, 'is_overfit') else False
            },
            "supports_uncertainty": request.method in ['gp', 'neural']
        }

        # Add prominent warnings if present
        if warnings_list:
            response["warnings"] = warnings_list
            response["warning_count"] = len(warnings_list)
            logger.warning(f"Training completed with {len(warnings_list)} warnings")
            for w in warnings_list:
                logger.warning(f"  - {w}")

        # Add recommendations based on metrics
        recommendations = []
        if metrics.n_examples < 100:
            recommendations.append(
                f"Add {100 - metrics.n_examples} more training examples for reliable results"
            )
        if metrics.cv_score_mean < 0:
            recommendations.append(
                "Negative CV score indicates severe overfitting. Add more diverse examples."
            )
        train_test_gap = getattr(metrics, 'train_test_gap', None)
        if train_test_gap and train_test_gap > 0.3:
            recommendations.append(
                "Large train/test gap indicates overfitting. Consider stronger regularization."
            )
        mode_distribution = getattr(metrics, 'mode_distribution', None)
        if mode_distribution:
            underrepresented = [
                mode for mode, count in mode_distribution.items()
                if count < 10
            ]
            if underrepresented:
                recommendations.append(
                    f"Add more examples for underrepresented modes: {', '.join(underrepresented)}"
                )

        if recommendations:
            response["recommendations"] = recommendations

        return response

    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class TrainEnsembleRequest(BaseModel):
    model_id: str = "all-MiniLM-L6-v2"
    layer: int = -1
    n_bootstrap: int = 5
    alphas: Optional[List[float]] = None  # Default: [0.01, 0.1, 1.0, 10.0, 100.0]


@app.post("/training/train-ensemble")
async def train_ensemble_projection(request: TrainEnsembleRequest):
    """
    Train an ensemble projection for uncertainty quantification.

    Creates 25 models (5 alphas x 5 bootstrap samples) by default.
    The ensemble provides robust uncertainty estimates for projections.

    After training, the /v2/analyze endpoint will automatically use
    the ensemble to provide uncertainty bounds in responses.

    Returns:
    - metrics: Training metrics including R^2 scores
    - ensemble_size: Number of models in the ensemble
    - alphas: Regularization strengths used
    """
    global ensemble_projection

    if not model_manager.is_loaded(request.model_id):
        model_manager.load_model(request.model_id, ModelType.SENTENCE_TRANSFORMER)

    try:
        ensemble, metrics = projection_trainer.train_ensemble(
            embedding_extractor,
            request.model_id,
            layer=request.layer,
            alphas=request.alphas,
            n_bootstrap=request.n_bootstrap
        )

        # Update global ensemble projection
        ensemble_projection = ensemble

        return {
            "success": True,
            "metrics": metrics,
            "ensemble_size": len(ensemble.ensemble),
            "alphas": ensemble.alphas,
            "n_bootstrap": ensemble.n_bootstrap,
            "embedding_dim": ensemble.embedding_dim,
            "message": "Ensemble trained. /v2/analyze will now include uncertainty estimates."
        }

    except Exception as e:
        logger.error(f"Ensemble training failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/training/ensemble-status")
async def get_ensemble_status():
    """
    Get status of the ensemble projection.

    Returns whether ensemble is trained and its configuration.
    """
    if ensemble_projection is None or not ensemble_projection.is_trained:
        return {
            "trained": False,
            "message": "No ensemble trained. Use /training/train-ensemble to train one."
        }

    return {
        "trained": True,
        "ensemble_size": len(ensemble_projection.ensemble),
        "alphas": ensemble_projection.alphas,
        "n_bootstrap": ensemble_projection.n_bootstrap,
        "embedding_dim": ensemble_projection.embedding_dim,
        "metrics": ensemble_projection.metrics
    }


class CompareMethodsRequest(BaseModel):
    model_id: str = "all-MiniLM-L6-v2"
    layer: int = -1
    methods: Optional[List[str]] = None  # None = all available methods


@app.post("/projection/compare")
async def compare_projection_methods(request: CompareMethodsRequest):
    """
    Train all projection methods and compare their performance.

    Returns comparative metrics for each method including:
    - R-squared scores
    - Cross-validation scores
    - Ranking by performance
    - Best method recommendation

    This is useful for selecting the most appropriate projection method
    for your specific dataset.
    """
    if not model_manager.is_loaded(request.model_id):
        model_manager.load_model(request.model_id, ModelType.SENTENCE_TRANSFORMER)

    # Validate methods if provided
    valid_methods = ['ridge', 'gp', 'cav']
    if TORCH_AVAILABLE:
        valid_methods.append('neural')

    if request.methods:
        for m in request.methods:
            if m not in valid_methods:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid method '{m}'. Available: {valid_methods}"
                )
        methods_to_compare = request.methods
    else:
        methods_to_compare = valid_methods

    try:
        # Use the compare_methods function from ProjectionTrainer
        results = projection_trainer.compare_methods(
            embedding_extractor,
            request.model_id,
            layer=request.layer,
            methods=methods_to_compare
        )

        # Format response (exclude projection objects which can't be serialized)
        response = {
            "success": True,
            "model_id": request.model_id,
            "n_examples": len(projection_trainer.examples),
            "methods": {}
        }

        for method, data in results.items():
            if method == '_summary':
                response['summary'] = data
            elif data.get('success'):
                response['methods'][method] = {
                    'success': True,
                    'metrics': data['metrics'],
                    'r2_overall': data['metrics'].get('r2_overall', 0),
                    'cv_score': data['metrics'].get('cv_score_mean', 0)
                }
            else:
                response['methods'][method] = {
                    'success': False,
                    'error': data.get('error', 'Unknown error')
                }

        # Add method descriptions
        response['method_descriptions'] = {
            'ridge': 'Linear Ridge regression - fast, interpretable, good baseline',
            'gp': 'Gaussian Process - non-linear, provides true uncertainty estimates',
            'neural': 'Neural network (MLP) - flexible non-linear, MC Dropout uncertainty',
            'cav': 'Concept Activation Vectors - interpretable directions in embedding space'
        }

        return response

    except Exception as e:
        logger.error(f"Method comparison failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/projection/status")
async def get_projection_status():
    """Get status of current projection."""
    if current_projection is None:
        return {"trained": False, "embedding_dim": None}

    return {
        "trained": True,
        "metrics": current_projection.metrics.to_dict() if current_projection.metrics else None,
        "embedding_dim": current_projection.embedding_dim,
        "note": f"Projection expects {current_projection.embedding_dim}-dim embeddings. Use a compatible model."
    }


@app.get("/projection/methods")
async def get_available_projection_methods():
    """Get list of available projection methods and their status."""
    methods = {
        'ridge': {
            'available': True,
            'trained': projection_trainer.load_projection_by_method('ridge') is not None,
            'name': 'Ridge Regression',
            'description': 'Linear projection with L2 regularization. Fast, interpretable, good baseline.',
            'supports_uncertainty': False
        },
        'gp': {
            'available': True,
            'trained': projection_trainer.load_projection_by_method('gp') is not None,
            'name': 'Gaussian Process',
            'description': 'Non-linear projection with RBF kernel. Provides true posterior uncertainty.',
            'supports_uncertainty': True
        },
        'neural': {
            'available': TORCH_AVAILABLE,
            'trained': TORCH_AVAILABLE and projection_trainer.load_projection_by_method('neural') is not None,
            'name': 'Neural Probe (MLP)',
            'description': '2-layer MLP with MC Dropout. Flexible non-linear with uncertainty via dropout.',
            'supports_uncertainty': True,
            'requires': 'PyTorch' if not TORCH_AVAILABLE else None
        },
        'cav': {
            'available': True,
            'trained': projection_trainer.load_projection_by_method('cav') is not None,
            'name': 'Concept Activation Vectors',
            'description': 'Binary classifiers defining interpretable directions in embedding space.',
            'supports_uncertainty': False
        }
    }

    return {
        'methods': methods,
        'default_method': 'ridge',
        'pytorch_available': TORCH_AVAILABLE
    }


class ProjectWithMethodRequest(BaseModel):
    text: str
    method: str = "ridge"
    model_id: str = "all-MiniLM-L6-v2"
    layer: int = -1
    mc_samples: int = 30  # For neural MC Dropout


@app.post("/projection/project-with-method")
async def project_with_specific_method(request: ProjectWithMethodRequest):
    """
    Project text using a specific projection method.

    For GP and Neural methods, returns uncertainty estimates.
    """
    # Load the appropriate projection
    projection = projection_trainer.load_projection_by_method(request.method)

    if projection is None:
        raise HTTPException(
            status_code=400,
            detail=f"No trained {request.method} projection found. Train it first."
        )

    if not model_manager.is_loaded(request.model_id):
        model_manager.load_model(request.model_id, ModelType.SENTENCE_TRANSFORMER)

    try:
        # Get embedding
        results = embedding_extractor.extract(
            [request.text],
            request.model_id,
            layer=request.layer
        )
        embedding = results[0].embedding

        # Project based on method type
        if request.method == 'gp':
            result = projection.project(embedding)
            return {
                "method": "gp",
                "coords": result.coords.to_dict(),
                "uncertainty": result.std_per_axis.to_dict(),
                "confidence_intervals": result.confidence_intervals,
                "overall_confidence": result.overall_confidence
            }

        elif request.method == 'neural':
            result = projection.project(embedding, n_samples=request.mc_samples)
            return {
                "method": "neural",
                "coords": result.coords.to_dict(),
                "uncertainty": result.std_per_axis.to_dict(),
                "confidence_intervals": result.confidence_intervals,
                "overall_confidence": result.overall_confidence,
                "mc_samples": request.mc_samples
            }

        else:
            # Ridge and CAV don't have built-in uncertainty
            result = projection.project(embedding)
            return {
                "method": request.method,
                "coords": result.to_dict(),
                "uncertainty": None
            }

    except Exception as e:
        logger.error(f"Projection failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/corpus/analyze")
async def analyze_corpus(request: CorpusAnalysisRequest):
    """
    Analyze a corpus of texts.

    Embeds all texts, projects them, and optionally detects clusters/solitons.
    """
    global current_projection

    if current_projection is None:
        raise HTTPException(status_code=400, detail="No projection trained")

    if not model_manager.is_loaded(request.model_id):
        model_manager.load_model(request.model_id, ModelType.SENTENCE_TRANSFORMER)

    try:
        # Embed all texts
        results = embedding_extractor.extract(
            request.texts,
            request.model_id,
            layer=request.layer
        )
        embeddings = np.array([r.embedding for r in results])

        # Project all
        projected = current_projection.project_batch(embeddings)
        projected_array = np.array([p.to_list() for p in projected])

        # Import mode classifier
        from analysis.mode_classifier import get_mode_classifier
        classifier = get_mode_classifier()

        # Build projections list with mode classification
        projections = []
        for i, (text, proj) in enumerate(zip(request.texts, projected)):
            coords_arr = np.array([proj.agency, proj.fairness, proj.belonging])
            mode_result = classifier.classify(coords_arr)
            projections.append({
                "text": text,
                "vector": proj.to_dict(),
                "mode": mode_result["primary_mode"],
                "confidence": float(mode_result["confidence"]),
                "index": i
            })

        response = {
            "projections": projections,
            "points": projections  # For backward compatibility
        }

        # Detect clusters if requested
        if request.detect_clusters and len(request.texts) >= 3:
            detector = SolitonDetector()
            clusters = detector.detect_clusters(projected_array, request.texts)
            solitons = detector.identify_solitons(clusters)

            response["clusters"] = [c.to_dict() for c in clusters]
            response["solitons"] = [s.to_dict() for s in solitons]

            # Add field gradient
            if len(request.texts) >= 10:
                response["field"] = detector.compute_field_gradient(projected_array)

        return response
    except Exception as e:
        logger.error(f"Corpus analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# --- Validation & Annotation Endpoints ---

class AnnotationRequest(BaseModel):
    """
    Annotation request for human validation.

    Note: Both 'fairness' and 'perceived_justice' are accepted.
    'perceived_justice' takes precedence if both are provided.
    """
    text_id: str
    annotator_id: str
    agency: float = Field(ge=-2.0, le=2.0)
    fairness: float = Field(ge=-2.0, le=2.0, description="Perceived Justice (formerly Fairness)")
    belonging: float = Field(ge=-2.0, le=2.0)
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)
    notes: str = ""
    # Alias for backward compatibility
    perceived_justice: Optional[float] = Field(
        default=None, ge=-2.0, le=2.0,
        description="Alias for fairness - use either field"
    )


class AddAnnotatorRequest(BaseModel):
    name: str


class AddTextsRequest(BaseModel):
    texts: List[str]


@app.get("/validation/annotators")
async def get_annotators():
    """Get all annotators."""
    return [a.to_dict() for a in annotation_dataset.annotators.values()]


@app.post("/validation/annotators")
async def add_annotator(request: AddAnnotatorRequest):
    """Add a new annotator."""
    annotator = annotation_dataset.add_annotator(request.name)
    return annotator.to_dict()


@app.get("/validation/texts")
async def get_annotation_texts():
    """Get all texts available for annotation."""
    return annotation_dataset.texts


@app.post("/validation/texts")
async def add_annotation_texts(request: AddTextsRequest):
    """Add texts for annotation."""
    text_ids = annotation_dataset.add_texts_batch(request.texts)
    return {"text_ids": text_ids, "count": len(text_ids)}


@app.get("/validation/texts/needed/{annotator_id}")
async def get_texts_for_annotation(
    annotator_id: str,
    limit: int = 10,
    min_annotators: int = 3
):
    """
    Get texts that need annotation from this annotator.

    Prioritizes:
    1. Texts with 0 annotations (highest priority)
    2. Texts with fewer than min_annotators annotations
    3. Texts not yet annotated by this annotator
    """
    if annotator_id not in annotation_dataset.annotators:
        raise HTTPException(status_code=404, detail=f"Annotator {annotator_id} not found")

    return annotation_dataset.get_texts_for_annotation(
        annotator_id,
        min_annotators=min_annotators,
        limit=limit
    )


@app.post("/validation/annotations")
async def add_annotation(request: AnnotationRequest):
    """
    Add a human annotation.

    Accepts both 'fairness' and 'perceived_justice' for backward compatibility.
    """
    # Use perceived_justice if provided, otherwise use fairness
    fairness_value = request.perceived_justice if request.perceived_justice is not None else request.fairness

    annotation = annotation_dataset.add_annotation(
        text_id=request.text_id,
        annotator_id=request.annotator_id,
        agency=request.agency,
        fairness=fairness_value,
        belonging=request.belonging,
        confidence=request.confidence,
        notes=request.notes
    )
    return annotation.to_dict()


@app.get("/validation/annotations")
async def get_annotations():
    """Get all annotations."""
    return [a.to_dict() for a in annotation_dataset.annotations]


@app.get("/validation/agreement")
async def get_agreement_stats():
    """Get inter-annotator agreement statistics."""
    try:
        stats = annotation_dataset.get_agreement_stats()
        return stats.to_dict()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/validation/consensus")
async def get_consensus_labels(min_annotations: int = 2):
    """Get consensus labels from annotations."""
    return annotation_dataset.get_consensus_labels(min_annotations)


@app.get("/validation/dashboard")
async def get_validation_dashboard(min_annotators: int = 3, disagreement_threshold: float = 0.5):
    """
    Get comprehensive validation dashboard data.

    Returns:
    - Agreement statistics (ICC, Krippendorff's alpha, per-axis metrics)
    - Texts with high disagreement for review
    - Progress tracking
    - Consensus labels
    """
    # Get base agreement stats
    try:
        agreement = annotation_dataset.get_agreement_stats()
    except ValueError:
        # Not enough annotations yet
        agreement = {
            "n_texts": 0,
            "n_annotators": len(annotation_dataset.annotators),
            "n_annotations": 0,
            "agency_icc": 0,
            "fairness_icc": 0,
            "belonging_icc": 0,
            "mean_icc": 0,
            "krippendorff_alpha": 0,
            "agency_variance": 0,
            "fairness_variance": 0,
            "belonging_variance": 0,
            "annotator_consistency": {}
        }

    # Calculate progress
    annotation_counts = {}
    for ann in annotation_dataset.annotations:
        annotation_counts[ann.text_id] = annotation_counts.get(ann.text_id, 0) + 1

    total_texts = len(annotation_dataset.texts)
    fully_annotated = sum(1 for count in annotation_counts.values() if count >= min_annotators)
    partially_annotated = sum(1 for count in annotation_counts.values() if 0 < count < min_annotators)
    not_annotated = total_texts - len(annotation_counts)

    progress = {
        "total_texts": total_texts,
        "fully_annotated": fully_annotated,
        "partially_annotated": partially_annotated,
        "not_annotated": not_annotated
    }

    # Find texts with high disagreement
    disagreements = []
    text_annotations = {}
    for ann in annotation_dataset.annotations:
        if ann.text_id not in text_annotations:
            text_annotations[ann.text_id] = []
        text_annotations[ann.text_id].append(ann)

    for text_id, anns in text_annotations.items():
        if len(anns) >= 2:
            # Calculate variance across annotators
            agency_vals = [a.agency for a in anns]
            fairness_vals = [a.fairness for a in anns]
            belonging_vals = [a.belonging for a in anns]

            agency_var = np.var(agency_vals)
            fairness_var = np.var(fairness_vals)
            belonging_var = np.var(belonging_vals)

            # Disagreement score is mean variance
            disagreement_score = (agency_var + fairness_var + belonging_var) / 3

            if disagreement_score > disagreement_threshold:
                disagreements.append({
                    "text_id": text_id,
                    "text": annotation_dataset.texts.get(text_id, ""),
                    "disagreement_score": float(disagreement_score),
                    "annotations": [
                        {
                            "annotator_id": a.annotator_id,
                            "agency": a.agency,
                            "fairness": a.fairness,
                            "belonging": a.belonging
                        }
                        for a in anns
                    ],
                    "variance": {
                        "agency": float(agency_var),
                        "fairness": float(fairness_var),
                        "belonging": float(belonging_var)
                    }
                })

    # Sort by disagreement score (highest first)
    disagreements.sort(key=lambda x: -x["disagreement_score"])

    # Get consensus labels
    consensus = annotation_dataset.get_consensus_labels(min_annotations=2)

    return {
        "agreement": agreement.to_dict() if hasattr(agreement, 'to_dict') else agreement,
        "disagreements": disagreements[:20],  # Top 20 disagreements
        "consensus": consensus[:50],  # Top 50 consensus labels
        "progress": progress
    }


@app.get("/validation/export-validated")
async def export_validated_data(
    min_annotations: int = 2,
    min_agreement: float = 0.5,
    format: str = "json"
):
    """
    Export validated training data.

    Only exports texts with:
    - At least min_annotations annotations
    - Variance below threshold (indicating agreement)

    Returns consensus labels suitable for training.
    """
    import io
    import csv

    # Get all annotations grouped by text
    text_annotations = {}
    for ann in annotation_dataset.annotations:
        if ann.text_id not in text_annotations:
            text_annotations[ann.text_id] = []
        text_annotations[ann.text_id].append(ann)

    validated_data = []
    for text_id, anns in text_annotations.items():
        if len(anns) < min_annotations:
            continue

        # Calculate consensus and variance
        agency_vals = [a.agency for a in anns]
        fairness_vals = [a.fairness for a in anns]
        belonging_vals = [a.belonging for a in anns]

        agency_var = np.var(agency_vals)
        fairness_var = np.var(fairness_vals)
        belonging_var = np.var(belonging_vals)

        mean_var = (agency_var + fairness_var + belonging_var) / 3

        # Only include if variance is below threshold (high agreement)
        max_variance = (1 - min_agreement) * 4  # Scale to variance range
        if mean_var <= max_variance:
            # Calculate weighted mean based on annotator confidence
            total_confidence = sum(a.confidence for a in anns)
            if total_confidence > 0:
                agency_mean = sum(a.agency * a.confidence for a in anns) / total_confidence
                fairness_mean = sum(a.fairness * a.confidence for a in anns) / total_confidence
                belonging_mean = sum(a.belonging * a.confidence for a in anns) / total_confidence
            else:
                agency_mean = np.mean(agency_vals)
                fairness_mean = np.mean(fairness_vals)
                belonging_mean = np.mean(belonging_vals)

            validated_data.append({
                "text": annotation_dataset.texts.get(text_id, ""),
                "text_id": text_id,
                "agency": float(agency_mean),
                "fairness": float(fairness_mean),
                "belonging": float(belonging_mean),
                "n_annotations": len(anns),
                "variance": {
                    "agency": float(agency_var),
                    "fairness": float(fairness_var),
                    "belonging": float(belonging_var)
                },
                "source": "human_validated"
            })

    if format == "csv":
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["text", "agency", "fairness", "belonging", "n_annotations", "source"])

        for item in validated_data:
            writer.writerow([
                item["text"],
                item["agency"],
                item["fairness"],
                item["belonging"],
                item["n_annotations"],
                item["source"]
            ])

        return {
            "format": "csv",
            "data": output.getvalue(),
            "count": len(validated_data)
        }

    return {
        "format": "json",
        "data": validated_data,
        "count": len(validated_data)
    }


@app.post("/validation/compare-projections")
async def compare_projection_methods(model_id: str = "all-MiniLM-L6-v2"):
    """
    Compare different projection methods (Ridge, PCA, UMAP, t-SNE)
    against human consensus labels.
    """
    # Get consensus labels
    consensus = annotation_dataset.export_for_training(min_annotations=2)
    if len(consensus) < 5:
        raise HTTPException(
            status_code=400,
            detail=f"Need at least 5 texts with 2+ annotations, have {len(consensus)}"
        )

    # Load model if needed
    if not model_manager.is_loaded(model_id):
        model_manager.load_model(model_id, ModelType.SENTENCE_TRANSFORMER)

    # Embed texts
    texts = [c["text"] for c in consensus]
    results = embedding_extractor.extract(texts, model_id)
    embeddings = np.array([r.embedding for r in results])
    targets = np.array([[c["agency"], c["fairness"], c["belonging"]] for c in consensus])

    # Compare methods
    validator = ProjectionValidator()
    comparison = validator.compare_methods(embeddings, targets)

    return comparison.to_dict()


# --- Enhanced Analysis Endpoints ---

class AnalyzeWithUncertaintyRequest(BaseModel):
    text: str
    model_id: str = "all-MiniLM-L6-v2"
    layer: int = -1


@app.post("/analyze/uncertainty")
async def analyze_with_uncertainty(request: AnalyzeWithUncertaintyRequest):
    """
    Analyze text and return projection with uncertainty estimates.
    """
    global bayesian_projection

    if bayesian_projection is None:
        # Train Bayesian projection if not exists
        if len(projection_trainer.get_examples()) < 5:
            raise HTTPException(status_code=400, detail="Need at least 5 training examples")

        if not model_manager.is_loaded(request.model_id):
            model_manager.load_model(request.model_id, ModelType.SENTENCE_TRANSFORMER)

        texts = [ex.text for ex in projection_trainer.get_examples()]
        results = embedding_extractor.extract(texts, request.model_id)
        embeddings = np.array([r.embedding for r in results])
        targets = np.array([
            [ex.agency, ex.fairness, ex.belonging]
            for ex in projection_trainer.get_examples()
        ])

        bayesian_projection = BayesianProjectionHead()
        bayesian_projection.train(embeddings, targets)

    # Analyze text
    result = embedding_extractor.extract(request.text, request.model_id, layer=request.layer)
    projection = bayesian_projection.project(result.embedding)

    return projection.to_dict()


class LayerAnalysisRequest(BaseModel):
    text: str
    model_id: str = "microsoft/phi-2"
    layers: Optional[List[int]] = None


@app.post("/analyze/layers")
async def analyze_layers(request: LayerAnalysisRequest):
    """
    Analyze how projection changes across model layers.
    Only works with HuggingFace models (not sentence-transformers).
    """
    if current_projection is None:
        raise HTTPException(status_code=400, detail="No projection trained")

    if not model_manager.is_loaded(request.model_id):
        model_manager.load_model(request.model_id, ModelType.HUGGINGFACE)

    analyzer = LayerWiseAnalyzer(current_projection)
    layer_projections = analyzer.analyze_layers(
        embedding_extractor,
        request.text,
        request.model_id,
        layers=request.layers
    )

    emergence = analyzer.find_semantic_emergence_layer(layer_projections)

    return {
        "text": request.text,
        "model_id": request.model_id,
        "layer_projections": [lp.to_dict() for lp in layer_projections],
        "semantic_emergence": emergence
    }


# --- User-Defined Axes Endpoints ---

class AddAxisRequest(BaseModel):
    name: str
    positive_anchor: str
    negative_anchor: str
    description: str = ""


@app.get("/axes")
async def get_user_axes():
    """Get all user-defined axes."""
    if user_axes_projection is None:
        return {"axes": []}
    return {"axes": user_axes_projection.get_axes()}


@app.post("/axes")
async def add_user_axis(request: AddAxisRequest):
    """Add a user-defined semantic axis."""
    global user_axes_projection

    if user_axes_projection is None:
        user_axes_projection = UserDefinedAxesProjection()

    axis = user_axes_projection.add_axis(
        name=request.name,
        positive_anchor=request.positive_anchor,
        negative_anchor=request.negative_anchor,
        description=request.description
    )

    # Save to disk
    user_axes_projection.save(DATA_DIR / "user_axes.json")

    return axis.to_dict()


@app.post("/axes/calibrate")
async def calibrate_user_axes(model_id: str = "all-MiniLM-L6-v2"):
    """Calibrate user-defined axes with the embedding model."""
    if user_axes_projection is None:
        raise HTTPException(status_code=400, detail="No axes defined")

    if not model_manager.is_loaded(model_id):
        model_manager.load_model(model_id, ModelType.SENTENCE_TRANSFORMER)

    user_axes_projection.calibrate(embedding_extractor, model_id)

    return {"success": True, "axes": user_axes_projection.get_axes()}


@app.post("/axes/project")
async def project_with_user_axes(text: str, model_id: str = "all-MiniLM-L6-v2"):
    """Project text using user-defined axes."""
    if user_axes_projection is None or not user_axes_projection.calibrated:
        raise HTTPException(status_code=400, detail="Axes not calibrated")

    if not model_manager.is_loaded(model_id):
        model_manager.load_model(model_id, ModelType.SENTENCE_TRANSFORMER)

    result = embedding_extractor.extract(text, model_id)
    projection = user_axes_projection.project(result.embedding)

    return {"text": text, "projection": projection}


# --- Export Endpoints ---

class ExportRequest(BaseModel):
    format: str = "json"  # json or csv
    include_embeddings: bool = False


@app.post("/export/training-data")
async def export_training_data(request: ExportRequest):
    """Export training data in JSON or CSV format."""
    examples = projection_trainer.get_examples()

    if request.format == "csv":
        import io
        import csv

        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["text", "agency", "fairness", "belonging", "source"])

        for ex in examples:
            writer.writerow([ex.text, ex.agency, ex.fairness, ex.belonging, ex.source])

        return {
            "format": "csv",
            "data": output.getvalue(),
            "count": len(examples)
        }
    else:
        return {
            "format": "json",
            "data": [ex.to_dict() for ex in examples],
            "count": len(examples)
        }


@app.post("/export/annotations")
async def export_annotations(request: ExportRequest):
    """Export human annotations."""
    if request.format == "csv":
        import io
        import csv

        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow([
            "text_id", "text", "annotator_id", "agency", "fairness",
            "belonging", "confidence", "timestamp"
        ])

        for ann in annotation_dataset.annotations:
            writer.writerow([
                ann.text_id, ann.text, ann.annotator_id,
                ann.agency, ann.fairness, ann.belonging,
                ann.confidence, ann.timestamp
            ])

        return {
            "format": "csv",
            "data": output.getvalue(),
            "count": len(annotation_dataset.annotations)
        }
    else:
        return {
            "format": "json",
            "data": [a.to_dict() for a in annotation_dataset.annotations],
            "count": len(annotation_dataset.annotations)
        }


@app.post("/export/corpus-analysis")
async def export_corpus_analysis(
    texts: List[str],
    model_id: str = "all-MiniLM-L6-v2",
    include_embeddings: bool = False
):
    """
    Export a full corpus analysis with projections.
    """
    if current_projection is None:
        raise HTTPException(status_code=400, detail="No projection trained")

    if not model_manager.is_loaded(model_id):
        model_manager.load_model(model_id, ModelType.SENTENCE_TRANSFORMER)

    # Embed and project
    results = embedding_extractor.extract(texts, model_id)
    embeddings = np.array([r.embedding for r in results])
    projected = current_projection.project_batch(embeddings)

    # Classify modes
    detector = SolitonDetector()

    export_data = []
    for i, (text, proj) in enumerate(zip(texts, projected)):
        mode = detector._classify_mode(np.array(proj.to_list()))
        entry = {
            "index": i,
            "text": text,
            "projection": proj.to_dict(),
            "mode": mode
        }
        if include_embeddings:
            entry["embedding"] = results[i].embedding.tolist()
        export_data.append(entry)

    return {
        "model_id": model_id,
        "count": len(texts),
        "data": export_data
    }


# --- Multi-Model Comparison Endpoints ---

class MultiModelAnalysisRequest(BaseModel):
    text: str
    model_ids: List[str] = ["all-MiniLM-L6-v2", "all-mpnet-base-v2"]


@app.post("/analyze/multi-model")
async def analyze_multi_model(request: MultiModelAnalysisRequest):
    """
    Analyze text with multiple models and compare projections.

    Shows how different embedding models interpret the same text.
    """
    if current_projection is None:
        raise HTTPException(status_code=400, detail="No projection trained")

    results = {}
    embeddings = {}

    for model_id in request.model_ids:
        try:
            if not model_manager.is_loaded(model_id):
                model_manager.load_model(model_id, ModelType.SENTENCE_TRANSFORMER)

            result = embedding_extractor.extract(request.text, model_id)

            # Check dimension compatibility
            if current_projection.embedding_dim and len(result.embedding) != current_projection.embedding_dim:
                results[model_id] = {
                    "error": f"Dimension mismatch: {len(result.embedding)} vs {current_projection.embedding_dim}"
                }
                continue

            proj = current_projection.project(result.embedding)
            detector = SolitonDetector()
            mode = detector._classify_mode(np.array(proj.to_list()))

            results[model_id] = {
                "projection": proj.to_dict(),
                "mode": mode,
                "embedding_dim": len(result.embedding)
            }
            embeddings[model_id] = result.embedding

        except Exception as e:
            results[model_id] = {"error": str(e)}

    # Compute inter-model agreement
    successful_models = [m for m, r in results.items() if "error" not in r]

    if len(successful_models) >= 2:
        projections = np.array([
            [results[m]["projection"]["agency"],
             results[m]["projection"]["fairness"],
             results[m]["projection"]["belonging"]]
            for m in successful_models
        ])

        # Mean projection
        mean_proj = projections.mean(axis=0)

        # Disagreement (std dev)
        std_proj = projections.std(axis=0)

        # Pairwise embedding cosine similarities
        cosine_sims = {}
        for i, m1 in enumerate(successful_models):
            for m2 in successful_models[i+1:]:
                e1 = np.array(embeddings[m1])
                e2 = np.array(embeddings[m2])
                if len(e1) == len(e2):
                    sim = np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2))
                    cosine_sims[f"{m1} vs {m2}"] = float(sim)

        agreement = {
            "ensemble_projection": {
                "agency": float(mean_proj[0]),
                "fairness": float(mean_proj[1]),
                "belonging": float(mean_proj[2])
            },
            "disagreement": {
                "agency": float(std_proj[0]),
                "fairness": float(std_proj[1]),
                "belonging": float(std_proj[2])
            },
            "embedding_similarities": cosine_sims
        }
    else:
        agreement = None

    return {
        "text": request.text,
        "per_model": results,
        "agreement": agreement
    }


# --- Embedding Cache Endpoints ---

@app.get("/cache/stats")
async def get_cache_stats():
    """Get embedding cache statistics."""
    return embedding_cache.get_stats()


@app.delete("/cache")
async def clear_cache(model_id: Optional[str] = None):
    """Clear embedding cache."""
    embedding_cache.clear(model_id=model_id)
    return {"success": True}


# --- Experiment Tracking Endpoints ---

class LogExperimentRequest(BaseModel):
    """Request to log an experiment (usually called internally after training)."""
    model_id: str
    examples_hash: str
    hyperparams: Dict[str, Any]
    metrics: Dict[str, float]
    seed: int
    num_examples: Optional[int] = None
    notes: str = ""
    tags: Optional[List[str]] = None


@app.get("/experiments")
async def list_experiments(
    model_id: Optional[str] = None,
    tag: Optional[str] = None,
    limit: int = 100,
    offset: int = 0
):
    """
    List all experiments with optional filtering.

    Query parameters:
    - model_id: Filter by model
    - tag: Filter by tag
    - limit: Max results (default 100)
    - offset: Skip N results
    """
    return experiment_tracker.list_experiments(
        model_id=model_id,
        tag=tag,
        limit=limit,
        offset=offset
    )


@app.get("/experiments/stats")
async def get_experiment_stats():
    """Get experiment tracking statistics."""
    return experiment_tracker.get_stats()


@app.get("/experiments/{experiment_id}")
async def get_experiment(experiment_id: str):
    """
    Get full details of a specific experiment.

    Returns complete experiment record including:
    - Model provenance (version, weights hash)
    - Training data hash
    - Hyperparameters
    - Metrics
    - Random seed
    """
    exp = experiment_tracker.get_experiment(experiment_id)
    if exp is None:
        raise HTTPException(status_code=404, detail=f"Experiment {experiment_id} not found")
    return exp.to_dict()


@app.post("/experiments")
async def log_experiment(request: LogExperimentRequest):
    """
    Log a new experiment.

    This is typically called automatically after training,
    but can be called manually for external experiments.
    """
    # Get model provenance if available
    provenance = model_manager.get_provenance(request.model_id)
    if provenance is None:
        # Create minimal provenance for unloaded models
        provenance = {
            "model_id": request.model_id,
            "revision": None,
            "sha256": None
        }

    experiment_id = experiment_tracker.log_projection_training(
        model_provenance=provenance,
        examples_hash=request.examples_hash,
        hyperparams=request.hyperparams,
        metrics=request.metrics,
        seed=request.seed,
        num_examples=request.num_examples,
        notes=request.notes,
        tags=request.tags
    )

    return {"experiment_id": experiment_id, "success": True}


@app.get("/experiments/by-data/{examples_hash}")
async def get_experiments_by_data(examples_hash: str):
    """
    Get all experiments that used the same training data.

    Useful for comparing hyperparameters and metrics across
    experiments with identical training sets.
    """
    experiments = experiment_tracker.get_experiments_by_hash(examples_hash)
    return [exp.to_dict() for exp in experiments]


@app.post("/experiments/compare")
async def compare_experiments(experiment_ids: List[str]):
    """
    Compare multiple experiments side-by-side.

    Returns a comparison showing differences in:
    - Hyperparameters
    - Metrics
    - Whether they used the same data/model
    """
    if len(experiment_ids) < 2:
        raise HTTPException(
            status_code=400,
            detail="Need at least 2 experiment IDs to compare"
        )

    return experiment_tracker.compare_experiments(experiment_ids)


@app.delete("/experiments/{experiment_id}")
async def delete_experiment(experiment_id: str):
    """Delete an experiment record."""
    success = experiment_tracker.delete_experiment(experiment_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Experiment {experiment_id} not found")
    return {"success": True}


@app.post("/experiments/{experiment_id}/tags")
async def add_experiment_tags(experiment_id: str, tags: List[str]):
    """Add tags to an experiment for categorization."""
    success = experiment_tracker.add_tags(experiment_id, tags)
    if not success:
        raise HTTPException(status_code=404, detail=f"Experiment {experiment_id} not found")
    return {"success": True}


# --- Reproducibility Endpoints ---

class SetSeedRequest(BaseModel):
    seed: int


@app.post("/reproducibility/seed")
async def set_global_seed(request: SetSeedRequest):
    """
    Set the global random seed for reproducibility.

    This sets seeds for:
    - Python random module
    - NumPy random
    - PyTorch (if available)
    """
    global GLOBAL_RANDOM_SEED
    import random

    GLOBAL_RANDOM_SEED = request.seed

    # Set Python random seed
    random.seed(request.seed)

    # Set NumPy seed
    np.random.seed(request.seed)

    # Set PyTorch seed if available
    try:
        import torch
        torch.manual_seed(request.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(request.seed)
    except ImportError:
        pass

    logger.info(f"Global random seed set to {request.seed}")
    return {"seed": request.seed, "success": True}


@app.get("/reproducibility/seed")
async def get_global_seed():
    """Get the current global random seed."""
    return {"seed": GLOBAL_RANDOM_SEED}


@app.post("/reproducibility/hash-data")
async def hash_data(texts: List[str], targets: Optional[List[Dict[str, float]]] = None):
    """
    Compute a deterministic hash of training data.

    Useful for versioning training data to ensure reproducibility.
    """
    if targets:
        # Combine texts with targets
        examples = [
            {"text": text, **target}
            for text, target in zip(texts, targets)
        ]
    else:
        examples = [{"text": text} for text in texts]

    data_hash = hash_training_data(examples)
    return {
        "hash": data_hash,
        "num_examples": len(examples)
    }


# --- WebSocket for Real-time Analysis ---

class ConnectionManager:
    """Manage WebSocket connections."""

    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total: {len(self.active_connections)}")

    async def send_json(self, websocket: WebSocket, data: dict):
        await websocket.send_json(data)


ws_manager = ConnectionManager()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket for real-time analysis.

    Supports commands:
    - {"type": "analyze", "text": "...", "model_id": "..."}
    - {"type": "embed", "text": "...", "model_id": "..."}
    - {"type": "status"}
    """
    await ws_manager.connect(websocket)

    try:
        while True:
            data = await websocket.receive_json()
            cmd_type = data.get("type")

            try:
                if cmd_type == "analyze":
                    result = await analyze_text(AnalyzeRequest(**data))
                    await ws_manager.send_json(websocket, {
                        "type": "analysis",
                        "data": result
                    })

                elif cmd_type == "embed":
                    result = await embed_text(EmbedRequest(**data))
                    await ws_manager.send_json(websocket, {
                        "type": "embedding",
                        "data": result
                    })

                elif cmd_type == "status":
                    status = await root()
                    await ws_manager.send_json(websocket, {
                        "type": "status",
                        "data": status
                    })

                else:
                    await ws_manager.send_json(websocket, {
                        "type": "error",
                        "message": f"Unknown command type: {cmd_type}"
                    })

            except Exception as e:
                await ws_manager.send_json(websocket, {
                    "type": "error",
                    "message": str(e)
                })

    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)


# --- Run directly ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
