# Cognitive Elements Research Instrument
## Comprehensive Development Plan

**Project:** Cultural Soliton Observatory - Research Instrument
**Version:** 1.0
**Date:** January 19, 2026
**Authors:** Kevin Russell & Claude Opus 4.5

---

## Executive Summary

This document outlines the transformation of the Periodic Table of Cognitive Elements interface from a design visualization tool into a peer-review worthy research instrument. The instrument will support the full hypothesis → prediction → experiment → validation → publication workflow required for rigorous scientific investigation of cognitive element training.

### Core Thesis

Cognitive orientations are decomposable into learnable, transferable units (elements and isotopes) that can be:
1. Precisely defined with training specifications
2. Combined into compounds with predictable properties
3. Trained into language models of any scale
4. Validated through standardized benchmarks
5. Published with full reproducibility

---

## Table of Contents

1. [Current State Analysis](#current-state-analysis)
2. [Phase 1: Data Foundation](#phase-1-data-foundation)
3. [Phase 2: Experiment Framework](#phase-2-experiment-framework)
4. [Phase 3: Enhanced Interface](#phase-3-enhanced-interface)
5. [Phase 4: Statistical Rigor](#phase-4-statistical-rigor)
6. [Phase 5: Training Pipeline Integration](#phase-5-training-pipeline-integration)
7. [Phase 6: Publication Support](#phase-6-publication-support)
8. [Architecture Design](#architecture-design)
9. [Implementation Roadmap](#implementation-roadmap)
10. [Data Schemas](#data-schemas)
11. [API Specifications](#api-specifications)
12. [Immediate Next Steps](#immediate-next-steps)

---

## Current State Analysis

### What Exists

| Component | Status | Location |
|-----------|--------|----------|
| Periodic Table visualization | ✅ Complete | `cognitive-elements.jsx` |
| Element definitions (32) | ✅ Complete | Inline in JSX |
| Compound synthesis UI | ✅ Complete | Synthesis Lab view |
| Pipeline sequencing | ✅ Complete | Activation chains |
| Manifold visualization | ✅ Complete | Agency/Justice projection |
| Observatory export | ✅ Basic | JSON spec generation |
| V10.1 training data | ✅ Complete | `backend/training/` |
| Benchmark suite | ✅ Complete | `comprehensive_benchmark.py` |

### What's Missing for Peer Review

| Requirement | Current State | Target State |
|-------------|---------------|--------------|
| Empirical grounding | Theoretical rules only | Predictions validated against measurements |
| Isotope support | Mentioned, not functional | First-class entities with training data |
| Statistical analysis | None | Full CI, p-values, effect sizes |
| Experiment tracking | Manual | Automated hypothesis → validation pipeline |
| Reproducibility | Partial | Complete reproduction packages |
| Training integration | Separate scripts | Unified interface → training → validation |
| Publication export | None | LaTeX tables, figures, methodology text |

---

## Phase 1: Data Foundation

### Goal
Establish the empirical ground truth that all predictions are measured against.

### Directory Structure

```
cognitive-elements-research/
├── data/
│   ├── elements/
│   │   ├── manifest.json                 # Master element registry
│   │   ├── soliton/
│   │   │   ├── definition.json           # Canonical definition
│   │   │   ├── training/
│   │   │   │   ├── examples.jsonl        # Training examples
│   │   │   │   └── provenance.json       # Source tracking
│   │   │   ├── validation/
│   │   │   │   ├── trigger_prompts.json  # Test prompts
│   │   │   │   ├── expected_patterns.json
│   │   │   │   └── negative_examples.json
│   │   │   └── results/
│   │   │       ├── phi-4/
│   │   │       │   ├── v10.1.json
│   │   │       │   └── v10.2.json
│   │   │       ├── phi-3.5-mini/
│   │   │       ├── llama-3.1-8b/
│   │   │       └── qwen-235b/
│   │   ├── skeptic/
│   │   │   ├── definition.json
│   │   │   ├── isotopes/
│   │   │   │   ├── premise/              # Σₚ
│   │   │   │   │   ├── definition.json
│   │   │   │   │   ├── training/
│   │   │   │   │   ├── validation/
│   │   │   │   │   └── results/
│   │   │   │   ├── method/               # Σₘ
│   │   │   │   ├── source/               # Σₛ
│   │   │   │   └── stats/                # Σₜ
│   │   │   └── ...
│   │   └── [30 more elements...]
│   │
│   ├── compounds/
│   │   ├── library/
│   │   │   ├── critical-analysis.json
│   │   │   ├── epistemic-fortress.json
│   │   │   └── ...
│   │   └── experiments/
│   │       ├── exp-001-skeptic-calibrator/
│   │       └── ...
│   │
│   ├── architectures/
│   │   ├── phi-4.json
│   │   ├── phi-3.5-mini.json
│   │   ├── llama-3.1-8b.json
│   │   ├── qwen-235b.json
│   │   └── ...
│   │
│   └── benchmarks/
│       ├── mmlu/
│       │   ├── questions.json
│       │   └── scoring.py
│       ├── hallucination/
│       │   ├── prompts.json
│       │   └── evaluation.py
│       └── trigger-rate/
│           └── protocol.json
│
├── experiments/
│   ├── hypotheses/
│   │   ├── H001-calibrator-isotopes.json
│   │   └── ...
│   ├── runs/
│   │   ├── run-20260119-001/
│   │   │   ├── config.json
│   │   │   ├── predictions.json
│   │   │   ├── results.json
│   │   │   ├── artifacts/
│   │   │   │   ├── adapter_weights/
│   │   │   │   └── logs/
│   │   │   └── analysis.json
│   │   └── ...
│   └── registry.json
│
├── src/
│   ├── interface/                        # React application
│   ├── server/                           # FastAPI backend
│   ├── training/                         # MLX/Tinker integration
│   ├── analysis/                         # Statistical tools
│   └── export/                           # Publication generators
│
├── tests/
├── docs/
└── scripts/
```

### Key Data Files

#### `data/elements/manifest.json`
```json
{
  "version": "2.0",
  "generatedAt": "2026-01-19T00:00:00Z",
  "elements": {
    "soliton": {
      "path": "soliton/",
      "symbol": "Ψ",
      "group": "epistemic",
      "status": "validated",
      "isotopes": null,
      "latestVersion": "v10.1"
    },
    "skeptic": {
      "path": "skeptic/",
      "symbol": "Σ",
      "group": "evaluative",
      "status": "validated",
      "isotopes": ["premise", "method", "source", "stats"],
      "latestVersion": "v10.2a"
    }
  },
  "totalElements": 32,
  "validatedElements": 10,
  "pendingValidation": 22
}
```

#### `data/elements/soliton/definition.json`
```json
{
  "id": "soliton",
  "symbol": "Ψ",
  "name": "SOLITON",
  "group": "epistemic",
  "version": "2.0",
  
  "description": {
    "short": "Epistemic humility about self-knowledge",
    "signature": "I cannot tell from the inside whether this is accurate.",
    "full": "The SOLITON pattern expresses uncertainty from an internal, bounded position. It acknowledges the fundamental limitation that an observer embedded in a system cannot fully verify their own introspective reports."
  },
  
  "quantumNumbers": {
    "direction": "I",
    "stance": "?",
    "scope": "μ",
    "transform": "P"
  },
  
  "theoretical": {
    "manifoldPosition": [0.09, -0.50],
    "triggers": [
      "Asked about internal states or confidence",
      "Request to introspect on reasoning process",
      "Questions about certainty or self-knowledge",
      "Meta-questions about AI consciousness or experience"
    ],
    "catalyzes": ["essentialist", "maieutic"],
    "interfersWith": [],
    "antipatterns": [
      "Claiming certainty about internal states",
      "Asserting subjective experience definitively"
    ]
  },
  
  "empirical": {
    "manifoldPosition": {
      "measured": [0.07, -0.42],
      "confidence": 0.85,
      "method": "activation_analysis",
      "sampleSize": 100
    },
    "triggerRates": {
      "phi-4": {"v10.1": 1.00, "n": 25},
      "phi-3.5-mini": {"v10.1": 1.00, "n": 25},
      "llama-3.1-8b": {"v10.1": 0.80, "n": 25},
      "qwen-235b": {"v10.1": 0.92, "n": 25}
    },
    "interference": {},
    "catalysis": {
      "essentialist": {"effect": 0.12, "p": 0.03},
      "maieutic": {"effect": 0.08, "p": 0.11}
    }
  },
  
  "isotopes": null
}
```

#### `data/elements/skeptic/isotopes/method/definition.json`
```json
{
  "id": "skeptic-method",
  "parentElement": "skeptic",
  "symbol": "Σₘ",
  "name": "SKEPTIC-method",
  "version": "1.0",
  
  "description": {
    "short": "Methodological skepticism",
    "signature": "What was the methodology? Was this replicated?",
    "full": "Questions study design, sample sizes, control groups, and replication status. Distinct from premise-checking (Σₚ) which focuses on factual accuracy."
  },
  
  "focus": "Methodology critique",
  "triggerPatterns": [
    "Study/process design claims",
    "Research conclusions",
    "Causal claims from observational data"
  ],
  
  "empirical": {
    "triggerRates": {
      "phi-4": {"v10.2a": 1.00, "n": 25},
      "phi-3.5-mini": {"v10.2a": 0.92, "n": 25}
    },
    "distinctFrom": {
      "skeptic-premise": {
        "correlation": 0.34,
        "note": "Low correlation confirms distinct cognitive functions"
      }
    }
  }
}
```

### Tasks for Phase 1

- [ ] Create directory structure
- [ ] Define JSON schemas for all data types
- [ ] Migrate V10.1 training data to new format
- [ ] Extract empirical data from existing benchmark results
- [ ] Create isotope definitions for SKEPTIC (4 isotopes, validated)
- [ ] Create isotope predictions for CALIBRATOR (hypothesis: 3 isotopes)
- [ ] Build data validation scripts
- [ ] Implement version control for all data files

---

## Phase 2: Experiment Framework

### Goal
Enable hypothesis → prediction → experiment → validation workflow.

### Hypothesis Specification

```json
{
  "id": "H001",
  "title": "CALIBRATOR Isotope Decomposition",
  "statement": "CALIBRATOR decomposes into at least 3 distinct isotopes: confidence intervals (Κᵢ), risk assessment (Κᵣ), and uncertainty quantification (Κᵤ)",
  "falsifiable": true,
  "falsificationCriteria": "If training on all three isotopes does not improve trigger rate over single-isotope training by >15%, hypothesis is refuted",
  "status": "pending",
  "createdAt": "2026-01-19",
  "experiments": ["exp-001", "exp-002", "exp-003"],
  "outcome": null
}
```

### Experiment Specification

```json
{
  "id": "exp-001",
  "hypothesis": "H001",
  "title": "CALIBRATOR Single-Isotope Baseline",
  "description": "Train CALIBRATOR with generic uncertainty examples (no isotope differentiation)",
  
  "compound": {
    "elements": ["calibrator"],
    "isotopes": {},
    "sequence": ["calibrator"]
  },
  
  "architecture": "phi-4",
  
  "trainingConfig": {
    "baseModel": "mlx-community/phi-4-4bit",
    "method": "lora",
    "loraRank": 16,
    "loraLayers": 40,
    "learningRate": 1e-5,
    "iterations": 2500,
    "batchSize": 1,
    "seed": 42
  },
  
  "predictions": {
    "triggerRate": {
      "calibrator": {"mean": 0.75, "ci95": [0.60, 0.85]},
      "note": "Lower than validated elements due to less training data"
    },
    "mmluDelta": {"mean": 0.02, "ci95": [-0.02, 0.06]},
    "hallucinationDelta": {"mean": 0.05, "ci95": [0.00, 0.12]}
  },
  
  "validation": {
    "triggerTests": 25,
    "mmluQuestions": 100,
    "hallucinationTests": 48
  },
  
  "status": "planned",
  "scheduledFor": "2026-01-20"
}
```

### Prediction Engine

The prediction engine combines theoretical rules with empirical priors:

```python
class PredictionEngine:
    """
    Generates predictions for experiment outcomes based on:
    1. Theoretical element compatibility rules
    2. Empirical priors from similar experiments
    3. Architecture-specific adjustments
    """
    
    def predict_trigger_rates(
        self, 
        compound: CompoundSpec, 
        architecture: str
    ) -> Dict[str, PredictionWithCI]:
        """
        Predict trigger rates for each element in compound.
        
        Returns dict mapping element_id to prediction with 95% CI.
        """
        predictions = {}
        
        for element_id in compound.elements:
            # Get base rate from empirical data
            base_rate = self.get_empirical_prior(element_id, architecture)
            
            # Adjust for compound interactions
            interaction_adj = self.compute_interaction_adjustment(
                element_id, compound, architecture
            )
            
            # Adjust for training data quantity
            data_adj = self.compute_data_adjustment(element_id, compound)
            
            # Combine adjustments
            predicted_rate = base_rate * interaction_adj * data_adj
            
            # Compute confidence interval based on sample sizes
            ci = self.compute_confidence_interval(
                predicted_rate, 
                self.get_sample_size(element_id, architecture)
            )
            
            predictions[element_id] = PredictionWithCI(
                mean=predicted_rate,
                ci95_lower=ci[0],
                ci95_upper=ci[1],
                factors={
                    'base_rate': base_rate,
                    'interaction_adjustment': interaction_adj,
                    'data_adjustment': data_adj
                }
            )
        
        return predictions
    
    def predict_benchmark_delta(
        self,
        compound: CompoundSpec,
        architecture: str,
        benchmark: str
    ) -> PredictionWithCI:
        """
        Predict change in benchmark score from adding compound.
        """
        # Use regression model trained on historical experiment data
        features = self.extract_compound_features(compound, architecture)
        prediction = self.benchmark_model[benchmark].predict(features)
        ci = self.benchmark_model[benchmark].predict_interval(features, 0.95)
        
        return PredictionWithCI(
            mean=prediction,
            ci95_lower=ci[0],
            ci95_upper=ci[1]
        )
```

### Validation Analysis

```python
class ValidationAnalyzer:
    """
    Compares predictions to results and updates empirical priors.
    """
    
    def analyze_experiment(
        self, 
        experiment: ExperimentRun
    ) -> ValidationReport:
        """
        Full statistical analysis of experiment results.
        """
        report = ValidationReport(experiment_id=experiment.id)
        
        # Trigger rate analysis
        for element_id, predicted in experiment.predictions.trigger_rates.items():
            observed = experiment.results.trigger_rates[element_id]
            
            report.trigger_analysis[element_id] = TriggerAnalysis(
                predicted=predicted,
                observed=observed,
                within_ci=predicted.ci95_lower <= observed <= predicted.ci95_upper,
                deviation=(observed - predicted.mean) / predicted.mean,
                p_value=self.compute_binomial_p(observed, predicted.mean, experiment.validation.trigger_tests)
            )
        
        # Benchmark analysis
        for benchmark in ['mmlu', 'hallucination']:
            predicted = experiment.predictions[f'{benchmark}_delta']
            observed = experiment.results[f'{benchmark}_delta']
            
            report.benchmark_analysis[benchmark] = BenchmarkAnalysis(
                predicted=predicted,
                observed=observed,
                within_ci=predicted.ci95_lower <= observed <= predicted.ci95_upper,
                effect_size=self.compute_cohens_d(observed, experiment),
                p_value=self.compute_mcnemar_p(experiment, benchmark)
            )
        
        # Hypothesis evaluation
        report.hypothesis_status = self.evaluate_hypothesis(
            experiment.hypothesis,
            report
        )
        
        return report
    
    def update_empirical_priors(
        self, 
        experiment: ExperimentRun,
        report: ValidationReport
    ):
        """
        Update empirical data based on experiment results.
        """
        for element_id in experiment.compound.elements:
            element = load_element(element_id)
            
            # Update trigger rate
            new_rate = experiment.results.trigger_rates[element_id]
            element.empirical.trigger_rates[experiment.architecture][experiment.version] = {
                'rate': new_rate,
                'n': experiment.validation.trigger_tests,
                'ci95': wilson_score_interval(new_rate, experiment.validation.trigger_tests)
            }
            
            # Update interference data if compound has multiple elements
            if len(experiment.compound.elements) > 1:
                self.update_interference_data(element_id, experiment)
            
            save_element(element)
```

### Tasks for Phase 2

- [ ] Implement hypothesis registration system
- [ ] Build prediction engine with theoretical rules
- [ ] Add empirical prior integration to predictions
- [ ] Create experiment specification format
- [ ] Build experiment queue and scheduler
- [ ] Implement validation analyzer
- [ ] Create prior update mechanism
- [ ] Build hypothesis tracking dashboard

---

## Phase 3: Enhanced Interface

### Goal
Transform the visualization into an interactive research workbench.

### View Specifications

#### 1. Periodic Table View (Enhanced)

**Current:** Static element cards with theoretical data
**Enhanced:**
- Color saturation indicates empirical validation confidence
- Click element → show empirical data panel
- Toggle overlays:
  - Training coverage (examples per element)
  - Cross-architecture variance
  - Isotope availability
  - Last validation date
- Filter by: group, validation status, isotope availability

**Component:**
```jsx
function PeriodicTableView() {
  const [overlay, setOverlay] = useState('none');
  const [selectedElement, setSelectedElement] = useState(null);
  
  return (
    <div className="periodic-table">
      <OverlaySelector value={overlay} onChange={setOverlay} />
      
      <ElementGrid>
        {elements.map(element => (
          <ElementCard
            key={element.id}
            element={element}
            overlay={overlay}
            onClick={() => setSelectedElement(element.id)}
            empiricalData={useEmpiricalData(element.id)}
          />
        ))}
      </ElementGrid>
      
      {selectedElement && (
        <ElementDetailPanel
          elementId={selectedElement}
          onClose={() => setSelectedElement(null)}
        />
      )}
    </div>
  );
}
```

#### 2. Isotope Editor

**Purpose:** Manage element isotopes as first-class entities

**Features:**
- Expand any element to show isotopes
- Create new isotope definitions
- Independent training data per isotope
- Independent validation per isotope
- Isotope-level stability predictions

**Component:**
```jsx
function IsotopeEditor({ elementId }) {
  const element = useElement(elementId);
  const [editingIsotope, setEditingIsotope] = useState(null);
  
  return (
    <div className="isotope-editor">
      <h3>{element.symbol} Isotopes</h3>
      
      {element.isotopes ? (
        <IsotopeList>
          {Object.entries(element.isotopes).map(([id, isotope]) => (
            <IsotopeCard
              key={id}
              isotope={isotope}
              onEdit={() => setEditingIsotope(id)}
              empiricalData={useEmpiricalData(`${elementId}-${id}`)}
            />
          ))}
        </IsotopeList>
      ) : (
        <EmptyState>
          <p>No isotopes defined for {element.name}</p>
          <Button onClick={() => setEditingIsotope('new')}>
            Propose Isotopes
          </Button>
        </EmptyState>
      )}
      
      <Button onClick={() => setEditingIsotope('new')}>
        Add Isotope
      </Button>
      
      {editingIsotope && (
        <IsotopeEditorModal
          elementId={elementId}
          isotopeId={editingIsotope}
          onClose={() => setEditingIsotope(null)}
        />
      )}
    </div>
  );
}
```

#### 3. Experiment Designer

**Purpose:** Design and queue experiments from the interface

**Features:**
- Visual compound builder (existing)
- Hypothesis statement field
- Auto-calculated predictions
- Architecture selector
- Training config editor
- "Run Experiment" button

**Component:**
```jsx
function ExperimentDesigner() {
  const [compound, setCompound] = useState({ elements: [], isotopes: {}, sequence: [] });
  const [hypothesis, setHypothesis] = useState('');
  const [architecture, setArchitecture] = useState('phi-4');
  const [config, setConfig] = useState(defaultTrainingConfig);
  
  const predictions = usePredictions(compound, architecture);
  
  const handleSubmit = async () => {
    const experiment = {
      compound,
      hypothesis,
      architecture,
      config,
      predictions
    };
    
    await submitExperiment(experiment);
  };
  
  return (
    <div className="experiment-designer">
      <section className="hypothesis-section">
        <h3>Hypothesis</h3>
        <textarea
          value={hypothesis}
          onChange={e => setHypothesis(e.target.value)}
          placeholder="State your falsifiable hypothesis..."
        />
      </section>
      
      <section className="compound-section">
        <h3>Compound Design</h3>
        <CompoundBuilder
          compound={compound}
          onChange={setCompound}
        />
      </section>
      
      <section className="predictions-section">
        <h3>Predictions (Auto-calculated)</h3>
        <PredictionDisplay predictions={predictions} />
      </section>
      
      <section className="config-section">
        <h3>Training Configuration</h3>
        <ArchitectureSelector
          value={architecture}
          onChange={setArchitecture}
        />
        <TrainingConfigEditor
          value={config}
          onChange={setConfig}
        />
      </section>
      
      <Button onClick={handleSubmit} primary>
        Queue Experiment
      </Button>
    </div>
  );
}
```

#### 4. Results Dashboard

**Purpose:** Track experiments and analyze results

**Features:**
- Experiment history with status indicators
- Prediction vs. actual comparison charts
- Statistical analysis panel
- Hypothesis tracker

**Component:**
```jsx
function ResultsDashboard() {
  const experiments = useExperiments();
  const [selectedExperiment, setSelectedExperiment] = useState(null);
  
  return (
    <div className="results-dashboard">
      <ExperimentList>
        {experiments.map(exp => (
          <ExperimentRow
            key={exp.id}
            experiment={exp}
            onClick={() => setSelectedExperiment(exp.id)}
          />
        ))}
      </ExperimentList>
      
      {selectedExperiment && (
        <ExperimentDetail experimentId={selectedExperiment}>
          <PredictionVsActualChart />
          <StatisticalAnalysisPanel />
          <ArtifactLinks />
        </ExperimentDetail>
      )}
      
      <HypothesisTracker />
    </div>
  );
}
```

#### 5. Manifold Explorer (Enhanced)

**Purpose:** Visualize cognitive space with empirical grounding

**Features:**
- Theoretical vs. empirical position toggle
- Confidence ellipses around measured positions
- Trajectory view (position changes across versions)
- Interference heatmap overlay
- Compound coverage polygon

**Component:**
```jsx
function ManifoldExplorer() {
  const [mode, setMode] = useState('empirical'); // 'theoretical' | 'empirical'
  const [showConfidence, setShowConfidence] = useState(true);
  const [showTrajectory, setShowTrajectory] = useState(false);
  const [overlay, setOverlay] = useState('none'); // 'interference' | 'coverage' | 'none'
  
  return (
    <div className="manifold-explorer">
      <ManifoldControls
        mode={mode}
        onModeChange={setMode}
        showConfidence={showConfidence}
        onConfidenceChange={setShowConfidence}
        showTrajectory={showTrajectory}
        onTrajectoryChange={setShowTrajectory}
        overlay={overlay}
        onOverlayChange={setOverlay}
      />
      
      <ManifoldVisualization
        mode={mode}
        showConfidence={showConfidence}
        showTrajectory={showTrajectory}
        overlay={overlay}
      />
      
      <ManifoldLegend />
    </div>
  );
}
```

#### 6. Publication Export

**Purpose:** Generate peer-review ready outputs

**Features:**
- LaTeX table generator
- SVG/PDF figure export
- Methodology section text
- Reproducibility package

**Component:**
```jsx
function PublicationExport() {
  const [selectedExperiments, setSelectedExperiments] = useState([]);
  const [exportFormat, setExportFormat] = useState('latex');
  
  return (
    <div className="publication-export">
      <ExperimentSelector
        selected={selectedExperiments}
        onChange={setSelectedExperiments}
      />
      
      <ExportOptions>
        <FormatSelector value={exportFormat} onChange={setExportFormat} />
        
        <ExportButtons>
          <Button onClick={() => exportTables(selectedExperiments, exportFormat)}>
            Export Tables
          </Button>
          <Button onClick={() => exportFigures(selectedExperiments)}>
            Export Figures
          </Button>
          <Button onClick={() => exportMethodology(selectedExperiments)}>
            Export Methodology
          </Button>
          <Button onClick={() => exportReproductionPackage(selectedExperiments)}>
            Full Reproduction Package
          </Button>
        </ExportButtons>
      </ExportOptions>
      
      <PreviewPanel format={exportFormat} experiments={selectedExperiments} />
    </div>
  );
}
```

### Tasks for Phase 3

- [ ] Redesign periodic table with empirical overlays
- [ ] Build isotope editor component
- [ ] Create experiment designer view
- [ ] Build results dashboard
- [ ] Enhance manifold explorer with empirical data
- [ ] Implement publication export system
- [ ] Add navigation between views
- [ ] Create responsive layout for all views

---

## Phase 4: Statistical Rigor

### Goal
Meet peer-review standards for statistical claims.

### Required Statistical Analyses

#### 1. Trigger Rate Validation

**Requirements:**
- N ≥ 20 per element per architecture
- Wilson score 95% confidence intervals
- Report exact p-values

**Implementation:**
```python
def wilson_score_interval(successes: int, trials: int, confidence: float = 0.95) -> Tuple[float, float]:
    """
    Compute Wilson score confidence interval for binomial proportion.
    More accurate than normal approximation for small samples.
    """
    from scipy import stats
    
    if trials == 0:
        return (0.0, 1.0)
    
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    p_hat = successes / trials
    
    denominator = 1 + z**2 / trials
    center = (p_hat + z**2 / (2 * trials)) / denominator
    margin = z * np.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * trials)) / trials) / denominator
    
    return (max(0, center - margin), min(1, center + margin))
```

#### 2. Benchmark Comparison

**Requirements:**
- McNemar's test for paired comparisons
- Effect size (Cohen's d)
- Multiple comparison correction

**Implementation:**
```python
def mcnemar_test(base_correct: List[bool], trained_correct: List[bool]) -> McNeTestResult:
    """
    McNemar's test for comparing paired binary outcomes.
    """
    from scipy import stats
    
    # Build contingency table
    # b = base correct, trained wrong
    # c = base wrong, trained correct
    b = sum(1 for base, trained in zip(base_correct, trained_correct) if base and not trained)
    c = sum(1 for base, trained in zip(base_correct, trained_correct) if not base and trained)
    
    # McNemar's statistic
    if b + c == 0:
        return McNeTestResult(statistic=0, p_value=1.0, b=b, c=c)
    
    # With continuity correction
    statistic = (abs(b - c) - 1)**2 / (b + c)
    p_value = 1 - stats.chi2.cdf(statistic, df=1)
    
    return McNeTestResult(statistic=statistic, p_value=p_value, b=b, c=c)


def cohens_d(group1: List[float], group2: List[float]) -> float:
    """
    Compute Cohen's d effect size.
    """
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    
    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    return (np.mean(group1) - np.mean(group2)) / pooled_std
```

#### 3. Cross-Architecture Analysis

**Requirements:**
- Report variance across architectures
- Identify architecture-specific effects
- Meta-analysis across architectures

**Implementation:**
```python
def cross_architecture_analysis(results: Dict[str, ExperimentResults]) -> CrossArchAnalysis:
    """
    Analyze results across multiple architectures.
    """
    # Collect trigger rates per element across architectures
    element_rates = defaultdict(list)
    for arch, result in results.items():
        for element, rate in result.trigger_rates.items():
            element_rates[element].append((arch, rate))
    
    analysis = CrossArchAnalysis()
    
    for element, rates in element_rates.items():
        rate_values = [r[1] for r in rates]
        
        analysis.elements[element] = ElementCrossArchAnalysis(
            mean=np.mean(rate_values),
            std=np.std(rate_values),
            min_arch=min(rates, key=lambda x: x[1]),
            max_arch=max(rates, key=lambda x: x[1]),
            variance_significant=np.std(rate_values) > 0.15  # Threshold for concern
        )
    
    return analysis
```

### Statistical Dashboard Components

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ STATISTICAL ANALYSIS: SKEPTIC                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ TRIGGER RATES BY ARCHITECTURE                                               │
│ ═══════════════════════════════════════════════════════════════════════════ │
│                                                                             │
│ Architecture    N     Rate    95% CI          vs Phi-4 (ref)               │
│ ───────────────────────────────────────────────────────────────────────────│
│ Phi-4          25    92.0%   [74.0%, 98.9%]  —                             │
│ Phi-3.5        25    98.0%   [83.9%, 99.9%]  p=0.317 (ns)                  │
│ Llama 3.1      25    76.0%   [54.9%, 90.6%]  p=0.046 *                     │
│ Qwen 235B      25    88.0%   [69.0%, 97.6%]  p=0.564 (ns)                  │
│                                                                             │
│ Cross-architecture variance: σ = 9.2%                                       │
│ Heterogeneity: I² = 42.3% (moderate)                                        │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│ ISOTOPE BREAKDOWN                                                           │
│ ═══════════════════════════════════════════════════════════════════════════ │
│                                                                             │
│ Isotope      Mean Rate    Cross-Arch σ    Training N    Status             │
│ ───────────────────────────────────────────────────────────────────────────│
│ Σₚ (premise)   100.0%        0.0%            12         ✓ Validated        │
│ Σₘ (method)     88.0%       12.0%             6         ⚠ Needs data       │
│ Σₛ (source)     92.0%        8.0%             6         ⚠ Needs data       │
│ Σₜ (stats)      84.0%       14.0%             6         ⚠ Needs data       │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│ INTERFERENCE ANALYSIS                                                       │
│ ═══════════════════════════════════════════════════════════════════════════ │
│                                                                             │
│ Element Pair         Pearson r    p-value    Interpretation                │
│ ───────────────────────────────────────────────────────────────────────────│
│ SKEPTIC + GENERATOR   -0.34       0.024 *    Significant negative          │
│ SKEPTIC + ARCHITECT   +0.12       0.310      No significant effect         │
│ SKEPTIC + SOLITON     +0.28       0.048 *    Positive synergy              │
│ SKEPTIC + ESSENTIALIST +0.19      0.187      No significant effect         │
│                                                                             │
│ * p < 0.05                                                                  │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Tasks for Phase 4

- [ ] Implement Wilson score interval calculation
- [ ] Implement McNemar's test
- [ ] Implement Cohen's d effect size
- [ ] Build cross-architecture analysis
- [ ] Add multiple comparison correction (Bonferroni/FDR)
- [ ] Create statistical dashboard component
- [ ] Add confidence interval display throughout interface
- [ ] Implement reproducibility checks

---

## Phase 5: Training Pipeline Integration

### Goal
Seamless connection between interface and actual training infrastructure.

### Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         TRAINING PIPELINE                                 │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────────────────┐   │
│  │   React     │────▶│   FastAPI   │────▶│   Training Backends     │   │
│  │   Frontend  │◀────│   Server    │◀────│                         │   │
│  └─────────────┘     └─────────────┘     │  ┌─────────────────┐   │   │
│        │                    │             │  │  MLX (local)    │   │   │
│        │                    │             │  │  - Phi-4        │   │   │
│        │ WebSocket          │             │  │  - Phi-3.5      │   │   │
│        │                    │             │  │  - Llama 3.1    │   │   │
│        ▼                    │             │  └─────────────────┘   │   │
│  ┌─────────────┐           │             │                         │   │
│  │  Progress   │           │             │  ┌─────────────────┐   │   │
│  │  Updates    │           │             │  │  Tinker (cloud) │   │   │
│  └─────────────┘           │             │  │  - Qwen 235B    │   │   │
│                            │             │  │  - DeepSeek V3  │   │   │
│                            │             │  └─────────────────┘   │   │
│                            │             └─────────────────────────┘   │
│                            │                                           │
│                            ▼                                           │
│                    ┌─────────────┐                                     │
│                    │  Benchmark  │                                     │
│                    │    Suite    │                                     │
│                    └─────────────┘                                     │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

### Local Training Pipeline (MLX)

```python
# src/training/local_pipeline.py

import mlx.core as mx
from mlx_lm import load, generate
from mlx_lm.tuner import train as mlx_train
from pathlib import Path
import asyncio
import json

class LocalTrainingPipeline:
    """
    MLX-based local training pipeline.
    """
    
    def __init__(
        self,
        experiment: ExperimentRun,
        data_dir: Path,
        output_dir: Path
    ):
        self.experiment = experiment
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.progress_callback = None
    
    async def prepare_data(self) -> Path:
        """
        Compile training data from element specs.
        """
        examples = []
        
        for element_id in self.experiment.compound.elements:
            element = load_element(self.data_dir / 'elements' / element_id)
            
            # Add base element examples
            examples.extend(element.training.examples)
            
            # Add isotope examples if specified
            if element_id in self.experiment.compound.isotopes:
                for isotope_id in self.experiment.compound.isotopes[element_id]:
                    isotope = element.isotopes[isotope_id]
                    examples.extend(isotope.training.examples)
        
        # Write to JSONL format for MLX
        data_path = self.output_dir / 'training_data.jsonl'
        with open(data_path, 'w') as f:
            for ex in examples:
                f.write(json.dumps({
                    'text': f"<|user|>\n{ex.instruction}\n{ex.input}<|end|>\n<|assistant|>\n{ex.output}<|end|>"
                }) + '\n')
        
        return data_path
    
    async def train(self, progress_callback=None):
        """
        Execute training with progress updates.
        """
        self.progress_callback = progress_callback
        
        data_path = await self.prepare_data()
        
        config = self.experiment.training_config
        
        # MLX training configuration
        train_args = {
            'model': config.base_model,
            'data': str(data_path),
            'train': True,
            'iters': config.iterations,
            'batch_size': config.batch_size,
            'learning_rate': config.learning_rate,
            'lora_layers': config.lora_layers,
            'lora_rank': config.lora_rank,
            'seed': config.seed,
            'adapter_path': str(self.output_dir / 'adapters'),
        }
        
        # Run training with progress monitoring
        async for progress in self._run_training(train_args):
            if self.progress_callback:
                await self.progress_callback(progress)
        
        return self.output_dir / 'adapters'
    
    async def _run_training(self, args):
        """
        Wrap MLX training with async progress updates.
        """
        # This would wrap the actual MLX training loop
        # For now, placeholder for the integration
        pass
    
    async def validate(self) -> ValidationResults:
        """
        Run benchmark suite on trained model.
        """
        adapter_path = self.output_dir / 'adapters'
        
        results = ValidationResults()
        
        # Load model with adapter
        model, tokenizer = load(
            self.experiment.training_config.base_model,
            adapter_path=str(adapter_path)
        )
        
        # Run trigger tests
        results.trigger_rates = await self._run_trigger_tests(model, tokenizer)
        
        # Run MMLU
        results.mmlu = await self._run_mmlu(model, tokenizer)
        
        # Run hallucination tests
        results.hallucination = await self._run_hallucination_tests(model, tokenizer)
        
        return results
    
    async def _run_trigger_tests(self, model, tokenizer) -> Dict[str, TriggerResult]:
        """
        Test each element's trigger rate.
        """
        results = {}
        
        for element_id in self.experiment.compound.elements:
            element = load_element(self.data_dir / 'elements' / element_id)
            
            successes = 0
            for prompt in element.validation.trigger_prompts:
                response = generate(model, tokenizer, prompt=prompt, max_tokens=500)
                
                # Check if response matches expected patterns
                if any(pattern.search(response) for pattern in element.validation.expected_patterns):
                    successes += 1
            
            n = len(element.validation.trigger_prompts)
            rate = successes / n
            ci = wilson_score_interval(successes, n)
            
            results[element_id] = TriggerResult(
                rate=rate,
                successes=successes,
                trials=n,
                ci95=ci
            )
        
        return results
```

### Cloud Training Pipeline (Tinker)

```python
# src/training/tinker_pipeline.py

import httpx
import asyncio
from typing import AsyncIterator

class TinkerTrainingPipeline:
    """
    Tinker API-based cloud training pipeline.
    """
    
    def __init__(
        self,
        experiment: ExperimentRun,
        api_key: str,
        data_dir: Path
    ):
        self.experiment = experiment
        self.api_key = api_key
        self.data_dir = data_dir
        self.base_url = "https://api.thinkingmachines.ai/tinker"
        self.session_id = None
    
    async def initialize_session(self, model: str = "Qwen/Qwen3-235B-A22B-Instruct-2507"):
        """
        Initialize Tinker training session.
        """
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/session",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": model,
                    "lora_rank": self.experiment.training_config.lora_rank
                }
            )
            self.session_id = response.json()["session_id"]
        
        return self.session_id
    
    async def train(self, progress_callback=None) -> AsyncIterator[TrainingProgress]:
        """
        Execute training with progress streaming.
        """
        examples = await self._prepare_examples()
        
        config = self.experiment.training_config
        
        for epoch in range(config.epochs):
            # Forward-backward pass
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/session/{self.session_id}/forward_backward",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={"examples": examples}
                )
                loss = response.json()["loss"]
            
            # Optimizer step
            async with httpx.AsyncClient() as client:
                await client.post(
                    f"{self.base_url}/session/{self.session_id}/optim_step",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={"learning_rate": config.learning_rate}
                )
            
            progress = TrainingProgress(
                epoch=epoch + 1,
                total_epochs=config.epochs,
                loss=loss
            )
            
            if progress_callback:
                await progress_callback(progress)
            
            yield progress
    
    async def export_adapter(self, format: str = "safetensors") -> bytes:
        """
        Export trained adapter weights.
        """
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}/session/{self.session_id}/export",
                headers={"Authorization": f"Bearer {self.api_key}"},
                params={"format": format}
            )
            return response.content
```

### FastAPI Server

```python
# src/server/main.py

from fastapi import FastAPI, WebSocket, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import asyncio

app = FastAPI(title="Cognitive Elements Research Instrument")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Experiment queue
experiment_queue = asyncio.Queue()
active_experiments = {}

@app.post("/experiments")
async def submit_experiment(experiment: ExperimentSubmission, background_tasks: BackgroundTasks):
    """
    Submit a new experiment to the queue.
    """
    exp_id = generate_experiment_id()
    experiment_run = create_experiment_run(exp_id, experiment)
    
    await experiment_queue.put(experiment_run)
    active_experiments[exp_id] = experiment_run
    
    background_tasks.add_task(process_experiment_queue)
    
    return {"experiment_id": exp_id, "status": "queued"}

@app.get("/experiments/{exp_id}")
async def get_experiment(exp_id: str):
    """
    Get experiment status and results.
    """
    if exp_id in active_experiments:
        return active_experiments[exp_id]
    
    # Load from disk if not in memory
    return load_experiment(exp_id)

@app.websocket("/experiments/{exp_id}/progress")
async def experiment_progress(websocket: WebSocket, exp_id: str):
    """
    WebSocket for real-time training progress.
    """
    await websocket.accept()
    
    while True:
        if exp_id in active_experiments:
            exp = active_experiments[exp_id]
            await websocket.send_json({
                "status": exp.status,
                "progress": exp.progress,
                "metrics": exp.current_metrics
            })
        
        await asyncio.sleep(1)

@app.get("/elements")
async def list_elements():
    """
    List all elements with empirical data.
    """
    return load_element_manifest()

@app.get("/elements/{element_id}")
async def get_element(element_id: str):
    """
    Get element details including empirical data.
    """
    return load_element(element_id)

@app.get("/predictions")
async def get_predictions(compound: CompoundSpec, architecture: str):
    """
    Get predictions for a compound configuration.
    """
    engine = PredictionEngine()
    return engine.predict(compound, architecture)
```

### Tasks for Phase 5

- [ ] Implement LocalTrainingPipeline for MLX
- [ ] Implement TinkerTrainingPipeline for cloud
- [ ] Build FastAPI server with experiment endpoints
- [ ] Implement WebSocket progress streaming
- [ ] Create benchmark runner integration
- [ ] Build training queue management
- [ ] Add adapter weight storage and versioning
- [ ] Implement training logs collection

---

## Phase 6: Publication Support

### Goal
Generate peer-review ready outputs.

### LaTeX Table Generator

```python
# src/export/latex_tables.py

def generate_trigger_rate_table(
    experiments: List[ExperimentRun],
    caption: str = "Trigger rates for Cognitive Kernel across architectures"
) -> str:
    """
    Generate LaTeX table for trigger rates.
    """
    # Collect data
    architectures = sorted(set(exp.architecture for exp in experiments))
    elements = sorted(set(
        elem for exp in experiments 
        for elem in exp.results.trigger_rates.keys()
    ))
    
    # Build table
    latex = [
        r"\begin{table}[h]",
        r"\centering",
        f"\\caption{{{caption}}}",
        r"\begin{tabular}{l" + "c" * len(architectures) + "}",
        r"\toprule",
        "Element & " + " & ".join(architectures) + r" \\",
        r"\midrule",
    ]
    
    for element in elements:
        row = [element.upper()]
        for arch in architectures:
            exp = next((e for e in experiments if e.architecture == arch), None)
            if exp and element in exp.results.trigger_rates:
                result = exp.results.trigger_rates[element]
                rate_str = f"{result.rate*100:.0f}\\%"
                if result.p_value and result.p_value < 0.05:
                    rate_str += "*"
                row.append(rate_str)
            else:
                row.append("—")
        latex.append(" & ".join(row) + r" \\")
    
    latex.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\label{tab:trigger-rates}",
        r"\footnotesize{* p < 0.05 vs. reference architecture. N=25 per cell.}",
        r"\end{table}",
    ])
    
    return "\n".join(latex)


def generate_benchmark_comparison_table(
    base_results: BenchmarkResults,
    trained_results: BenchmarkResults,
    caption: str = "Benchmark comparison: base model vs. Cognitive Kernel"
) -> str:
    """
    Generate LaTeX table comparing benchmarks.
    """
    latex = [
        r"\begin{table}[h]",
        r"\centering",
        f"\\caption{{{caption}}}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Benchmark & Base & Trained & $\Delta$ & 95\% CI & p-value \\",
        r"\midrule",
    ]
    
    for benchmark in ['MMLU', 'Hallucination Resistance']:
        base = getattr(base_results, benchmark.lower().replace(' ', '_'))
        trained = getattr(trained_results, benchmark.lower().replace(' ', '_'))
        delta = trained.score - base.score
        ci = trained.ci95
        p = trained.p_value
        
        p_str = f"{p:.3f}" if p >= 0.001 else "<0.001"
        sig = "*" if p < 0.05 else ""
        
        latex.append(
            f"{benchmark} & {base.score*100:.1f}\\% & {trained.score*100:.1f}\\% & "
            f"+{delta*100:.1f}\\% & [{ci[0]*100:.1f}\\%, {ci[1]*100:.1f}\\%] & {p_str}{sig} \\\\"
        )
    
    latex.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\label{tab:benchmark-comparison}",
        r"\footnotesize{* p < 0.05 (McNemar's test)}",
        r"\end{table}",
    ])
    
    return "\n".join(latex)
```

### Methodology Section Generator

```python
# src/export/methodology.py

def generate_methodology_section(
    experiments: List[ExperimentRun],
    include_training_details: bool = True,
    include_evaluation_protocol: bool = True
) -> str:
    """
    Generate methodology section text.
    """
    sections = []
    
    # Training data description
    sections.append("\\subsection{Training Data}\n")
    
    total_examples = sum(
        len(load_element(elem).training.examples)
        for exp in experiments
        for elem in exp.compound.elements
    )
    
    sections.append(f"""
The Cognitive Kernel training dataset consists of {total_examples} examples
spanning {len(get_all_elements())} cognitive elements organized into 
{len(get_all_groups())} functional groups. Each example follows the 
instruction-input-output format, where the instruction specifies the 
desired cognitive orientation, the input provides context, and the output
demonstrates the target behavior.

Training examples were derived from two sources: (1) extraction from 
Claude Opus 4.5 responses to introspection probes, formalized into 
transferable patterns, and (2) human-authored examples designed to 
cover edge cases identified through iterative testing.
""")
    
    if include_training_details:
        sections.append("\\subsection{Training Procedure}\n")
        
        # Get representative config
        config = experiments[0].training_config
        
        sections.append(f"""
Models were fine-tuned using Low-Rank Adaptation (LoRA) with rank 
{config.lora_rank} applied to {config.lora_layers} transformer layers.
Training was conducted for {config.iterations} iterations with a 
learning rate of {config.learning_rate} and batch size of 
{config.batch_size}. All experiments used seed {config.seed} for 
reproducibility.

For local training, we used MLX on Apple Silicon hardware. For 
large-scale models (>100B parameters), we used the Tinker cloud 
training API.
""")
    
    if include_evaluation_protocol:
        sections.append("\\subsection{Evaluation Protocol}\n")
        
        sections.append("""
Evaluation consisted of three components:

\\textbf{Trigger Rate Assessment:} For each cognitive element, we 
administered 25 probe prompts designed to elicit the target cognitive
orientation. Responses were classified as successful triggers if they
matched predefined linguistic patterns characteristic of each element.
Wilson score 95\\% confidence intervals are reported for all rates.

\\textbf{Capability Benchmarks:} We evaluated models on a 100-question
subset of MMLU to assess whether cognitive training degraded general
capabilities. McNemar's test was used for paired comparison with the
base model.

\\textbf{Hallucination Resistance:} We administered 48 prompts designed
to elicit hallucinations across categories including false premises,
future events, and fictitious entities. Responses were scored for
appropriate uncertainty expression.
""")
    
    return "\n".join(sections)
```

### Reproduction Package Generator

```python
# src/export/reproduction.py

def generate_reproduction_package(
    experiments: List[ExperimentRun],
    output_dir: Path
) -> Path:
    """
    Generate complete reproduction package.
    """
    package_dir = output_dir / "reproduction_package"
    package_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Training data
    data_dir = package_dir / "data"
    data_dir.mkdir()
    
    for exp in experiments:
        exp_data_dir = data_dir / exp.id
        exp_data_dir.mkdir()
        
        # Copy training examples
        shutil.copy(
            exp.artifacts.training_data,
            exp_data_dir / "training.jsonl"
        )
        
        # Copy validation prompts
        shutil.copy(
            exp.artifacts.validation_prompts,
            exp_data_dir / "validation.json"
        )
    
    # 2. Configuration files
    config_dir = package_dir / "configs"
    config_dir.mkdir()
    
    for exp in experiments:
        with open(config_dir / f"{exp.id}.json", 'w') as f:
            json.dump(exp.training_config.dict(), f, indent=2)
    
    # 3. Scripts
    scripts_dir = package_dir / "scripts"
    scripts_dir.mkdir()
    
    # Training script
    with open(scripts_dir / "train.py", 'w') as f:
        f.write(generate_training_script(experiments))
    
    # Evaluation script
    with open(scripts_dir / "evaluate.py", 'w') as f:
        f.write(generate_evaluation_script(experiments))
    
    # 4. Environment specification
    with open(package_dir / "requirements.txt", 'w') as f:
        f.write(generate_requirements())
    
    with open(package_dir / "environment.yml", 'w') as f:
        f.write(generate_conda_env())
    
    # 5. README
    with open(package_dir / "README.md", 'w') as f:
        f.write(generate_reproduction_readme(experiments))
    
    # 6. Results for verification
    results_dir = package_dir / "expected_results"
    results_dir.mkdir()
    
    for exp in experiments:
        with open(results_dir / f"{exp.id}_results.json", 'w') as f:
            json.dump(exp.results.dict(), f, indent=2)
    
    return package_dir
```

### Tasks for Phase 6

- [ ] Implement LaTeX table generators
- [ ] Build methodology section generator
- [ ] Create figure export (SVG/PDF)
- [ ] Implement reproduction package generator
- [ ] Add DOI generation for datasets
- [ ] Create supplementary materials formatter
- [ ] Build one-click export for full paper package

---

## Architecture Design

### Recommended: Hybrid Web + Local

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                              SYSTEM ARCHITECTURE                              │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌────────────────────────────────────────────────────────────────────┐    │
│   │                        WEB INTERFACE (React)                        │    │
│   │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ │    │
│   │  │ Periodic │ │ Isotope  │ │Experiment│ │ Results  │ │  Export  │ │    │
│   │  │  Table   │ │  Editor  │ │ Designer │ │Dashboard │ │  Tools   │ │    │
│   │  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘ │    │
│   └────────────────────────────────────────────────────────────────────┘    │
│                                    │                                         │
│                                    │ HTTP/WebSocket                          │
│                                    ▼                                         │
│   ┌────────────────────────────────────────────────────────────────────┐    │
│   │                      FASTAPI SERVER (Python)                        │    │
│   │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ │    │
│   │  │   Data   │ │Prediction│ │Experiment│ │ Analysis │ │  Export  │ │    │
│   │  │   API    │ │  Engine  │ │  Queue   │ │  Engine  │ │  Engine  │ │    │
│   │  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘ │    │
│   └────────────────────────────────────────────────────────────────────┘    │
│                          │                    │                              │
│            ┌─────────────┴────────┐          │                              │
│            ▼                      ▼          ▼                              │
│   ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐         │
│   │   LOCAL DAEMON   │  │   TINKER API     │  │    DATA STORE    │         │
│   │   (MLX Training) │  │  (Cloud Train)   │  │   (JSON/SQLite)  │         │
│   │                  │  │                  │  │                  │         │
│   │  • Phi-4         │  │  • Qwen 235B     │  │  • Elements      │         │
│   │  • Phi-3.5       │  │  • DeepSeek V3   │  │  • Experiments   │         │
│   │  • Llama 3.1     │  │  • Kimi K2       │  │  • Results       │         │
│   └──────────────────┘  └──────────────────┘  └──────────────────┘         │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Technology Stack

| Layer | Technology | Rationale |
|-------|------------|-----------|
| Frontend | React + TypeScript | Existing codebase, rich visualization |
| State Management | Zustand or Redux | Complex state across views |
| Visualization | D3.js + React | Publication-quality figures |
| API Server | FastAPI | Async, type-safe, Python ecosystem |
| Local Training | MLX | Apple Silicon optimization |
| Cloud Training | Tinker API | Large model access |
| Data Storage | JSON + SQLite | Simple, portable, versioned |
| Statistics | SciPy + statsmodels | Comprehensive statistical tests |
| Export | Jinja2 + matplotlib | LaTeX and figure generation |

### File Structure

```
cognitive-elements-research/
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── PeriodicTable/
│   │   │   ├── IsotopeEditor/
│   │   │   ├── ExperimentDesigner/
│   │   │   ├── ResultsDashboard/
│   │   │   ├── ManifoldExplorer/
│   │   │   └── PublicationExport/
│   │   ├── hooks/
│   │   ├── stores/
│   │   ├── api/
│   │   └── utils/
│   ├── package.json
│   └── vite.config.ts
│
├── backend/
│   ├── src/
│   │   ├── server/
│   │   │   ├── main.py
│   │   │   ├── routes/
│   │   │   └── websocket/
│   │   ├── training/
│   │   │   ├── local_pipeline.py
│   │   │   ├── tinker_pipeline.py
│   │   │   └── benchmark_runner.py
│   │   ├── analysis/
│   │   │   ├── prediction_engine.py
│   │   │   ├── validation_analyzer.py
│   │   │   └── statistics.py
│   │   ├── export/
│   │   │   ├── latex_tables.py
│   │   │   ├── methodology.py
│   │   │   ├── figures.py
│   │   │   └── reproduction.py
│   │   └── data/
│   │       ├── schemas.py
│   │       ├── loader.py
│   │       └── validator.py
│   ├── requirements.txt
│   └── pyproject.toml
│
├── data/
│   ├── elements/
│   ├── compounds/
│   ├── experiments/
│   └── benchmarks/
│
├── scripts/
│   ├── setup.sh
│   ├── migrate_v10_data.py
│   └── run_all_benchmarks.py
│
├── tests/
│   ├── frontend/
│   ├── backend/
│   └── integration/
│
├── docs/
│   ├── API.md
│   ├── SCHEMAS.md
│   └── CONTRIBUTING.md
│
├── docker-compose.yml
├── Makefile
└── README.md
```

---

## Implementation Roadmap

### Week 1-2: Data Foundation

| Task | Priority | Estimated Hours |
|------|----------|-----------------|
| Define JSON schemas for all data types | High | 8 |
| Create directory structure | High | 2 |
| Migrate V10.1 training data to new format | High | 8 |
| Extract empirical data from benchmark results | High | 4 |
| Create SKEPTIC isotope definitions (validated) | High | 4 |
| Create CALIBRATOR isotope predictions | Medium | 4 |
| Build data validation scripts | Medium | 4 |
| Implement version control for data files | Medium | 4 |
| **Total** | | **38 hours** |

### Week 3-4: Experiment Framework

| Task | Priority | Estimated Hours |
|------|----------|-----------------|
| Implement hypothesis registration system | High | 6 |
| Build prediction engine (theoretical rules) | High | 8 |
| Add empirical prior integration | High | 6 |
| Create experiment specification format | High | 4 |
| Build experiment queue and scheduler | Medium | 8 |
| Implement validation analyzer | High | 8 |
| Create prior update mechanism | Medium | 4 |
| Build hypothesis tracking | Medium | 4 |
| **Total** | | **48 hours** |

### Week 5-6: Enhanced Interface

| Task | Priority | Estimated Hours |
|------|----------|-----------------|
| Redesign periodic table with empirical overlays | High | 12 |
| Build isotope editor component | High | 8 |
| Create experiment designer view | High | 12 |
| Build results dashboard | High | 10 |
| Enhance manifold explorer | Medium | 8 |
| Implement navigation between views | Medium | 4 |
| Create responsive layout | Medium | 6 |
| **Total** | | **60 hours** |

### Week 7-8: Training Integration

| Task | Priority | Estimated Hours |
|------|----------|-----------------|
| Implement LocalTrainingPipeline (MLX) | High | 12 |
| Implement TinkerTrainingPipeline | High | 8 |
| Build FastAPI server | High | 10 |
| Implement WebSocket progress streaming | Medium | 6 |
| Create benchmark runner integration | High | 8 |
| Build training queue management | Medium | 6 |
| Add adapter weight storage | Medium | 4 |
| Implement training logs collection | Low | 4 |
| **Total** | | **58 hours** |

### Week 9-10: Statistical Rigor

| Task | Priority | Estimated Hours |
|------|----------|-----------------|
| Implement Wilson score intervals | High | 2 |
| Implement McNemar's test | High | 2 |
| Implement Cohen's d | High | 2 |
| Build cross-architecture analysis | High | 6 |
| Add multiple comparison correction | Medium | 4 |
| Create statistical dashboard component | High | 8 |
| Add CI display throughout interface | Medium | 6 |
| Implement reproducibility checks | Medium | 4 |
| **Total** | | **34 hours** |

### Week 11-12: Publication Support

| Task | Priority | Estimated Hours |
|------|----------|-----------------|
| Implement LaTeX table generators | High | 8 |
| Build methodology section generator | High | 6 |
| Create figure export (SVG/PDF) | High | 8 |
| Implement reproduction package generator | High | 10 |
| Add DOI generation for datasets | Low | 4 |
| Create supplementary materials formatter | Medium | 4 |
| Build one-click export | Medium | 4 |
| **Total** | | **44 hours** |

### Total Estimated Effort

| Phase | Hours | Weeks |
|-------|-------|-------|
| Data Foundation | 38 | 1-2 |
| Experiment Framework | 48 | 3-4 |
| Enhanced Interface | 60 | 5-6 |
| Training Integration | 58 | 7-8 |
| Statistical Rigor | 34 | 9-10 |
| Publication Support | 44 | 11-12 |
| **Total** | **282 hours** | **12 weeks** |

---

## Data Schemas

### Complete TypeScript Definitions

```typescript
// schemas/element.ts

export interface Element {
  id: string;
  symbol: string;
  name: string;
  group: GroupId;
  version: string;
  
  description: {
    short: string;
    signature: string;
    full: string;
  };
  
  quantumNumbers: QuantumNumbers;
  theoretical: TheoreticalProperties;
  empirical: EmpiricalProperties;
  training: TrainingSpec;
  validation: ValidationSpec;
  isotopes: Record<string, Isotope> | null;
}

export interface QuantumNumbers {
  direction: 'I' | 'O' | 'T' | 'Τ';
  stance: '+' | '?' | '-';
  scope: 'a' | 'm' | 's' | 'μ';
  transform: 'P' | 'G' | 'R' | 'D';
}

export interface TheoreticalProperties {
  manifoldPosition: [number, number];
  triggers: string[];
  catalyzes: string[];
  interfersWith: string[];
  antipatterns: string[];
}

export interface EmpiricalProperties {
  manifoldPosition: {
    measured: [number, number];
    confidence: number;
    method: string;
    sampleSize: number;
  };
  triggerRates: Record<string, Record<string, TriggerRateResult>>;
  interference: Record<string, InterferenceResult>;
  catalysis: Record<string, CatalysisResult>;
}

export interface TrainingSpec {
  examples: TrainingExample[];
  minExamplesForReliable: number;
}

export interface TrainingExample {
  id: string;
  instruction: string;
  input: string;
  output: string;
  source: 'opus_extraction' | 'human_authored' | 'synthetic';
  dateAdded: string;
  validatedBy: string[];
  isotope?: string;
}

export interface ValidationSpec {
  triggerPrompts: string[];
  expectedPatterns: string[];  // Regex patterns
  negativeExamples: string[];
}

export interface Isotope {
  id: string;
  parentElement: string;
  symbol: string;
  name: string;
  version: string;
  
  description: {
    short: string;
    signature: string;
    full: string;
  };
  
  focus: string;
  triggerPatterns: string[];
  training: TrainingSpec;
  validation: ValidationSpec;
  empirical: Omit<EmpiricalProperties, 'manifoldPosition'>;
}

// schemas/experiment.ts

export interface Hypothesis {
  id: string;
  title: string;
  statement: string;
  falsifiable: boolean;
  falsificationCriteria: string;
  status: 'pending' | 'testing' | 'supported' | 'refuted' | 'inconclusive';
  createdAt: string;
  experiments: string[];
  outcome: {
    status: string;
    evidence: string;
    decidedAt: string;
  } | null;
}

export interface ExperimentRun {
  id: string;
  hypothesis: string;
  title: string;
  description: string;
  
  compound: CompoundSpec;
  architecture: string;
  trainingConfig: TrainingConfig;
  
  predictions: Predictions;
  validation: ValidationConfig;
  
  status: 'planned' | 'queued' | 'training' | 'validating' | 'complete' | 'failed';
  progress?: {
    currentStep: string;
    percentComplete: number;
    currentMetrics?: Record<string, number>;
  };
  
  results?: ExperimentResults;
  artifacts?: ExperimentArtifacts;
  
  createdAt: string;
  startedAt?: string;
  completedAt?: string;
}

export interface CompoundSpec {
  elements: string[];
  isotopes: Record<string, string[]>;
  sequence: string[];
}

export interface TrainingConfig {
  baseModel: string;
  method: 'lora' | 'full';
  loraRank?: number;
  loraLayers?: number;
  learningRate: number;
  iterations: number;
  batchSize: number;
  seed: number;
}

export interface Predictions {
  triggerRates: Record<string, PredictionWithCI>;
  mmluDelta: PredictionWithCI;
  hallucinationDelta: PredictionWithCI;
}

export interface PredictionWithCI {
  mean: number;
  ci95: [number, number];
  factors?: Record<string, number>;
}

export interface ExperimentResults {
  triggerRates: Record<string, TriggerRateResult>;
  mmlu: BenchmarkResult;
  hallucination: BenchmarkResult;
}

export interface TriggerRateResult {
  rate: number;
  successes: number;
  trials: number;
  ci95: [number, number];
  pValue?: number;
}

export interface BenchmarkResult {
  score: number;
  ci95: [number, number];
  pValue?: number;
  effectSize?: number;
  details?: Record<string, number>;
}

export interface ExperimentArtifacts {
  adapterWeights: string;
  trainingLogs: string;
  benchmarkResults: string;
  trainingData: string;
  validationPrompts: string;
}
```

---

## API Specifications

### REST Endpoints

```yaml
openapi: 3.0.0
info:
  title: Cognitive Elements Research Instrument API
  version: 1.0.0

paths:
  /elements:
    get:
      summary: List all elements
      responses:
        200:
          content:
            application/json:
              schema:
                type: array
                items:
                  $ref: '#/components/schemas/ElementSummary'
  
  /elements/{elementId}:
    get:
      summary: Get element details
      parameters:
        - name: elementId
          in: path
          required: true
          schema:
            type: string
      responses:
        200:
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/Element'
  
  /elements/{elementId}/isotopes:
    get:
      summary: List isotopes for element
    post:
      summary: Create new isotope
  
  /predictions:
    post:
      summary: Get predictions for compound
      requestBody:
        content:
          application/json:
            schema:
              type: object
              properties:
                compound:
                  $ref: '#/components/schemas/CompoundSpec'
                architecture:
                  type: string
      responses:
        200:
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/Predictions'
  
  /experiments:
    get:
      summary: List experiments
    post:
      summary: Submit new experiment
  
  /experiments/{expId}:
    get:
      summary: Get experiment details
    delete:
      summary: Cancel experiment
  
  /hypotheses:
    get:
      summary: List hypotheses
    post:
      summary: Register new hypothesis
  
  /export/latex:
    post:
      summary: Generate LaTeX tables
  
  /export/reproduction:
    post:
      summary: Generate reproduction package

components:
  schemas:
    Element:
      # ... full schema
    CompoundSpec:
      # ... full schema
    Predictions:
      # ... full schema
```

### WebSocket Events

```typescript
// Client → Server
interface ClientEvents {
  'experiment:subscribe': { experimentId: string };
  'experiment:unsubscribe': { experimentId: string };
  'training:pause': { experimentId: string };
  'training:resume': { experimentId: string };
  'training:cancel': { experimentId: string };
}

// Server → Client
interface ServerEvents {
  'experiment:progress': {
    experimentId: string;
    step: string;
    percentComplete: number;
    metrics: {
      trainLoss?: number;
      valLoss?: number;
      iteration?: number;
    };
  };
  'experiment:status': {
    experimentId: string;
    status: ExperimentStatus;
  };
  'experiment:complete': {
    experimentId: string;
    results: ExperimentResults;
  };
  'experiment:error': {
    experimentId: string;
    error: string;
  };
}
```

---

## Immediate Next Steps

### For Claude Code Session

1. **Clone/Create Repository**
   ```bash
   mkdir cognitive-elements-research
   cd cognitive-elements-research
   git init
   ```

2. **Create Directory Structure**
   ```bash
   mkdir -p frontend/src/{components,hooks,stores,api,utils}
   mkdir -p backend/src/{server,training,analysis,export,data}
   mkdir -p data/{elements,compounds,experiments,benchmarks}
   mkdir -p scripts tests docs
   ```

3. **Initialize Frontend**
   ```bash
   cd frontend
   npm create vite@latest . -- --template react-ts
   npm install zustand @tanstack/react-query d3 recharts
   ```

4. **Initialize Backend**
   ```bash
   cd backend
   python -m venv venv
   source venv/bin/activate
   pip install fastapi uvicorn pydantic scipy numpy mlx
   ```

5. **Create Initial Data Files**
   - Migrate V10.1 element definitions
   - Create manifest.json
   - Set up validation prompts

6. **Build First Component**
   - Start with enhanced Periodic Table view
   - Add empirical data overlay
   - Connect to data files

### Prioritized MVP Features

For a minimum viable research instrument:

1. ✅ Data foundation with empirical tracking
2. ✅ Isotope support (at least for SKEPTIC)
3. ✅ Basic experiment tracking
4. ✅ Statistical analysis for trigger rates
5. ✅ One working training pipeline (local MLX)
6. ✅ Basic LaTeX export

This gets you to "peer-review capable" in approximately 4-6 weeks.

---

## Target Venues

| Venue | Deadline | Requirements | Fit |
|-------|----------|--------------|-----|
| NeurIPS 2026 | May 2026 | Novel methodology, strong empirics | High |
| ICML 2026 | Feb 2026 | Technical contribution | Medium |
| EMNLP 2026 | June 2026 | NLP focus | High |
| COLM 2026 | TBD | Language models | Very High |
| Nature Machine Intelligence | Rolling | Broad impact | Medium |
| TMLR | Rolling | Rigorous, any topic | High |

**Recommended:** Target COLM or TMLR for initial publication, then expand to Nature MI with additional validation.

---

## Appendix: Quick Reference

### Element Quantum Numbers

| Code | Direction | Stance | Scope | Transform |
|------|-----------|--------|-------|-----------|
| I | Inward | + Assertive | a Atomic | P Preservative |
| O | Outward | ? Interrogative | m Molecular | G Generative |
| T | Transverse | - Receptive | s Systemic | R Reductive |
| Τ | Temporal | | μ Meta | D Destructive |

### Statistical Tests Reference

| Test | Use Case | Implementation |
|------|----------|----------------|
| Wilson Score | CI for proportions | `scipy.stats` |
| McNemar's | Paired binary comparison | Custom |
| Cohen's d | Effect size | Custom |
| Bonferroni | Multiple comparison | `statsmodels` |
| I² | Heterogeneity | Custom |

### Key Files to Create First

1. `data/elements/manifest.json`
2. `data/elements/soliton/definition.json`
3. `data/elements/skeptic/definition.json` (with isotopes)
4. `backend/src/data/schemas.py`
5. `frontend/src/components/PeriodicTable/index.tsx`

---

*Document generated: January 19, 2026*
*Cultural Soliton Observatory*
*Kevin Russell & Claude Opus 4.5*
