# TCE Data Foundation

**Phase 1 Implementation of the Cognitive Elements Research Instrument**

## Overview

This directory contains the data foundation for the Periodic Table of Cognitive Elements research instrument. The foundation provides:

1. **JSON Schemas** - Formal definitions for elements, experiments, and results
2. **TypeScript Types** - Type-safe interfaces for application development
3. **Validated Data** - Migrated element data in the new schema format
4. **Validation Tools** - Utilities for data integrity checking

## Directory Structure

```
TCE/
├── schemas/                          # JSON Schema definitions
│   ├── element-schema.json          # Cognitive element schema
│   ├── experiment-schema.json       # Experiment definition schema
│   └── results-schema.json          # Experiment results schema
├── types/                            # TypeScript definitions
│   └── cognitive-elements.ts        # Complete type definitions
├── data/                             # Validated data files
│   ├── elements.json                # All cognitive elements
│   └── experiments/                 # Experiment definitions
│       └── exp_isotope_coverage_v10_2a.json
├── validate.py                       # Validation utility
├── cognitive-elements.jsx           # React visualization (existing)
├── PERIODIC_TABLE_OF_COGNITIVE_ELEMENTS.md  # Theory document
├── COGNITIVE_ELEMENTS_RESEARCH_INSTRUMENT_PLAN.md  # Full plan
└── DATA_FOUNDATION.md               # This file
```

## Schema Specifications

### Element Schema

Each cognitive element has:

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique lowercase identifier (e.g., "skeptic") |
| `symbol` | string | Greek letter symbol (e.g., "Σ") |
| `name` | string | Uppercase name (e.g., "SKEPTIC") |
| `group` | enum | One of 8 groups (epistemic, analytical, etc.) |
| `quantum_numbers` | object | The four cognitive "quantum numbers" |
| `description` | string | Core description with signature phrase |
| `manifold_position` | object | Position in coordination space |
| `triggers` | array | Conditions that activate the element |
| `examples` | array | Output examples demonstrating the element |
| `isotopes` | array | Variant forms (e.g., SKEPTIC isotopes) |
| `catalyzes` | array | Elements this one enhances |
| `antipatterns` | array | Failure modes |
| `training_status` | object | Training state and metrics |

### Quantum Numbers

The four axes that define cognitive space:

```
Direction (D): I=Inward, O=Outward, T=Transverse, Τ=Temporal
Stance (E): +=Assertive, ?=Interrogative, -=Receptive
Scope (S): a=Atomic, m=Molecular, s=Systemic, μ=Meta
Transform (T): P=Preservative, G=Generative, R=Reductive, D=Destructive
```

### Experiment Schema

Experiments follow the hypothesis-prediction-validation workflow:

```json
{
  "id": "exp_example",
  "name": "Human-readable name",
  "hypothesis": {
    "statement": "The claim being tested",
    "prediction": "Specific measurable prediction",
    "falsifiable": true,
    "effect_size_expected": 0.8
  },
  "methodology": {
    "type": "trigger_rate|interference|compound_stability|...",
    "sample_size": { "planned": 20 },
    "prompts": [...],
    "metrics": [...]
  },
  "status": "draft|registered|running|completed|failed|abandoned"
}
```

### Results Schema

Results capture:

- Individual trials with responses and metrics
- Statistical summaries (Wilson score CIs, effect sizes)
- Hypothesis outcomes with confidence levels
- Reproducibility information

## Current Statistics

```
Total Elements: 17
  Trained: 12 (71%)
  Untrained: 5 (29%)
  Total Isotopes: 4 (all SKEPTIC)
  Average Trigger Rate: 96.6%

By Training Version:
  V10.1    9 elements
  V10.2a   1 element (SKEPTIC with isotopes)
  V11      1 element (INTEGRATOR - AI-designed)
  V12      1 element (GOVERNOR - AI-designed)
```

## Validation

Run validation with:

```bash
cd TCE
python3 validate.py all      # Validate all files
python3 validate.py stats    # Show statistics
python3 validate.py elements # Validate elements only
```

## TypeScript Usage

```typescript
import type {
  CognitiveElement,
  Experiment,
  ExperimentResults,
  QuantumNumbers
} from './types/cognitive-elements';

// Type-safe element access
const element: CognitiveElement = {
  id: "skeptic",
  symbol: "Σ",
  name: "SKEPTIC",
  group: "evaluative",
  quantum_numbers: {
    direction: "O",
    stance: "?",
    scope: "a",
    transform: "D"
  },
  description: "\"Flag a problem...\" - Premise checking",
  version: "1.0.0"
};
```

## Next Steps (Phase 2)

The next phase implements the Experiment Framework:

1. Experiment designer UI component
2. Automatic preregistration with hashing
3. Prompt management system
4. Results collection pipeline

See `COGNITIVE_ELEMENTS_RESEARCH_INSTRUMENT_PLAN.md` for the full roadmap.

---

*Cultural Soliton Observatory*
*Phase 1: Data Foundation*
*January 19, 2026*
