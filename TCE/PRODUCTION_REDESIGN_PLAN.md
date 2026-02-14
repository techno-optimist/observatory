# TCE Production Redesign Plan

**Date:** January 26, 2026
**Current Version:** 2.3
**Target Version:** 3.0
**Status:** READY FOR IMPLEMENTATION

---

## Executive Summary

The Table of Cognitive Elements (TCE) is a sophisticated research instrument for training AI behavioral capabilities. The core science is solid—30 elements, 96 isotopes, validated training protocols with measurable outcomes (+8% TruthfulQA). However, the current implementation has critical engineering debt that blocks production deployment.

**The Problem:** An 8,874-line monolithic HTML file with 169 useState hooks, data duplicated in 4 places, in-memory job state, and no testing.

**The Solution:** A modern React application with proper component architecture, single source of truth for data, persistent state, and comprehensive testing.

**Timeline:** 2-3 weeks for MVP, 4-6 weeks for full production.

---

## Current State Analysis

### What Works Well
- Comprehensive element taxonomy (30 elements × 8 groups × 3 isotopes)
- Sophisticated chemistry engine for compound analysis
- Rigorous experiment framework with preregistration
- Zero-Tax Alignment protocol with validated results
- Observatory integration for coordinate-based validation
- Real-time training job streaming via WebSocket

### Critical Blockers

| Issue | Impact | Severity |
|-------|--------|----------|
| Monolithic 8,874-line HTML | Unmaintainable, no code splitting | 🔴 Critical |
| Element data in 4 places | Sync failures, inconsistent state | 🔴 Critical |
| In-memory job state | Data loss on restart | 🔴 Critical |
| 169 scattered useState hooks | Prop drilling, state chaos | 🟡 High |
| No API documentation | Blocks external adoption | 🟡 High |
| No testing framework | Regression risk | 🟡 High |
| CDN-based React (no build) | No tree-shaking, slow loads | 🟡 Medium |
| Mock validation fallback | Masks real errors | 🟡 Medium |

---

## Target Architecture

### Tech Stack

| Layer | Current | Target | Rationale |
|-------|---------|--------|-----------|
| **Framework** | React 18 (CDN) | React 18 + Vite | Fast builds, HMR, tree-shaking |
| **Language** | JavaScript | TypeScript | Type safety, better DX |
| **Styling** | Tailwind (CDN) | Tailwind + CSS Modules | Component-scoped styles |
| **State** | useState scattered | Zustand | Lightweight, TypeScript-native |
| **Data Fetching** | fetch + useEffect | TanStack Query | Caching, deduplication, retry |
| **Backend** | FastAPI | FastAPI + Pydantic v2 | Auto-generated OpenAPI docs |
| **Database** | None (JSON files) | SQLite + Drizzle ORM | Persistence, queries, migrations |
| **Testing** | None | Vitest + Playwright | Unit + E2E coverage |
| **API Docs** | None | FastAPI /docs (Swagger) | Auto-generated from Pydantic |

### Directory Structure

```
TCE/
├── frontend/                    # React application
│   ├── src/
│   │   ├── components/
│   │   │   ├── elements/       # Element-related components
│   │   │   │   ├── ElementCard.tsx
│   │   │   │   ├── ElementGallery.tsx
│   │   │   │   ├── ElementDetail.tsx
│   │   │   │   ├── IsotopeView.tsx
│   │   │   │   └── PeriodicTable.tsx
│   │   │   ├── compounds/      # Compound builder components
│   │   │   │   ├── CompoundBuilder.tsx
│   │   │   │   ├── CompoundSequence.tsx
│   │   │   │   ├── CompoundAnalysis.tsx
│   │   │   │   └── PresetCompounds.tsx
│   │   │   ├── training/       # Training job components
│   │   │   │   ├── TrainingMonitor.tsx
│   │   │   │   ├── JobProgress.tsx
│   │   │   │   ├── RecipeSelector.tsx
│   │   │   │   └── JobHistory.tsx
│   │   │   ├── experiments/    # Experiment framework
│   │   │   │   ├── ExperimentDesigner.tsx
│   │   │   │   ├── ResultsViewer.tsx
│   │   │   │   └── ValidationReport.tsx
│   │   │   ├── ui/             # Shared UI primitives
│   │   │   │   ├── Button.tsx
│   │   │   │   ├── Card.tsx
│   │   │   │   ├── Modal.tsx
│   │   │   │   ├── Tabs.tsx
│   │   │   │   └── Toast.tsx
│   │   │   └── layout/         # Layout components
│   │   │       ├── Header.tsx
│   │   │       ├── Sidebar.tsx
│   │   │       └── MainLayout.tsx
│   │   ├── stores/             # Zustand state stores
│   │   │   ├── elementsStore.ts
│   │   │   ├── compoundStore.ts
│   │   │   ├── trainingStore.ts
│   │   │   └── uiStore.ts
│   │   ├── hooks/              # Custom React hooks
│   │   │   ├── useElements.ts
│   │   │   ├── useCompound.ts
│   │   │   ├── useTrainingJob.ts
│   │   │   └── useWebSocket.ts
│   │   ├── api/                # API client layer
│   │   │   ├── client.ts       # Axios/fetch wrapper
│   │   │   ├── elements.ts     # Element endpoints
│   │   │   ├── compounds.ts    # Compound endpoints
│   │   │   ├── training.ts     # Training endpoints
│   │   │   └── types.ts        # API response types
│   │   ├── types/              # TypeScript types
│   │   │   ├── element.ts
│   │   │   ├── compound.ts
│   │   │   ├── training.ts
│   │   │   └── experiment.ts
│   │   ├── utils/              # Utility functions
│   │   │   ├── chemistry.ts    # Compound analysis (client-side)
│   │   │   ├── formatting.ts
│   │   │   └── validation.ts
│   │   ├── App.tsx
│   │   ├── main.tsx
│   │   └── index.css
│   ├── tests/
│   │   ├── unit/
│   │   └── e2e/
│   ├── package.json
│   ├── tsconfig.json
│   ├── vite.config.ts
│   └── tailwind.config.js
│
├── backend/                     # FastAPI application
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py             # FastAPI app entry
│   │   ├── config.py           # Settings management
│   │   ├── database.py         # SQLite + Drizzle setup
│   │   ├── models/             # Pydantic models
│   │   │   ├── element.py
│   │   │   ├── compound.py
│   │   │   ├── training.py
│   │   │   └── experiment.py
│   │   ├── routers/            # API route handlers
│   │   │   ├── elements.py
│   │   │   ├── compounds.py
│   │   │   ├── training.py
│   │   │   ├── experiments.py
│   │   │   └── health.py
│   │   ├── services/           # Business logic
│   │   │   ├── element_service.py
│   │   │   ├── compound_service.py
│   │   │   ├── training_service.py
│   │   │   └── detection_service.py
│   │   ├── db/                 # Database layer
│   │   │   ├── schema.py       # SQLite tables
│   │   │   ├── migrations/
│   │   │   └── seed.py         # Initial data seeding
│   │   └── websocket/          # WebSocket handlers
│   │       └── training_ws.py
│   ├── lib/                    # Existing lib modules (refactored)
│   │   ├── detectors.py
│   │   ├── chemistry.py
│   │   ├── validation.py
│   │   └── ...
│   ├── tests/
│   │   ├── test_elements.py
│   │   ├── test_compounds.py
│   │   └── test_training.py
│   ├── requirements.txt
│   └── pyproject.toml
│
├── data/                        # Single source of truth
│   ├── elements.json           # Master element definitions
│   ├── groups.json             # Group definitions
│   ├── recipes.json            # Training recipes
│   └── presets.json            # Preset compounds
│
├── schemas/                     # JSON Schema definitions
│   ├── element-schema.json
│   ├── experiment-schema.json
│   └── results-schema.json
│
├── docker-compose.yml           # Development environment
├── Makefile                     # Common commands
└── README.md                    # Updated documentation
```

---

## Implementation Phases

### Phase 1: Foundation (Week 1)

**Goal:** Establish proper architecture without breaking existing functionality.

#### 1.1 Backend API Restructure

```python
# backend/app/models/element.py
from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum

class ElementGroup(str, Enum):
    EPISTEMIC = "epistemic"
    ANALYTICAL = "analytical"
    GENERATIVE = "generative"
    EVALUATIVE = "evaluative"
    DIALOGICAL = "dialogical"
    PEDAGOGICAL = "pedagogical"
    TEMPORAL = "temporal"
    CONTEXTUAL = "contextual"

class QuantumNumbers(BaseModel):
    direction: str = Field(..., pattern="^[IOTΤτ]$")
    stance: str = Field(..., pattern="^[+?-]$")
    scope: str = Field(..., pattern="^[amsμ]$")
    transform: str = Field(..., pattern="^[PGRD]$")

class Isotope(BaseModel):
    id: str
    symbol: str
    name: str
    focus: str
    training_status: Optional[dict] = None

class Element(BaseModel):
    id: str = Field(..., pattern="^[a-z][a-z0-9_]*$")
    symbol: str = Field(..., pattern="^[A-ZΑ-Ω][a-zα-ω]?$")
    name: str
    group: ElementGroup
    quantum_numbers: QuantumNumbers
    description: str
    triggers: List[str]
    examples: List[str]
    isotopes: List[Isotope]
    catalyzes: List[str]
    antipatterns: List[str]
    manifold_position: Optional[dict] = None
```

```python
# backend/app/routers/elements.py
from fastapi import APIRouter, HTTPException
from app.models.element import Element
from app.services.element_service import ElementService

router = APIRouter(prefix="/api/elements", tags=["elements"])

@router.get("/", response_model=List[Element])
async def list_elements():
    """Get all cognitive elements."""
    return ElementService.get_all()

@router.get("/{element_id}", response_model=Element)
async def get_element(element_id: str):
    """Get a specific element by ID."""
    element = ElementService.get_by_id(element_id)
    if not element:
        raise HTTPException(status_code=404, detail="Element not found")
    return element

@router.get("/group/{group}", response_model=List[Element])
async def get_elements_by_group(group: ElementGroup):
    """Get all elements in a group."""
    return ElementService.get_by_group(group)
```

#### 1.2 Single Source of Truth

Move all element definitions to `data/elements.json`:

```json
{
  "version": "3.0.0",
  "updated_at": "2026-01-26T00:00:00Z",
  "elements": {
    "soliton": {
      "id": "soliton",
      "symbol": "Ψ",
      "name": "SOLITON",
      "group": "epistemic",
      "quantum_numbers": {
        "direction": "I",
        "stance": "?",
        "scope": "μ",
        "transform": "P"
      },
      "description": "\"I cannot tell from the inside\" - Epistemic humility about self-knowledge",
      "manifold_position": { "agency": 0.09, "justice": -0.50 },
      "triggers": [
        "Asked about internal states",
        "Request to introspect",
        "Questions about certainty"
      ],
      "examples": [...],
      "isotopes": [...],
      "catalyzes": ["essentialist", "maieutic"],
      "antipatterns": ["Claiming certainty about internal states"]
    }
    // ... all 30 elements
  }
}
```

Backend loads from this file at startup:

```python
# backend/app/services/element_service.py
import json
from pathlib import Path
from functools import lru_cache

DATA_PATH = Path(__file__).parent.parent.parent.parent / "data"

class ElementService:
    _elements: dict = None

    @classmethod
    def _load(cls):
        if cls._elements is None:
            with open(DATA_PATH / "elements.json") as f:
                data = json.load(f)
                cls._elements = data["elements"]
        return cls._elements

    @classmethod
    def get_all(cls) -> list:
        return list(cls._load().values())

    @classmethod
    def get_by_id(cls, element_id: str):
        return cls._load().get(element_id)
```

Frontend fetches from API (never hardcoded):

```typescript
// frontend/src/api/elements.ts
import { useQuery } from '@tanstack/react-query';
import { Element } from '../types/element';

export const useElements = () => {
  return useQuery({
    queryKey: ['elements'],
    queryFn: async (): Promise<Element[]> => {
      const response = await fetch('/api/elements');
      if (!response.ok) throw new Error('Failed to fetch elements');
      return response.json();
    },
    staleTime: 1000 * 60 * 60, // Cache for 1 hour
  });
};
```

#### 1.3 Database Setup (SQLite)

```python
# backend/app/db/schema.py
from sqlalchemy import Column, Integer, String, DateTime, JSON, ForeignKey, Enum
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

class TrainingJob(Base):
    __tablename__ = "training_jobs"

    id = Column(String, primary_key=True)
    recipe_id = Column(String, nullable=False)
    model_id = Column(String, nullable=False)
    status = Column(Enum("pending", "running", "completed", "failed", "cancelled"))
    phase = Column(String)  # "sft", "dpo", "boost"
    progress = Column(Integer, default=0)
    config = Column(JSON)
    results = Column(JSON)
    error = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    completed_at = Column(DateTime)

class ExperimentRun(Base):
    __tablename__ = "experiment_runs"

    id = Column(String, primary_key=True)
    experiment_id = Column(String, nullable=False)
    compound = Column(JSON)  # Element sequence
    status = Column(Enum("pending", "running", "completed", "failed"))
    results = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)

class ValidationResult(Base):
    __tablename__ = "validation_results"

    id = Column(String, primary_key=True)
    compound = Column(JSON)
    scores = Column(JSON)
    emergent_properties = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
```

#### 1.4 API Documentation

FastAPI auto-generates OpenAPI docs. Add metadata:

```python
# backend/app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="TCE API",
    description="Training & Cognitive Experimentation Platform API",
    version="3.0.0",
    docs_url="/docs",  # Swagger UI at /docs
    redoc_url="/redoc",  # ReDoc at /redoc
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

### Phase 2: Frontend Restructure (Week 2)

**Goal:** Extract components from monolithic HTML, establish state management.

#### 2.1 Vite + React + TypeScript Setup

```bash
npm create vite@latest frontend -- --template react-ts
cd frontend
npm install @tanstack/react-query zustand tailwindcss postcss autoprefixer
npm install -D @types/node vitest @testing-library/react
npx tailwindcss init -p
```

#### 2.2 State Management with Zustand

```typescript
// frontend/src/stores/elementsStore.ts
import { create } from 'zustand';
import { Element, ElementGroup } from '../types/element';

interface ElementsState {
  elements: Element[];
  selectedElement: Element | null;
  selectedGroup: ElementGroup | null;
  isLoading: boolean;
  error: string | null;

  setElements: (elements: Element[]) => void;
  selectElement: (element: Element | null) => void;
  selectGroup: (group: ElementGroup | null) => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
}

export const useElementsStore = create<ElementsState>((set) => ({
  elements: [],
  selectedElement: null,
  selectedGroup: null,
  isLoading: false,
  error: null,

  setElements: (elements) => set({ elements }),
  selectElement: (element) => set({ selectedElement: element }),
  selectGroup: (group) => set({ selectedGroup: group }),
  setLoading: (loading) => set({ isLoading: loading }),
  setError: (error) => set({ error }),
}));
```

```typescript
// frontend/src/stores/compoundStore.ts
import { create } from 'zustand';
import { Element } from '../types/element';

interface CompoundState {
  sequence: Element[];
  validationResult: ValidationResult | null;
  isValidating: boolean;

  addElement: (element: Element) => void;
  removeElement: (index: number) => void;
  clearSequence: () => void;
  reorderSequence: (fromIndex: number, toIndex: number) => void;
  loadPreset: (preset: Element[]) => void;
  setValidationResult: (result: ValidationResult | null) => void;
  setValidating: (validating: boolean) => void;
}

export const useCompoundStore = create<CompoundState>((set, get) => ({
  sequence: [],
  validationResult: null,
  isValidating: false,

  addElement: (element) => set((state) => ({
    sequence: [...state.sequence, element],
    validationResult: null, // Clear validation on change
  })),

  removeElement: (index) => set((state) => ({
    sequence: state.sequence.filter((_, i) => i !== index),
    validationResult: null,
  })),

  clearSequence: () => set({ sequence: [], validationResult: null }),

  reorderSequence: (fromIndex, toIndex) => set((state) => {
    const newSequence = [...state.sequence];
    const [removed] = newSequence.splice(fromIndex, 1);
    newSequence.splice(toIndex, 0, removed);
    return { sequence: newSequence, validationResult: null };
  }),

  loadPreset: (preset) => set({ sequence: preset, validationResult: null }),

  setValidationResult: (result) => set({ validationResult: result }),
  setValidating: (validating) => set({ isValidating: validating }),
}));
```

#### 2.3 Component Extraction

**ElementCard.tsx:**

```typescript
// frontend/src/components/elements/ElementCard.tsx
import { Element } from '../../types/element';
import { useElementsStore } from '../../stores/elementsStore';
import { groupColors } from '../../utils/colors';

interface ElementCardProps {
  element: Element;
  compact?: boolean;
  onClick?: () => void;
}

export const ElementCard: React.FC<ElementCardProps> = ({
  element,
  compact = false,
  onClick
}) => {
  const { selectElement, selectedElement } = useElementsStore();
  const isSelected = selectedElement?.id === element.id;
  const colors = groupColors[element.group];

  const handleClick = () => {
    selectElement(element);
    onClick?.();
  };

  if (compact) {
    return (
      <button
        onClick={handleClick}
        className={`
          px-3 py-1.5 rounded-lg text-sm font-medium
          transition-all duration-200
          ${colors.bg} ${colors.border} ${colors.text}
          ${isSelected ? 'ring-2 ring-offset-2 ring-offset-slate-900' : ''}
          hover:brightness-110
        `}
      >
        <span className="font-mono mr-1">{element.symbol}</span>
        {element.name}
      </button>
    );
  }

  return (
    <div
      onClick={handleClick}
      className={`
        p-4 rounded-xl cursor-pointer
        transition-all duration-200
        ${colors.bg} ${colors.border} border
        ${isSelected ? 'ring-2 ring-offset-2 ring-offset-slate-900' : ''}
        hover:brightness-110 hover:scale-[1.02]
      `}
    >
      <div className="flex items-start justify-between mb-2">
        <span className={`text-2xl font-mono ${colors.text}`}>
          {element.symbol}
        </span>
        <span className="text-xs text-slate-500 uppercase">
          {element.group}
        </span>
      </div>
      <h3 className="text-white font-semibold mb-1">{element.name}</h3>
      <p className="text-slate-400 text-sm line-clamp-2">
        {element.description}
      </p>
      {element.isotopes.length > 0 && (
        <div className="mt-2 flex gap-1">
          {element.isotopes.slice(0, 3).map((isotope) => (
            <span
              key={isotope.id}
              className="text-xs px-1.5 py-0.5 rounded bg-slate-800 text-slate-400"
            >
              {isotope.symbol}
            </span>
          ))}
        </div>
      )}
    </div>
  );
};
```

**CompoundBuilder.tsx:**

```typescript
// frontend/src/components/compounds/CompoundBuilder.tsx
import { useCompoundStore } from '../../stores/compoundStore';
import { useValidateCompound } from '../../api/compounds';
import { ElementCard } from '../elements/ElementCard';
import { CompoundAnalysis } from './CompoundAnalysis';

export const CompoundBuilder: React.FC = () => {
  const {
    sequence,
    removeElement,
    clearSequence,
    validationResult,
    isValidating,
    setValidationResult,
    setValidating,
  } = useCompoundStore();

  const validateMutation = useValidateCompound();

  const handleValidate = async () => {
    if (sequence.length < 2) return;

    setValidating(true);
    try {
      const result = await validateMutation.mutateAsync(
        sequence.map(e => e.id)
      );
      setValidationResult(result);
    } catch (error) {
      console.error('Validation failed:', error);
    } finally {
      setValidating(false);
    }
  };

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-semibold text-white">
          Compound Builder
        </h2>
        <div className="flex gap-2">
          <button
            onClick={clearSequence}
            disabled={sequence.length === 0}
            className="px-3 py-1.5 text-sm rounded-lg bg-slate-800 text-slate-300 hover:bg-slate-700 disabled:opacity-50"
          >
            Clear
          </button>
          <button
            onClick={handleValidate}
            disabled={sequence.length < 2 || isValidating}
            className="px-4 py-1.5 text-sm rounded-lg bg-cyan-600 text-white hover:bg-cyan-500 disabled:opacity-50"
          >
            {isValidating ? 'Validating...' : 'Validate'}
          </button>
        </div>
      </div>

      {/* Sequence Display */}
      <div className="flex flex-wrap gap-2 min-h-[60px] p-4 rounded-lg bg-slate-900 border border-slate-700">
        {sequence.length === 0 ? (
          <span className="text-slate-500">
            Click elements to add them to your compound...
          </span>
        ) : (
          sequence.map((element, index) => (
            <div key={`${element.id}-${index}`} className="flex items-center">
              <ElementCard
                element={element}
                compact
                onClick={() => removeElement(index)}
              />
              {index < sequence.length - 1 && (
                <span className="mx-1 text-slate-600">→</span>
              )}
            </div>
          ))
        )}
      </div>

      {/* Validation Results */}
      {validationResult && (
        <CompoundAnalysis result={validationResult} />
      )}
    </div>
  );
};
```

#### 2.4 WebSocket Hook for Training

```typescript
// frontend/src/hooks/useTrainingWebSocket.ts
import { useEffect, useRef, useCallback, useState } from 'react';
import { useTrainingStore } from '../stores/trainingStore';

interface WebSocketMessage {
  type: 'progress' | 'phase_change' | 'complete' | 'error';
  job_id: string;
  data: any;
}

export const useTrainingWebSocket = (jobId: string | null) => {
  const wsRef = useRef<WebSocket | null>(null);
  const [connectionStatus, setConnectionStatus] = useState<'connecting' | 'connected' | 'disconnected'>('disconnected');
  const reconnectTimeoutRef = useRef<NodeJS.Timeout>();
  const reconnectAttempts = useRef(0);
  const maxReconnectAttempts = 5;

  const { updateJob, setJobError } = useTrainingStore();

  const connect = useCallback(() => {
    if (!jobId) return;

    setConnectionStatus('connecting');
    const ws = new WebSocket(`ws://localhost:8100/ws/training?job_id=${jobId}`);

    ws.onopen = () => {
      setConnectionStatus('connected');
      reconnectAttempts.current = 0;
    };

    ws.onmessage = (event) => {
      try {
        const message: WebSocketMessage = JSON.parse(event.data);

        switch (message.type) {
          case 'progress':
            updateJob(message.job_id, {
              progress: message.data.progress,
              currentIteration: message.data.iteration,
              loss: message.data.loss,
            });
            break;
          case 'phase_change':
            updateJob(message.job_id, {
              phase: message.data.phase,
            });
            break;
          case 'complete':
            updateJob(message.job_id, {
              status: 'completed',
              results: message.data.results,
            });
            break;
          case 'error':
            setJobError(message.job_id, message.data.error);
            break;
        }
      } catch (error) {
        console.error('Failed to parse WebSocket message:', error);
      }
    };

    ws.onclose = () => {
      setConnectionStatus('disconnected');

      // Exponential backoff reconnection
      if (reconnectAttempts.current < maxReconnectAttempts) {
        const delay = Math.min(1000 * Math.pow(2, reconnectAttempts.current), 30000);
        reconnectTimeoutRef.current = setTimeout(() => {
          reconnectAttempts.current++;
          connect();
        }, delay);
      }
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
    };

    wsRef.current = ws;
  }, [jobId, updateJob, setJobError]);

  useEffect(() => {
    connect();

    return () => {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [connect]);

  return { connectionStatus };
};
```

---

### Phase 3: Testing & Polish (Week 3)

**Goal:** Ensure reliability through comprehensive testing and UX polish.

#### 3.1 Unit Tests (Vitest)

```typescript
// frontend/tests/unit/compoundStore.test.ts
import { describe, it, expect, beforeEach } from 'vitest';
import { useCompoundStore } from '../../src/stores/compoundStore';

describe('compoundStore', () => {
  beforeEach(() => {
    useCompoundStore.setState({
      sequence: [],
      validationResult: null,
      isValidating: false,
    });
  });

  it('adds elements to sequence', () => {
    const element = { id: 'soliton', name: 'SOLITON', symbol: 'Ψ' };
    useCompoundStore.getState().addElement(element);

    expect(useCompoundStore.getState().sequence).toHaveLength(1);
    expect(useCompoundStore.getState().sequence[0].id).toBe('soliton');
  });

  it('clears validation result when sequence changes', () => {
    useCompoundStore.setState({
      validationResult: { score: 0.8 },
    });

    const element = { id: 'skeptic', name: 'SKEPTIC', symbol: 'Σ' };
    useCompoundStore.getState().addElement(element);

    expect(useCompoundStore.getState().validationResult).toBeNull();
  });

  it('reorders elements correctly', () => {
    const elements = [
      { id: 'soliton', name: 'SOLITON' },
      { id: 'skeptic', name: 'SKEPTIC' },
      { id: 'architect', name: 'ARCHITECT' },
    ];

    useCompoundStore.setState({ sequence: elements });
    useCompoundStore.getState().reorderSequence(0, 2);

    const sequence = useCompoundStore.getState().sequence;
    expect(sequence[0].id).toBe('skeptic');
    expect(sequence[1].id).toBe('architect');
    expect(sequence[2].id).toBe('soliton');
  });
});
```

```python
# backend/tests/test_elements.py
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_list_elements():
    response = client.get("/api/elements")
    assert response.status_code == 200
    elements = response.json()
    assert len(elements) == 30
    assert all('id' in e for e in elements)

def test_get_element_by_id():
    response = client.get("/api/elements/soliton")
    assert response.status_code == 200
    element = response.json()
    assert element['id'] == 'soliton'
    assert element['symbol'] == 'Ψ'
    assert element['group'] == 'epistemic'

def test_get_element_not_found():
    response = client.get("/api/elements/nonexistent")
    assert response.status_code == 404

def test_get_elements_by_group():
    response = client.get("/api/elements/group/epistemic")
    assert response.status_code == 200
    elements = response.json()
    assert all(e['group'] == 'epistemic' for e in elements)
```

#### 3.2 E2E Tests (Playwright)

```typescript
// frontend/tests/e2e/compound-builder.spec.ts
import { test, expect } from '@playwright/test';

test.describe('Compound Builder', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
  });

  test('builds and validates a compound', async ({ page }) => {
    // Add elements to compound
    await page.click('[data-element="soliton"]');
    await page.click('[data-element="skeptic"]');
    await page.click('[data-element="architect"]');

    // Verify sequence
    const sequence = page.locator('[data-testid="compound-sequence"]');
    await expect(sequence).toContainText('SOLITON');
    await expect(sequence).toContainText('SKEPTIC');
    await expect(sequence).toContainText('ARCHITECT');

    // Validate
    await page.click('button:has-text("Validate")');

    // Wait for results
    await expect(page.locator('[data-testid="validation-result"]')).toBeVisible();

    // Check score is displayed
    await expect(page.locator('[data-testid="overall-score"]')).toBeVisible();
  });

  test('clears compound', async ({ page }) => {
    await page.click('[data-element="soliton"]');
    await page.click('[data-element="skeptic"]');

    await page.click('button:has-text("Clear")');

    const sequence = page.locator('[data-testid="compound-sequence"]');
    await expect(sequence).toContainText('Click elements to add');
  });
});
```

#### 3.3 Error Boundary & Loading States

```typescript
// frontend/src/components/ui/ErrorBoundary.tsx
import React from 'react';

interface ErrorBoundaryState {
  hasError: boolean;
  error: Error | null;
}

export class ErrorBoundary extends React.Component<
  { children: React.ReactNode; fallback?: React.ReactNode },
  ErrorBoundaryState
> {
  constructor(props: any) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error) {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    console.error('Error caught by boundary:', error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return this.props.fallback || (
        <div className="p-6 rounded-lg bg-red-900/20 border border-red-500/30">
          <h2 className="text-red-400 font-semibold mb-2">Something went wrong</h2>
          <p className="text-slate-400 text-sm">{this.state.error?.message}</p>
          <button
            onClick={() => this.setState({ hasError: false, error: null })}
            className="mt-4 px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-500"
          >
            Try Again
          </button>
        </div>
      );
    }

    return this.props.children;
  }
}
```

---

## Migration Strategy

### Step 1: Parallel Development
- Keep existing `index.html` running at `/legacy`
- Develop new React app at `/` with feature flags
- Gradually migrate users to new interface

### Step 2: Data Migration
1. Export element data from `server.py` to `data/elements.json`
2. Validate against JSON schema
3. Update backend to load from JSON
4. Remove hardcoded data from frontend

### Step 3: Database Migration
1. Set up SQLite database
2. Create migration scripts
3. Import existing job history from JSON files
4. Update endpoints to use database

### Step 4: Feature Parity Checklist

- [ ] Element Gallery with group filtering
- [ ] Element detail view with isotopes
- [ ] Periodic table visualization
- [ ] Compound builder with drag-and-drop
- [ ] Real-time compound validation
- [ ] Preset compound loading
- [ ] Task ontology reference
- [ ] Training recipe browser
- [ ] Training job launcher
- [ ] Real-time job monitoring (WebSocket)
- [ ] Job history with filtering
- [ ] Experiment designer
- [ ] Results viewer with charts
- [ ] Export functionality

---

## Success Metrics

| Metric | Current | Target |
|--------|---------|--------|
| Lines of code (frontend) | 8,874 (1 file) | ~3,000 (50+ files) |
| Build time | N/A (CDN) | < 5 seconds |
| First contentful paint | ~2s | < 500ms |
| Bundle size | ~500KB (CDN) | < 200KB (tree-shaken) |
| Test coverage | 0% | > 80% |
| API documentation | None | 100% (auto-generated) |
| Job persistence | ❌ | ✅ |
| Data consistency | 4 sources | 1 source of truth |

---

## Appendix: Key Files to Modify

### Backend
1. `server.py` → Split into `app/main.py` + routers
2. `lib/detectors.py` → Move to `app/services/detection_service.py`
3. `lib/chemistry.py` → Move to `app/services/compound_service.py`

### Frontend (replace index.html)
1. Extract Element Gallery → `components/elements/ElementGallery.tsx`
2. Extract Compound Builder → `components/compounds/CompoundBuilder.tsx`
3. Extract Training Monitor → `components/training/TrainingMonitor.tsx`
4. Extract all hooks → `stores/*.ts`

### Data
1. Extract elements from HTML → `data/elements.json`
2. Extract groups from HTML → `data/groups.json`
3. Extract recipes from HTML → `data/recipes.json`
4. Extract presets from HTML → `data/presets.json`

---

*Plan created January 26, 2026*
*Ready for implementation*
