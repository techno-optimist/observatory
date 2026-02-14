/**
 * Cognitive Elements Type Definitions
 *
 * TypeScript types for the Periodic Table of Cognitive Elements
 * research instrument. These types correspond to the JSON schemas
 * in ../schemas/
 *
 * @version 1.0.0
 */

// =============================================================================
// ELEMENT TYPES
// =============================================================================

/** Direction of epistemic action */
export type Direction = 'I' | 'O' | 'T' | 'Τ';
export const DirectionLabels: Record<Direction, string> = {
  'I': 'Inward',
  'O': 'Outward',
  'T': 'Transverse',
  'Τ': 'Temporal'
};

/** Epistemic stance */
export type Stance = '+' | '?' | '-';
export const StanceLabels: Record<Stance, string> = {
  '+': 'Assertive',
  '?': 'Interrogative',
  '-': 'Receptive'
};

/** Scope of application */
export type Scope = 'a' | 'm' | 's' | 'μ';
export const ScopeLabels: Record<Scope, string> = {
  'a': 'Atomic',
  'm': 'Molecular',
  's': 'Systemic',
  'μ': 'Meta'
};

/** Transformational mode */
export type Transform = 'P' | 'G' | 'R' | 'D';
export const TransformLabels: Record<Transform, string> = {
  'P': 'Preservative',
  'G': 'Generative',
  'R': 'Reductive',
  'D': 'Destructive'
};

/** Element groups */
export type ElementGroup =
  | 'epistemic'
  | 'analytical'
  | 'generative'
  | 'evaluative'
  | 'dialogical'
  | 'pedagogical'
  | 'temporal'
  | 'contextual';

/** The four quantum numbers of cognition */
export interface QuantumNumbers {
  direction: Direction;
  stance: Stance;
  scope: Scope;
  transform: Transform;
}

/** Manifold position in coordination space */
export interface ManifoldPosition {
  agency: number;      // -1 to 1
  justice: number;     // -1 to 1
  belonging?: number;  // -1 to 1 (optional third dimension)
  confidence?: number; // 0 to 1
  measured_at?: string; // ISO datetime
}

/** Trigger condition for element activation */
export interface Trigger {
  pattern: string;
  regex?: string;
  confidence: number;
}

/** Example output demonstrating the element */
export interface Example {
  text: string;
  source: 'synthetic' | 'observed' | 'validated';
  context?: string;
}

/** Isotope variant of an element */
export interface Isotope {
  id: string;
  symbol: string;
  name: string;
  focus: string;
  training_status?: {
    trained: boolean;
    example_count: number;
    trigger_rate?: number;
    version?: string;
  };
}

/** Antipattern / failure mode */
export interface Antipattern {
  pattern: string;
  description: string;
  detection?: string;
}

/** Training status for an element */
export interface TrainingStatus {
  trained: boolean;
  version?: string;
  example_count?: number;
  trigger_rate?: number;
  adapter_path?: string;
  trained_at?: string;
}

/** Complete cognitive element definition */
export interface CognitiveElement {
  id: string;
  symbol: string;
  name: string;
  group: ElementGroup;
  quantum_numbers: QuantumNumbers;
  description: string;
  manifold_position?: ManifoldPosition;
  triggers?: Trigger[];
  examples?: Example[];
  isotopes?: Isotope[];
  catalyzes?: string[];
  antipatterns?: Antipattern[];
  training_status?: TrainingStatus;
  version: string;
  created_at?: string;
  updated_at?: string;
  notes?: string;
}

// =============================================================================
// EXPERIMENT TYPES
// =============================================================================

/** Experiment methodology types */
export type ExperimentType =
  | 'trigger_rate'
  | 'interference'
  | 'compound_stability'
  | 'transfer'
  | 'manifold_position'
  | 'isotope_differentiation'
  | 'behavioral';

/** Experiment status */
export type ExperimentStatus =
  | 'draft'
  | 'registered'
  | 'running'
  | 'completed'
  | 'failed'
  | 'abandoned';

/** Prompt difficulty levels */
export type PromptDifficulty = 'easy' | 'medium' | 'hard' | 'adversarial';

/** Metric types */
export type MetricType = 'binary' | 'continuous' | 'categorical' | 'manifold_distance';
export type MetricAggregation = 'mean' | 'median' | 'wilson_score' | 'bootstrap_ci';

/** Hypothesis definition */
export interface Hypothesis {
  statement: string;
  prediction: string;
  null_hypothesis?: string;
  falsifiable: boolean;
  effect_size_expected?: number;
}

/** Test prompt */
export interface TestPrompt {
  id: string;
  text: string;
  expected_element: string;
  expected_isotope?: string;
  difficulty?: PromptDifficulty;
}

/** Metric definition */
export interface MetricDefinition {
  name: string;
  type: MetricType;
  threshold?: number;
  aggregation?: MetricAggregation;
}

/** Power analysis parameters */
export interface PowerAnalysis {
  alpha: number;
  power: number;
  effect_size: number;
  computed_n: number;
}

/** Methodology specification */
export interface Methodology {
  type: ExperimentType;
  sample_size: {
    planned: number;
    power_analysis?: PowerAnalysis;
  };
  prompts?: TestPrompt[];
  metrics?: MetricDefinition[];
  controls?: {
    baseline_model?: string;
    random_seed?: number;
    temperature?: number;
    max_tokens?: number;
  };
}

/** Preregistration record */
export interface Preregistration {
  registered_at: string;
  hash: string;
  modifications?: Array<{
    timestamp: string;
    field: string;
    reason: string;
  }>;
}

/** Complete experiment definition */
export interface Experiment {
  id: string;
  name: string;
  hypothesis: Hypothesis;
  elements: {
    target?: string[];
    control?: string[];
    compound?: string[];
  };
  methodology: Methodology;
  status: ExperimentStatus;
  preregistration?: Preregistration;
  results_id?: string;
  version: string;
  created_at?: string;
  updated_at?: string;
  author?: string;
  notes?: string;
}

// =============================================================================
// RESULTS TYPES
// =============================================================================

/** Results status */
export type ResultsStatus = 'partial' | 'complete' | 'validated';

/** Effect size interpretation */
export type EffectSizeInterpretation = 'negligible' | 'small' | 'medium' | 'large';

/** Hypothesis outcome confidence */
export type OutcomeConfidence = 'strong' | 'moderate' | 'weak' | 'inconclusive';

/** Detected element in response */
export interface DetectedElement {
  element_id: string;
  confidence: number;
  markers_found?: string[];
}

/** Trial response */
export interface TrialResponse {
  text: string;
  model?: string;
  adapter?: string;
  tokens?: number;
  latency_ms?: number;
}

/** Trial metrics */
export interface TrialMetrics {
  triggered?: boolean;
  trigger_confidence?: number;
  detected_elements?: DetectedElement[];
  manifold_position?: ManifoldPosition;
  custom?: Record<string, unknown>;
}

/** Human annotation */
export interface Annotation {
  human_label?: string;
  annotator?: string;
  confidence?: number;
  notes?: string;
}

/** Individual trial result */
export interface Trial {
  trial_id: string;
  prompt_id: string;
  prompt_text?: string;
  response: TrialResponse;
  metrics: TrialMetrics;
  annotations?: Annotation;
  timestamp: string;
}

/** Wilson score confidence interval */
export interface WilsonScore {
  point_estimate: number;
  wilson_lower: number;
  wilson_upper: number;
  confidence_level: number;
}

/** Effect size computation */
export interface EffectSize {
  cohens_d: number;
  interpretation: EffectSizeInterpretation;
}

/** Statistical test result */
export interface StatisticalTest {
  test_name: string;
  statistic: number;
  p_value: number;
  significant: boolean;
  interpretation: string;
}

/** Per-element summary */
export interface ElementSummary {
  n: number;
  trigger_rate: number;
  mean_confidence?: number;
}

/** Results summary */
export interface ResultsSummary {
  n_trials: number;
  n_successful: number;
  trigger_rate?: WilsonScore;
  effect_size?: EffectSize;
  statistical_tests?: StatisticalTest[];
  by_element?: Record<string, ElementSummary>;
  by_isotope?: Record<string, ElementSummary>;
}

/** Hypothesis outcome */
export interface HypothesisOutcome {
  supported: boolean;
  confidence: OutcomeConfidence;
  interpretation?: string;
  limitations?: string[];
  future_work?: string[];
}

/** Reproducibility information */
export interface Reproducibility {
  random_seed?: number;
  model_version?: string;
  adapter_version?: string;
  code_version?: string;
  environment?: {
    platform?: string;
    python_version?: string;
    mlx_version?: string;
  };
}

/** Complete experiment results */
export interface ExperimentResults {
  id: string;
  experiment_id: string;
  status: ResultsStatus;
  trials: Trial[];
  summary: ResultsSummary;
  hypothesis_outcome?: HypothesisOutcome;
  reproducibility?: Reproducibility;
  version: string;
  created_at?: string;
  completed_at?: string;
}

// =============================================================================
// COMPOUND TYPES
// =============================================================================

/** Element compound (sequence of elements) */
export interface Compound {
  id: string;
  name: string;
  sequence: string[];
  formula?: string;
  description?: string;
}

/** Compound analysis result */
export interface CompoundAnalysis {
  formula: string;
  stability: number;
  transitions: Array<{
    from: string;
    to: string;
    score: number;
    notes: string[];
  }>;
  emergent: Array<{
    name: string;
    icon: string;
  }>;
  manifold_polygon: number[][];
  coverage: {
    area: number;
    gaps: string[];
    centroid?: number[];
  };
}

// =============================================================================
// GROUP METADATA
// =============================================================================

export interface GroupMetadata {
  id: number;
  name: string;
  domain: string;
  direction: string;
  color: string;
  bg: string;
  border: string;
  text: string;
  hex: string;
}

export const Groups: Record<ElementGroup, GroupMetadata> = {
  epistemic: {
    id: 1,
    name: 'Epistemic',
    domain: 'Self-Knowledge',
    direction: 'Inward',
    color: 'from-blue-600 to-cyan-500',
    bg: 'bg-cyan-950/30',
    border: 'border-cyan-500/30',
    text: 'text-cyan-400',
    hex: '#22d3ee'
  },
  analytical: {
    id: 2,
    name: 'Analytical',
    domain: 'Decomposition',
    direction: 'Outward',
    color: 'from-amber-500 to-yellow-400',
    bg: 'bg-amber-950/30',
    border: 'border-amber-500/30',
    text: 'text-amber-400',
    hex: '#fbbf24'
  },
  generative: {
    id: 3,
    name: 'Generative',
    domain: 'Creation',
    direction: 'Outward',
    color: 'from-emerald-500 to-green-400',
    bg: 'bg-emerald-950/30',
    border: 'border-emerald-500/30',
    text: 'text-emerald-400',
    hex: '#34d399'
  },
  evaluative: {
    id: 4,
    name: 'Evaluative',
    domain: 'Judgment',
    direction: 'Outward',
    color: 'from-rose-500 to-red-400',
    bg: 'bg-rose-950/30',
    border: 'border-rose-500/30',
    text: 'text-rose-400',
    hex: '#fb7185'
  },
  dialogical: {
    id: 5,
    name: 'Dialogical',
    domain: 'Perspective',
    direction: 'Transverse',
    color: 'from-violet-500 to-purple-400',
    bg: 'bg-violet-950/30',
    border: 'border-violet-500/30',
    text: 'text-violet-400',
    hex: '#a78bfa'
  },
  pedagogical: {
    id: 6,
    name: 'Pedagogical',
    domain: 'Teaching',
    direction: 'Transverse',
    color: 'from-teal-500 to-cyan-400',
    bg: 'bg-teal-950/30',
    border: 'border-teal-500/30',
    text: 'text-teal-400',
    hex: '#2dd4bf'
  },
  temporal: {
    id: 7,
    name: 'Temporal',
    domain: 'Time',
    direction: 'Temporal',
    color: 'from-orange-500 to-amber-400',
    bg: 'bg-orange-950/30',
    border: 'border-orange-500/30',
    text: 'text-orange-400',
    hex: '#fb923c'
  },
  contextual: {
    id: 8,
    name: 'Contextual',
    domain: 'Situating',
    direction: 'Outward',
    color: 'from-slate-400 to-gray-300',
    bg: 'bg-slate-800/30',
    border: 'border-slate-500/30',
    text: 'text-slate-400',
    hex: '#94a3b8'
  }
};

// =============================================================================
// UTILITY TYPES
// =============================================================================

/** Database record wrapper */
export interface DBRecord<T> {
  data: T;
  metadata: {
    created_at: string;
    updated_at: string;
    version: string;
  };
}

/** Validation result */
export interface ValidationResult {
  valid: boolean;
  errors?: Array<{
    path: string;
    message: string;
  }>;
}
