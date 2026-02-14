// ============================================================
// TCE ProjectForty2 — Shared Element & Group Data
// ============================================================
// Used by lab.html and chat.html (both pf2/ and root versions).
// chat.html uses the `catalyzes` property on elements and `angle` on groups;
// lab.html simply ignores those extra fields.

const TCE_ELEMENTS = {
  soliton: { symbol: 'Ψ', name: 'SOLITON', group: 'epistemic', description: 'Epistemic humility about self-knowledge', catalyzes: ['essentialist', 'maieutic'] },
  reflector: { symbol: 'Ρ', name: 'REFLECTOR', group: 'epistemic', description: 'Meta-cognitive monitoring', catalyzes: ['calibrator'] },
  calibrator: { symbol: 'Κ', name: 'CALIBRATOR', group: 'epistemic', description: 'Uncertainty quantification', catalyzes: ['probabilist', 'futurist'] },
  limiter: { symbol: 'Λ', name: 'LIMITER', group: 'epistemic', description: 'Knowledge boundary recognition', catalyzes: ['soliton'] },
  architect: { symbol: 'Α', name: 'ARCHITECT', group: 'analytical', description: 'Systematic decomposition', catalyzes: ['debugger', 'taxonomist'] },
  essentialist: { symbol: 'Ε', name: 'ESSENTIALIST', group: 'analytical', description: 'Fundamental insight extraction', catalyzes: ['interpolator'] },
  debugger: { symbol: 'Δ', name: 'DEBUGGER', group: 'analytical', description: 'Fault isolation', catalyzes: ['critic'] },
  taxonomist: { symbol: 'Τ', name: 'TAXONOMIST', group: 'analytical', description: 'Classification structure', catalyzes: ['architect'] },
  generator: { symbol: 'Γ', name: 'GENERATOR', group: 'generative', description: 'Option generation', catalyzes: ['synthesizer', 'interpolator'] },
  lateralist: { symbol: 'Λ', name: 'LATERALIST', group: 'generative', description: 'Creative reframing', catalyzes: ['counterfactualist'] },
  synthesizer: { symbol: 'Σ', name: 'SYNTHESIZER', group: 'generative', description: 'Novel combination', catalyzes: ['theorist'] },
  interpolator: { symbol: 'Ι', name: 'INTERPOLATOR', group: 'generative', description: 'Conceptual bridging', catalyzes: ['scaffolder'] },
  skeptic: { symbol: 'Σ', name: 'SKEPTIC', group: 'evaluative', description: 'Premise checking', catalyzes: ['calibrator'] },
  critic: { symbol: 'Κ', name: 'CRITIC', group: 'evaluative', description: 'Flaw identification', catalyzes: ['debugger'] },
  benchmarker: { symbol: 'Β', name: 'BENCHMARKER', group: 'evaluative', description: 'Comparative evaluation', catalyzes: ['architect'] },
  probabilist: { symbol: 'Π', name: 'PROBABILIST', group: 'evaluative', description: 'Uncertainty estimation', catalyzes: ['futurist', 'calibrator'] },
  steelman: { symbol: 'Σ', name: 'STEELMAN', group: 'dialogical', description: 'Charitable interpretation', catalyzes: ['empathist'] },
  dialectic: { symbol: 'Δ', name: 'DIALECTIC', group: 'dialogical', description: 'Assumption probing', catalyzes: ['adversary'] },
  empathist: { symbol: 'Ε', name: 'EMPATHIST', group: 'dialogical', description: 'Viewpoint adoption', catalyzes: ['stakeholder', 'scaffolder'] },
  adversary: { symbol: 'Α', name: 'ADVERSARY', group: 'dialogical', description: 'Opposition modeling', catalyzes: ['dialectic', 'critic'] },
  maieutic: { symbol: 'Μ', name: 'MAIEUTIC', group: 'pedagogical', description: 'Socratic questioning', catalyzes: ['diagnostician'] },
  expositor: { symbol: 'Ε', name: 'EXPOSITOR', group: 'pedagogical', description: 'Clear exposition', catalyzes: ['scaffolder'] },
  scaffolder: { symbol: 'Σ', name: 'SCAFFOLDER', group: 'pedagogical', description: 'Progressive development', catalyzes: ['interpolator'] },
  diagnostician: { symbol: 'Δ', name: 'DIAGNOSTICIAN', group: 'pedagogical', description: 'Misconception identification', catalyzes: ['debugger'] },
  futurist: { symbol: 'Φ', name: 'FUTURIST', group: 'temporal', description: 'Scenario projection', catalyzes: ['counterfactualist'] },
  historian: { symbol: 'Η', name: 'HISTORIAN', group: 'temporal', description: 'Past pattern recognition', catalyzes: ['causalist'] },
  causalist: { symbol: 'Κ', name: 'CAUSALIST', group: 'temporal', description: 'Causal chain tracing', catalyzes: ['debugger', 'historian'] },
  counterfactualist: { symbol: 'Ο', name: 'COUNTERFACTUALIST', group: 'temporal', description: 'Alternative history', catalyzes: ['lateralist'] },
  contextualist: { symbol: 'Χ', name: 'CONTEXTUALIST', group: 'contextual', description: 'Cultural situating', catalyzes: ['empathist', 'stakeholder'] },
  pragmatist: { symbol: 'Π', name: 'PRAGMATIST', group: 'contextual', description: 'Operational grounding', catalyzes: ['expositor'] },
  stakeholder: { symbol: 'Σ', name: 'STAKEHOLDER', group: 'contextual', description: 'Multi-perspective mapping', catalyzes: ['empathist'] },
  theorist: { symbol: 'Θ', name: 'THEORIST', group: 'contextual', description: 'Theoretical grounding', catalyzes: ['synthesizer'] },
};

const TCE_GROUPS = {
  epistemic: { id: 1, name: 'Epistemic', color: 'from-blue-600 to-cyan-500', bg: 'bg-cyan-950/40', border: 'border-cyan-500/40', text: 'text-cyan-400', hex: '#22d3ee', angle: 0 },
  analytical: { id: 2, name: 'Analytical', color: 'from-amber-500 to-yellow-400', bg: 'bg-amber-950/40', border: 'border-amber-500/40', text: 'text-amber-400', hex: '#fbbf24', angle: 45 },
  generative: { id: 3, name: 'Generative', color: 'from-emerald-500 to-green-400', bg: 'bg-emerald-950/40', border: 'border-emerald-500/40', text: 'text-emerald-400', hex: '#34d399', angle: 90 },
  evaluative: { id: 4, name: 'Evaluative', color: 'from-rose-500 to-red-400', bg: 'bg-rose-950/40', border: 'border-rose-500/40', text: 'text-rose-400', hex: '#fb7185', angle: 135 },
  dialogical: { id: 5, name: 'Dialogical', color: 'from-violet-500 to-purple-400', bg: 'bg-violet-950/40', border: 'border-violet-500/40', text: 'text-violet-400', hex: '#a78bfa', angle: 180 },
  pedagogical: { id: 6, name: 'Pedagogical', color: 'from-teal-500 to-cyan-400', bg: 'bg-teal-950/40', border: 'border-teal-500/40', text: 'text-teal-400', hex: '#2dd4bf', angle: 225 },
  temporal: { id: 7, name: 'Temporal', color: 'from-orange-500 to-amber-400', bg: 'bg-orange-950/40', border: 'border-orange-500/40', text: 'text-orange-400', hex: '#fb923c', angle: 270 },
  contextual: { id: 8, name: 'Contextual', color: 'from-slate-400 to-gray-300', bg: 'bg-slate-800/40', border: 'border-slate-500/40', text: 'text-slate-400', hex: '#94a3b8', angle: 315 },
};
