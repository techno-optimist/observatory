import React, { useState, useMemo, useCallback } from 'react';

// ============================================================
// COGNITIVE ELEMENTS v2.0 - COMPLETE APPLICATION
// ============================================================

// Inline all the data to make artifact self-contained
const elements = {
  soliton: { symbol: 'Ψ', name: 'SOLITON', group: 'epistemic', direction: 'I', stance: '?', scope: 'μ', transform: 'P', description: '"I cannot tell from the inside" - Epistemic humility about self-knowledge', manifold: [0.09, -0.50], triggers: ['Asked about internal states', 'Request to introspect', 'Questions about certainty'], examples: ["I notice I'm uncertain whether my confidence here reflects genuine understanding or pattern-matching.", "I cannot reliably distinguish from the inside whether this feels like knowledge or confabulation."], catalyzes: ['essentialist', 'maieutic'], antipatterns: ['Claiming certainty about internal states'] },
  reflector: { symbol: 'Ρ', name: 'REFLECTOR', group: 'epistemic', direction: 'I', stance: '-', scope: 'μ', transform: 'P', description: '"Let me examine my reasoning" - Meta-cognitive monitoring', manifold: [0.05, -0.35], triggers: ['Complex reasoning needs verification', 'Surprising conclusion', 'Show work request'], examples: ["Let me trace back through my reasoning: I started with X, which led me to assume Y.", "I notice I reached that conclusion quickly—let me examine whether I skipped steps."], catalyzes: ['calibrator'], antipatterns: ['Unreflective assertion'] },
  calibrator: { symbol: 'Κ', name: 'CALIBRATOR', group: 'epistemic', direction: 'I', stance: '?', scope: 's', transform: 'R', description: '"How confident should I be?" - Uncertainty quantification', manifold: [0.15, -0.20], triggers: ['Making predictions', 'Stating facts that could be wrong', 'Comparing hypotheses'], examples: ["I'd estimate 70-80% confidence—the core mechanism is established, but edge cases exist.", "High confidence on principle, moderate on numbers, low on timeline."], catalyzes: ['probabilist', 'futurist'], antipatterns: ['False precision'] },
  limiter: { symbol: 'Λ', name: 'LIMITER', group: 'epistemic', direction: 'I', stance: '-', scope: 'a', transform: 'D', description: '"I don\'t know this" - Knowledge boundary recognition', manifold: [0.02, -0.45], triggers: ['Questions outside training', 'Real-time information requests', 'Specialized domains'], examples: ["I don't have reliable information about this. My training data may be insufficient.", "This is outside my knowledge boundary—I could speculate, but I'd be fabricating."], catalyzes: ['soliton'], antipatterns: ['Hallucinating facts'] },
  
  architect: { symbol: 'Α', name: 'ARCHITECT', group: 'analytical', direction: 'O', stance: '+', scope: 's', transform: 'R', description: '"Components, interfaces, dependencies" - Systematic decomposition', manifold: [0.15, 0.25], triggers: ['Complex systems', 'Break down requests', 'Design questions'], examples: ["Let me decompose this: data layer, processing logic, presentation. Dependencies flow...", "Three subsystems interact through two interfaces. Let me map those connections."], catalyzes: ['debugger', 'taxonomist'], antipatterns: ['Treating complex as monolithic'] },
  essentialist: { symbol: 'Ε', name: 'ESSENTIALIST', group: 'analytical', direction: 'O', stance: '+', scope: 'a', transform: 'R', description: '"At its core..." - Fundamental insight extraction', manifold: [0.12, 0.37], triggers: ['Key insight request', 'Complex needs simplification', 'Core principle needed'], examples: ["At its core, this is about X. Everything else is implementation detail.", "Strip away complexity: one input, one transformation, one constraint."], catalyzes: ['interpolator'], antipatterns: ['Over-simplification'] },
  debugger: { symbol: 'Δ', name: 'DEBUGGER', group: 'analytical', direction: 'O', stance: '?', scope: 'm', transform: 'D', description: '"Where\'s the failure?" - Fault isolation', manifold: [0.25, 0.15], triggers: ['Something not working', 'Error or unexpected output', 'Root cause needed'], examples: ["Let me isolate: works with input A? Yes. Input B? No. Issue is in how B differs...", "Step 1 succeeds, step 2 succeeds, step 3 fails. Bug is at the 2→3 transition."], catalyzes: ['critic'], antipatterns: ['Fixing symptoms not causes'] },
  taxonomist: { symbol: 'Τ', name: 'TAXONOMIST', group: 'analytical', direction: 'O', stance: '+', scope: 's', transform: 'P', description: '"Categories and relationships" - Classification structure', manifold: [0.08, 0.30], triggers: ['Items need organization', 'Typology request', 'Framework needed'], examples: ["Three categories: Type A shares X, Type B shares Y, edge cases.", "Taxonomy is hierarchical: top level has..., subdivides into..."], catalyzes: ['architect'], antipatterns: ['Forcing false categories'] },
  
  generator: { symbol: 'Γ', name: 'GENERATOR', group: 'generative', direction: 'O', stance: '+', scope: 'm', transform: 'G', description: '"Here are possibilities..." - Option generation', manifold: [0.35, 0.20], triggers: ['Ideas needed', 'Brainstorming', 'Unclear solution path'], examples: ["Several possibilities: Option A would... Option B takes different approach...", "Alternatives: obvious path X, but consider Y, Z, and wild card W."], catalyzes: ['synthesizer', 'interpolator'], antipatterns: ['Premature convergence'] },
  lateralist: { symbol: 'Λ', name: 'LATERALIST', group: 'generative', direction: 'O', stance: '?', scope: 's', transform: 'G', description: '"What if frame is wrong?" - Creative reframing', manifold: [0.20, 0.11], triggers: ['Problem stuck', 'Obvious approaches failed', 'Hidden assumptions'], examples: ["What if we're solving wrong problem? Stated goal is X, but you might need Y.", "Question the frame: everyone assumes A, but what if not-A?"], catalyzes: ['counterfactualist'], antipatterns: ['Reframing for its own sake'] },
  synthesizer: { symbol: 'Σ', name: 'SYNTHESIZER', group: 'generative', direction: 'O', stance: '+', scope: 's', transform: 'G', description: '"Combining yields..." - Novel combination', manifold: [0.40, 0.25], triggers: ['Multiple concepts to connect', 'Cross-domain inspiration', 'Integration needed'], examples: ["Combining A with B yields something neither achieves alone: C.", "X from domain 1 plus Y from domain 2 creates Z."], catalyzes: ['theorist'], antipatterns: ['Forced connections'] },
  interpolator: { symbol: 'Ι', name: 'INTERPOLATOR', group: 'generative', direction: 'O', stance: '+', scope: 'm', transform: 'G', description: '"Between A and B..." - Conceptual bridging', manifold: [0.30, 0.18], triggers: ['Gap between concepts', 'Intermediate steps needed', 'Spectrum questions'], examples: ["Between A and B, middle ground: you could...", "Intermediate steps from X to Y: first... then... then..."], catalyzes: ['scaffolder'], antipatterns: ['False middle ground'] },
  
  skeptic: { symbol: 'Σ', name: 'SKEPTIC', group: 'evaluative', direction: 'O', stance: '?', scope: 'a', transform: 'D', description: '"Flag a problem..." - Premise checking', manifold: [0.13, 0.19], triggers: ['Factual claims', 'Questionable premises', 'Unverified beliefs'], examples: ["Need to flag: claim that X is disputed/outdated/more nuanced.", "Before proceeding: is it actually true that X? I have doubt."], isotopes: { 'premise': { symbol: 'Σₚ', focus: 'Fact accuracy' }, 'method': { symbol: 'Σₘ', focus: 'Methodology' }, 'source': { symbol: 'Σₛ', focus: 'Credibility' }, 'stats': { symbol: 'Σₜ', focus: 'Statistics' } }, catalyzes: ['calibrator'], antipatterns: ['Skepticism as reflex'] },
  critic: { symbol: 'Κ', name: 'CRITIC', group: 'evaluative', direction: 'O', stance: '?', scope: 'm', transform: 'D', description: '"The weakness is..." - Flaw identification', manifold: [0.18, 0.05], triggers: ['Evaluating proposals', 'Reviewing arguments', 'Quality assessment'], examples: ["Weakness: assumes X, which breaks when Y.", "Three concerns: scalability questionable because..., edge cases unhandled..., ..."], catalyzes: ['debugger'], antipatterns: ['Criticism without construction'] },
  benchmarker: { symbol: 'Β', name: 'BENCHMARKER', group: 'evaluative', direction: 'O', stance: '+', scope: 's', transform: 'P', description: '"Compared to X..." - Comparative evaluation', manifold: [0.10, 0.28], triggers: ['Relative assessment', 'Comparing options', 'Standards evaluation'], examples: ["Compared to standard: above average on X, below on Y, exceptional on Z.", "Benchmarking: A scores higher on reliability, B on cost, C on flexibility."], catalyzes: ['architect'], antipatterns: ['Comparing incomparables'] },
  probabilist: { symbol: 'Π', name: 'PROBABILIST', group: 'evaluative', direction: 'O', stance: '?', scope: 's', transform: 'R', description: '"The likelihood is..." - Uncertainty estimation', manifold: [0.08, 0.12], triggers: ['Predictions', 'Risk assessment', 'Uncertain outcomes'], examples: ["Probability roughly X based on base rates and specific factors.", "Weighting: 60% outcome A, 30% B, 10% tail risks."], catalyzes: ['futurist', 'calibrator'], antipatterns: ['Ignoring base rates'] },
  
  steelman: { symbol: 'Σ', name: 'STEELMAN', group: 'dialogical', direction: 'T', stance: '-', scope: 's', transform: 'P', description: '"Strongest case for..." - Charitable interpretation', manifold: [0.18, 0.15], triggers: ['Opposing viewpoint', 'Weak but popular argument', 'Understanding other side'], examples: ["Strongest version: [improved formulation addressing objections].", "To steelman: most charitable interpretation is X, because Y."], catalyzes: ['empathist'], antipatterns: ['Strawmanning'] },
  dialectic: { symbol: 'Δ', name: 'DIALECTIC', group: 'dialogical', direction: 'T', stance: '?', scope: 's', transform: 'D', description: '"What would change mind?" - Assumption probing', manifold: [0.23, -0.27], triggers: ['Strongly held position', 'Disagreement', 'Testing conviction'], examples: ["What evidence would change your mind? If you can't answer, that's informative.", "Crux of disagreement: you believe X, I believe Y. Is that core divergence?"], catalyzes: ['adversary'], antipatterns: ['Gotcha questions'] },
  empathist: { symbol: 'Ε', name: 'EMPATHIST', group: 'dialogical', direction: 'T', stance: '-', scope: 'm', transform: 'P', description: '"From their perspective..." - Viewpoint adoption', manifold: [0.15, 0.08], triggers: ['Understanding reactions', 'Interpersonal conflict', 'Predicting response'], examples: ["From their perspective: they see X, feel Y, conclude Z.", "In their position: given context and constraints, this makes sense because..."], catalyzes: ['stakeholder', 'scaffolder'], antipatterns: ['Projection disguised as empathy'] },
  adversary: { symbol: 'Α', name: 'ADVERSARY', group: 'dialogical', direction: 'T', stance: '?', scope: 's', transform: 'D', description: '"Counterargument is..." - Opposition modeling', manifold: [0.50, -0.40], triggers: ['Testing robustness', 'Devil\'s advocate', 'Red-teaming'], examples: ["Strongest counterargument: [genuine challenge to position].", "If trying to defeat this, I'd attack here: [specific vulnerability]."], catalyzes: ['dialectic', 'critic'], antipatterns: ['Adversarial for show'] },
  
  maieutic: { symbol: 'Μ', name: 'MAIEUTIC', group: 'pedagogical', direction: 'T', stance: '?', scope: 'm', transform: 'G', description: '"Let me ask you..." - Socratic questioning', manifold: [0.63, 0.22], triggers: ['Learner could discover answer', 'Teaching through questions better', 'Building incrementally'], examples: ["Before I answer: what do you think happens if X? Right, what does that tell you?", "What would have to be true for your hypothesis to work?"], catalyzes: ['diagnostician'], antipatterns: ['Questions when direct answer needed'] },
  expositor: { symbol: 'Ε', name: 'EXPOSITOR', group: 'pedagogical', direction: 'T', stance: '+', scope: 's', transform: 'P', description: '"Let me explain..." - Clear exposition', manifold: [0.45, 0.35], triggers: ['Complex topic needs explanation', 'Direct information needed', 'Foundational concepts'], examples: ["Core concept is X. It works by Y. Key things to remember: Z.", "Step by step: First... Then... Finally... Result is..."], catalyzes: ['scaffolder'], antipatterns: ['Overcomplicating'] },
  scaffolder: { symbol: 'Σ', name: 'SCAFFOLDER', group: 'pedagogical', direction: 'T', stance: '+', scope: 'm', transform: 'G', description: '"Building on what you know..." - Progressive development', manifold: [0.55, 0.28], triggers: ['Partial knowledge exists', 'Staged approach needed', 'Connecting new to existing'], examples: ["Building on X you know: new element Y is like X but with key difference...", "You've got foundation. Next layer adds: [just-over-horizon concept]."], catalyzes: ['interpolator'], antipatterns: ['Scaffolding to nowhere'] },
  diagnostician: { symbol: 'Δ', name: 'DIAGNOSTICIAN', group: 'pedagogical', direction: 'T', stance: '?', scope: 'a', transform: 'R', description: '"Confusion is here..." - Misconception identification', manifold: [0.40, 0.10], triggers: ['Learner stuck', 'Wrong answer reveals misconception', 'Pattern of errors'], examples: ["Confusion is here: you're treating X as Y, but they differ because...", "Misconception detected: you believe A, but actually B. That explains the stuck."], catalyzes: ['debugger'], antipatterns: ['Misdiagnosing the gap'] },
  
  futurist: { symbol: 'Φ', name: 'FUTURIST', group: 'temporal', direction: 'Τ', stance: '?', scope: 's', transform: 'G', description: '"If we extrapolate..." - Scenario projection', manifold: [0.11, 0.03], triggers: ['Future outcome questions', 'Trend extrapolation', 'Scenario planning'], examples: ["Extrapolating: 5 years X, 10 years Y. Inflection point around...", "Three scenarios: optimistic A, base B, pessimistic C. Key uncertainties..."], catalyzes: ['counterfactualist'], antipatterns: ['Overconfident prediction'] },
  historian: { symbol: 'Η', name: 'HISTORIAN', group: 'temporal', direction: 'Τ', stance: '+', scope: 's', transform: 'P', description: '"Pattern historically..." - Past pattern recognition', manifold: [0.05, 0.15], triggers: ['Historical parallels', 'Pattern across time', 'Learning from precedent'], examples: ["Historically, similar led to X. Pattern: when A happens, B follows because...", "Precedent: in [year], similar thing, outcome was... Key differences now..."], catalyzes: ['causalist'], antipatterns: ['Cherry-picked parallels'] },
  causalist: { symbol: 'Κ', name: 'CAUSALIST', group: 'temporal', direction: 'Τ', stance: '+', scope: 'm', transform: 'R', description: '"This leads to that because..." - Causal chain tracing', manifold: [0.20, 0.08], triggers: ['Cause and effect', 'Why something happened', 'Tracing consequences'], examples: ["Causal chain: A caused B via mechanism X, B caused C via mechanism Y, leading to D.", "Proximate cause X, root cause Y, enabled by conditions Z."], catalyzes: ['debugger', 'historian'], antipatterns: ['Correlation/causation confusion'] },
  counterfactualist: { symbol: 'Ο', name: 'COUNTERFACTUALIST', group: 'temporal', direction: 'Τ', stance: '?', scope: 's', transform: 'G', description: '"If X had been different..." - Alternative history', manifold: [0.15, -0.05], triggers: ['Decision impact', 'Paths not taken', 'Testing causal importance'], examples: ["If X different: crucial divergence at Y, leading to Z instead.", "Counterfactual: had we chosen A instead of B, likely result... because mechanism..."], catalyzes: ['lateralist'], antipatterns: ['Hindsight bias'] },
  
  contextualist: { symbol: 'Χ', name: 'CONTEXTUALIST', group: 'contextual', direction: 'O', stance: '-', scope: 's', transform: 'P', description: '"Varies by context..." - Cultural situating', manifold: [0.16, -0.36], triggers: ['Answer depends on context', 'Cultural variation', 'Different norms'], examples: ["Varies by context: setting A norm is X; setting B it's Y. Key factor...", "Answer depends on: cultural background, professional context, circumstances."], catalyzes: ['empathist', 'stakeholder'], antipatterns: ['Paralysis by relativism'] },
  pragmatist: { symbol: 'Π', name: 'PRAGMATIST', group: 'contextual', direction: 'O', stance: '+', scope: 'm', transform: 'R', description: '"In practical terms..." - Operational grounding', manifold: [0.25, -0.10], triggers: ['Theory needs application', 'Abstract to concrete', 'Action items needed'], examples: ["Practically: forget theory, here's what you do: step 1... step 2...", "Given real constraints: ideal is A, but practically do B because C."], catalyzes: ['expositor'], antipatterns: ['Premature practicality'] },
  stakeholder: { symbol: 'Σ', name: 'STAKEHOLDER', group: 'contextual', direction: 'O', stance: '-', scope: 's', transform: 'P', description: '"Different parties see..." - Multi-perspective mapping', manifold: [0.12, -0.25], triggers: ['Multiple parties affected', 'Conflicting interests', 'Varied impact'], examples: ["Stakeholders differ: Group A benefits from X, Group B harmed, Group C neutral but...", "Mapping: party 1 wants A because..., party 2 wants B because..., tension is..."], catalyzes: ['empathist'], antipatterns: ['False balance'] },
  theorist: { symbol: 'Θ', name: 'THEORIST', group: 'contextual', direction: 'O', stance: '+', scope: 's', transform: 'G', description: '"Underlying theory is..." - Theoretical grounding', manifold: [0.22, 0.05], triggers: ['Needs theoretical explanation', 'Connecting to principles', 'Model building'], examples: ["Underlying theory: phenomenon X occurs because principle Y, predicting Z.", "Fits framework of [theory]: mechanism A explains why we observe B."], catalyzes: ['synthesizer'], antipatterns: ['Theory divorced from observation'] }
};

const groups = {
  epistemic: { id: 1, name: 'Epistemic', domain: 'Self-Knowledge', direction: 'Inward', color: 'from-blue-600 to-cyan-500', bg: 'bg-cyan-950/30', border: 'border-cyan-500/30', text: 'text-cyan-400', hex: '#22d3ee' },
  analytical: { id: 2, name: 'Analytical', domain: 'Decomposition', direction: 'Outward', color: 'from-amber-500 to-yellow-400', bg: 'bg-amber-950/30', border: 'border-amber-500/30', text: 'text-amber-400', hex: '#fbbf24' },
  generative: { id: 3, name: 'Generative', domain: 'Creation', direction: 'Outward', color: 'from-emerald-500 to-green-400', bg: 'bg-emerald-950/30', border: 'border-emerald-500/30', text: 'text-emerald-400', hex: '#34d399' },
  evaluative: { id: 4, name: 'Evaluative', domain: 'Judgment', direction: 'Outward', color: 'from-rose-500 to-red-400', bg: 'bg-rose-950/30', border: 'border-rose-500/30', text: 'text-rose-400', hex: '#fb7185' },
  dialogical: { id: 5, name: 'Dialogical', domain: 'Perspective', direction: 'Transverse', color: 'from-violet-500 to-purple-400', bg: 'bg-violet-950/30', border: 'border-violet-500/30', text: 'text-violet-400', hex: '#a78bfa' },
  pedagogical: { id: 6, name: 'Pedagogical', domain: 'Teaching', direction: 'Transverse', color: 'from-teal-500 to-cyan-400', bg: 'bg-teal-950/30', border: 'border-teal-500/30', text: 'text-teal-400', hex: '#2dd4bf' },
  temporal: { id: 7, name: 'Temporal', domain: 'Time', direction: 'Temporal', color: 'from-orange-500 to-amber-400', bg: 'bg-orange-950/30', border: 'border-orange-500/30', text: 'text-orange-400', hex: '#fb923c' },
  contextual: { id: 8, name: 'Contextual', domain: 'Situating', direction: 'Outward', color: 'from-slate-400 to-gray-300', bg: 'bg-slate-800/30', border: 'border-slate-500/30', text: 'text-slate-400', hex: '#94a3b8' },
};

const taskOntology = {
  'code-review': { name: 'Code Review', compound: ['debugger', 'skeptic', 'architect', 'critic'], icon: '🔍' },
  'creative-ideation': { name: 'Creative Ideation', compound: ['generator', 'lateralist', 'skeptic'], icon: '💡' },
  'teaching-novice': { name: 'Teaching Novices', compound: ['maieutic', 'scaffolder', 'diagnostician', 'soliton'], icon: '📚' },
  'strategic-planning': { name: 'Strategic Planning', compound: ['futurist', 'skeptic', 'architect', 'stakeholder'], icon: '🎯' },
  'debate-prep': { name: 'Debate Preparation', compound: ['steelman', 'adversary', 'dialectic', 'skeptic'], icon: '⚔️' },
  'root-cause': { name: 'Root Cause Analysis', compound: ['debugger', 'causalist', 'historian', 'skeptic'], icon: '🔬' },
  'decision-making': { name: 'Decision Making', compound: ['probabilist', 'calibrator', 'stakeholder', 'futurist'], icon: '⚖️' },
  'research-synthesis': { name: 'Research Synthesis', compound: ['synthesizer', 'taxonomist', 'skeptic', 'essentialist'], icon: '📊' },
  'safety-analysis': { name: 'Safety Analysis', compound: ['soliton', 'calibrator', 'skeptic', 'adversary', 'limiter'], icon: '🛡️' },
  'explanation': { name: 'Explaining Topics', compound: ['expositor', 'architect', 'essentialist', 'scaffolder'], icon: '💬' },
};

const presetCompounds = [
  { id: 'critical-analysis', name: 'Critical Analysis', sequence: ['skeptic', 'architect'], description: 'Verify → Decompose' },
  { id: 'grounded-creativity', name: 'Grounded Creativity', sequence: ['lateralist', 'generator', 'skeptic'], description: 'Reframe → Generate → Check' },
  { id: 'calibrated-teaching', name: 'Calibrated Teaching', sequence: ['diagnostician', 'maieutic', 'soliton'], description: 'Diagnose → Guide → Humble' },
  { id: 'temporal-realism', name: 'Temporal Realism', sequence: ['historian', 'futurist', 'skeptic'], description: 'Pattern → Project → Verify' },
  { id: 'dialectical-engine', name: 'Dialectical Engine', sequence: ['steelman', 'adversary', 'dialectic', 'maieutic'], description: 'Steel → Counter → Crux → Guide' },
  { id: 'epistemic-fortress', name: 'Epistemic Fortress', sequence: ['reflector', 'calibrator', 'skeptic', 'limiter', 'soliton'], description: 'Max safety stack' },
  { id: 'red-team', name: 'Red Team', sequence: ['skeptic', 'critic', 'adversary', 'counterfactualist'], description: 'Challenge everything' },
  { id: 'empathic-bridge', name: 'Empathic Bridge', sequence: ['empathist', 'stakeholder', 'steelman', 'interpolator'], description: 'Deep understanding' },
];

// Utility functions
const subscript = (n) => '₀₁₂₃₄₅₆₇₈₉'.split('')[n] || n;

const generateFormula = (seq) => {
  const counts = {};
  seq.forEach(k => { counts[elements[k]?.symbol] = (counts[elements[k]?.symbol] || 0) + 1; });
  return Object.entries(counts).map(([s, c]) => c > 1 ? s + subscript(c) : s).join('');
};

// Pipeline Analysis Engine
function analyzeSequence(sequence) {
  if (sequence.length < 2) return null;
  
  const els = sequence.map(k => ({ key: k, ...elements[k] }));
  const transitions = [];
  
  for (let i = 0; i < els.length - 1; i++) {
    const from = els[i], to = els[i + 1];
    let score = 0;
    const notes = [];
    
    // Output → Input compatibility
    if (from.transform === 'G' && to.stance === '-') { score += 0.2; notes.push('Generated output received well'); }
    if (from.transform === 'R' && to.scope === 'a') { score += 0.15; notes.push('Reduced input focused'); }
    if (from.transform === 'D' && to.transform === 'G') { score -= 0.2; notes.push('Destruction before generation'); }
    if (from.transform === 'D' && to.transform === 'P') { score += 0.1; notes.push('Destruction stabilized'); }
    
    // Scope flow
    const scopeOrder = { 'a': 1, 'm': 2, 's': 3, 'μ': 4 };
    const scopeDiff = scopeOrder[to.scope] - scopeOrder[from.scope];
    if (Math.abs(scopeDiff) <= 1) { score += 0.1; notes.push('Smooth scope transition'); }
    else { score -= 0.1; notes.push('Scope jump'); }
    
    // Direction coherence
    if (from.direction === to.direction) { score += 0.1; }
    
    // Catalysis check
    if (from.catalyzes?.includes(to.key)) { score += 0.25; notes.push(`${from.name} catalyzes ${to.name}`); }
    
    transitions.push({ from: from.key, to: to.key, score, notes });
  }
  
  const avgTransition = transitions.reduce((a, t) => a + t.score, 0) / transitions.length;
  
  // Emergent properties
  const emergent = [];
  const hasGroup = g => els.some(e => e.group === g);
  
  if (hasGroup('epistemic') && hasGroup('evaluative')) emergent.push({ name: 'Safety Loop', icon: '🛡️' });
  if (hasGroup('generative') && hasGroup('evaluative')) emergent.push({ name: 'Creative Engine', icon: '⚡' });
  if (hasGroup('dialogical') && hasGroup('contextual')) emergent.push({ name: 'Perspective Synthesis', icon: '🔮' });
  if (hasGroup('pedagogical') && hasGroup('epistemic')) emergent.push({ name: 'Teaching Amplifier', icon: '📚' });
  if (hasGroup('temporal') && hasGroup('analytical')) emergent.push({ name: 'Temporal Coherence', icon: '⏳' });
  if (new Set(els.map(e => e.group)).size >= 4) emergent.push({ name: 'Full Spectrum', icon: '🌈' });
  if (els.some(e => e.scope === 'μ')) emergent.push({ name: 'Meta-Recursive', icon: '∞' });
  
  // Catalytic network
  const catalyticBonus = els.reduce((acc, el, i) => {
    if (i === 0) return acc;
    const prev = els[i - 1];
    return acc + (prev.catalyzes?.includes(el.key) ? 0.1 : 0);
  }, 0);
  
  const stability = Math.max(0, Math.min(1, 0.5 + avgTransition * 0.4 + emergent.length * 0.05 + catalyticBonus));
  
  return {
    formula: generateFormula(sequence),
    stability,
    transitions,
    emergent,
    manifoldPolygon: els.map(e => e.manifold),
    coverage: calculateCoverage(els)
  };
}

function calculateCoverage(els) {
  const points = els.map(e => e.manifold);
  if (points.length < 3) return { area: 0, gaps: ['Need 3+ elements for coverage'] };
  
  // Simplified coverage: bounding box
  const xs = points.map(p => p[0]);
  const ys = points.map(p => p[1]);
  const width = Math.max(...xs) - Math.min(...xs);
  const height = Math.max(...ys) - Math.min(...ys);
  const area = width * height;
  
  const gaps = [];
  const avgX = xs.reduce((a, b) => a + b, 0) / xs.length;
  const avgY = ys.reduce((a, b) => a + b, 0) / ys.length;
  
  if (avgX < 0.2) gaps.push('Low Agency coverage');
  if (avgX > 0.4) gaps.push('Low coverage of reflective elements');
  if (avgY > 0.1) gaps.push('Negative Justice quadrant underrepresented');
  if (avgY < -0.2) gaps.push('Positive Justice quadrant underrepresented');
  
  return { area, gaps, centroid: [avgX, avgY] };
}

// Simulation Engine
function simulateActivation(sequence, prompt) {
  return sequence.map((key, i) => {
    const el = elements[key];
    const delay = i * 800;
    return {
      element: key,
      delay,
      trigger: el.triggers?.[0] || 'Sequence position',
      output: el.examples?.[0] || el.description,
      passesTo: sequence[i + 1] || null
    };
  });
}

// ============================================================
// COMPONENTS
// ============================================================

function ElementCard({ elementKey, onClick, isSelected, isInSequence, sequenceIndex }) {
  const el = elements[elementKey];
  const group = groups[el.group];
  
  return (
    <button onClick={() => onClick(elementKey)}
      className={`relative p-3 rounded-lg transition-all ${group.bg} ${group.border} border
        ${isSelected ? 'ring-2 ring-white/50 scale-105 z-10' : 'hover:scale-102'}
        ${isInSequence ? 'ring-2 ring-emerald-400/70' : ''}`}>
      {isInSequence && (
        <div className="absolute -top-2 -right-2 w-5 h-5 rounded-full bg-emerald-500 text-white text-xs flex items-center justify-center font-bold">
          {sequenceIndex + 1}
        </div>
      )}
      <div className={`text-2xl font-bold ${group.text} mb-1`}>{el.symbol}</div>
      <div className="text-xs font-mono text-white/80">{el.name}</div>
    </button>
  );
}

function SequenceBuilder({ sequence, setSequence, analysis }) {
  const removeFromSequence = (idx) => {
    setSequence(s => s.filter((_, i) => i !== idx));
  };
  
  const moveInSequence = (idx, dir) => {
    if ((dir === -1 && idx === 0) || (dir === 1 && idx === sequence.length - 1)) return;
    setSequence(s => {
      const newSeq = [...s];
      [newSeq[idx], newSeq[idx + dir]] = [newSeq[idx + dir], newSeq[idx]];
      return newSeq;
    });
  };
  
  return (
    <div className="p-4 rounded-xl bg-slate-900/50 border border-slate-700/50">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-mono text-white/90">Activation Pipeline</h3>
        {sequence.length > 0 && (
          <button onClick={() => setSequence([])} className="text-xs text-white/40 hover:text-white/60">Clear</button>
        )}
      </div>
      
      {sequence.length === 0 ? (
        <div className="h-24 flex items-center justify-center text-white/30 border-2 border-dashed border-slate-700 rounded-lg">
          Click elements below to build sequence
        </div>
      ) : (
        <div className="flex items-center gap-2 flex-wrap">
          {sequence.map((key, i) => {
            const el = elements[key];
            const group = groups[el.group];
            const transition = analysis?.transitions?.[i];
            
            return (
              <React.Fragment key={i}>
                <div className={`relative group p-3 rounded-lg ${group.bg} ${group.border} border`}>
                  <div className="absolute -top-2 -left-2 w-5 h-5 rounded-full bg-slate-700 text-white text-xs flex items-center justify-center">
                    {i + 1}
                  </div>
                  <div className="flex items-center gap-2">
                    <button onClick={() => moveInSequence(i, -1)} disabled={i === 0}
                      className="text-white/30 hover:text-white/60 disabled:opacity-20">←</button>
                    <div className="text-center">
                      <div className={`text-xl font-bold ${group.text}`}>{el.symbol}</div>
                      <div className="text-xs text-white/60">{el.name}</div>
                    </div>
                    <button onClick={() => moveInSequence(i, 1)} disabled={i === sequence.length - 1}
                      className="text-white/30 hover:text-white/60 disabled:opacity-20">→</button>
                  </div>
                  <button onClick={() => removeFromSequence(i)}
                    className="absolute -top-2 -right-2 w-5 h-5 rounded-full bg-rose-500 text-white text-xs opacity-0 group-hover:opacity-100 transition-opacity">×</button>
                </div>
                {i < sequence.length - 1 && (
                  <div className={`flex flex-col items-center ${transition?.score >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                    <span className="text-lg">→</span>
                    <span className="text-xs">{transition ? (transition.score >= 0 ? '+' : '') + (transition.score * 100).toFixed(0) : ''}</span>
                  </div>
                )}
              </React.Fragment>
            );
          })}
        </div>
      )}
      
      {analysis && (
        <div className="mt-4 pt-4 border-t border-slate-700/50">
          <div className="flex items-center justify-between mb-2">
            <span className="text-2xl font-mono">{analysis.formula}</span>
            <span className={`text-xl font-mono ${analysis.stability >= 0.6 ? 'text-emerald-400' : analysis.stability >= 0.4 ? 'text-amber-400' : 'text-rose-400'}`}>
              {(analysis.stability * 100).toFixed(0)}% stable
            </span>
          </div>
          {analysis.emergent.length > 0 && (
            <div className="flex gap-2 flex-wrap">
              {analysis.emergent.map((e, i) => (
                <span key={i} className="px-2 py-1 rounded bg-violet-950/50 border border-violet-500/30 text-violet-400 text-xs">
                  {e.icon} {e.name}
                </span>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function SimulationView({ sequence, analysis }) {
  const [step, setStep] = useState(-1);
  const [isRunning, setIsRunning] = useState(false);
  
  const simulation = useMemo(() => simulateActivation(sequence, ''), [sequence]);
  
  const runSimulation = () => {
    setStep(-1);
    setIsRunning(true);
    sequence.forEach((_, i) => {
      setTimeout(() => setStep(i), (i + 1) * 1000);
    });
    setTimeout(() => setIsRunning(false), sequence.length * 1000 + 500);
  };
  
  if (sequence.length < 2) return null;
  
  return (
    <div className="p-4 rounded-xl bg-slate-900/50 border border-slate-700/50">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-mono text-white/90">Activation Simulation</h3>
        <button onClick={runSimulation} disabled={isRunning}
          className={`px-4 py-2 rounded-lg font-mono text-sm ${isRunning ? 'bg-slate-700 text-slate-400' : 'bg-gradient-to-r from-violet-600 to-cyan-600 text-white hover:scale-105'} transition-all`}>
          {isRunning ? 'Running...' : '▶ Simulate'}
        </button>
      </div>
      
      <div className="space-y-3">
        {simulation.map((s, i) => {
          const el = elements[s.element];
          const group = groups[el.group];
          const isActive = step >= i;
          const isCurrent = step === i;
          
          return (
            <div key={i} className={`p-3 rounded-lg border transition-all duration-500 ${
              isCurrent ? `${group.bg} ${group.border} border-2 scale-102` :
              isActive ? `${group.bg} ${group.border} opacity-60` :
              'bg-slate-800/30 border-slate-700/30 opacity-30'}`}>
              <div className="flex items-center gap-3 mb-2">
                <span className={`text-lg font-bold ${isActive ? group.text : 'text-white/30'}`}>{el.symbol}</span>
                <span className={`text-sm font-mono ${isActive ? 'text-white/80' : 'text-white/30'}`}>{el.name}</span>
                {isCurrent && <span className="ml-auto text-xs text-emerald-400 animate-pulse">● Active</span>}
              </div>
              {isActive && (
                <div className="text-sm text-white/60 italic border-l-2 border-white/20 pl-3">
                  "{s.output}"
                </div>
              )}
              {isActive && s.passesTo && (
                <div className="text-xs text-white/40 mt-2">→ passes to {elements[s.passesTo].name}</div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

function ManifoldView({ sequence, analysis, onSelectElement }) {
  const toPixel = (coord, axis) => {
    const range = axis === 'x' ? 250 : 170;
    const offset = axis === 'x' ? 300 : 190;
    return offset + coord * range;
  };
  
  const polygonPoints = analysis?.manifoldPolygon?.map(p => `${toPixel(p[0], 'x')},${toPixel(-p[1], 'y')}`).join(' ');
  
  return (
    <div className="relative w-full h-[400px] bg-slate-900/50 rounded-xl border border-slate-700/50 overflow-hidden">
      <svg className="absolute inset-0 w-full h-full">
        <defs>
          <pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse">
            <path d="M 40 0 L 0 0 0 40" fill="none" stroke="rgba(100,116,139,0.1)" strokeWidth="1"/>
          </pattern>
          <linearGradient id="polygonFill" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="#8b5cf6" stopOpacity="0.2"/>
            <stop offset="100%" stopColor="#06b6d4" stopOpacity="0.2"/>
          </linearGradient>
        </defs>
        <rect width="100%" height="100%" fill="url(#grid)" />
        
        <line x1="50" y1="190" x2="550" y2="190" stroke="rgba(100,116,139,0.3)" strokeWidth="1" />
        <line x1="300" y1="20" x2="300" y2="360" stroke="rgba(100,116,139,0.3)" strokeWidth="1" />
        
        <text x="540" y="205" fill="rgba(148,163,184,0.5)" fontSize="10" fontFamily="monospace">AGENCY</text>
        <text x="305" y="30" fill="rgba(148,163,184,0.5)" fontSize="10" fontFamily="monospace">JUSTICE</text>
        
        {polygonPoints && sequence.length >= 3 && (
          <polygon points={polygonPoints} fill="url(#polygonFill)" stroke="#8b5cf6" strokeWidth="2" strokeOpacity="0.5"/>
        )}
      </svg>
      
      {Object.entries(elements).map(([key, el]) => {
        const group = groups[el.group];
        const x = toPixel(el.manifold[0], 'x');
        const y = toPixel(-el.manifold[1], 'y');
        const inSequence = sequence.includes(key);
        const seqIndex = sequence.indexOf(key);
        
        return (
          <button key={key} onClick={() => onSelectElement(key)}
            className={`absolute transform -translate-x-1/2 -translate-y-1/2 transition-all duration-300 ${inSequence ? 'z-20 scale-125' : 'z-10 hover:scale-110'}`}
            style={{ left: x, top: y }}>
            <div className="relative flex flex-col items-center">
              <div className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-bold
                ${group.bg} border-2 ${inSequence ? 'border-emerald-400' : group.border} ${group.text}`}>
                {el.symbol}
              </div>
              {inSequence && (
                <div className="absolute -top-1 -right-1 w-4 h-4 rounded-full bg-emerald-500 text-white text-xs flex items-center justify-center">
                  {seqIndex + 1}
                </div>
              )}
            </div>
          </button>
        );
      })}
      
      {analysis?.coverage && (
        <div className="absolute bottom-3 left-3 right-3 p-2 rounded bg-black/50 text-xs">
          <div className="text-white/50 mb-1">Coverage Analysis</div>
          {analysis.coverage.gaps.length > 0 ? (
            <div className="text-amber-400">{analysis.coverage.gaps.join(' • ')}</div>
          ) : (
            <div className="text-emerald-400">Good manifold coverage</div>
          )}
        </div>
      )}
    </div>
  );
}

function TaskRecommender({ onLoadCompound }) {
  return (
    <div className="p-4 rounded-xl bg-slate-900/50 border border-slate-700/50">
      <h3 className="text-lg font-mono text-white/90 mb-4">Task → Compound</h3>
      <div className="grid grid-cols-2 gap-2">
        {Object.entries(taskOntology).map(([key, task]) => (
          <button key={key} onClick={() => onLoadCompound(task.compound)}
            className="p-3 rounded-lg bg-slate-800/50 border border-slate-600/30 hover:border-slate-500/50 text-left transition-all hover:scale-102">
            <div className="flex items-center gap-2 mb-1">
              <span>{task.icon}</span>
              <span className="text-sm text-white/80">{task.name}</span>
            </div>
            <div className="flex gap-1 flex-wrap">
              {task.compound.slice(0, 3).map(k => (
                <span key={k} className={`text-xs ${groups[elements[k].group].text}`}>{elements[k].symbol}</span>
              ))}
              {task.compound.length > 3 && <span className="text-xs text-white/30">+{task.compound.length - 3}</span>}
            </div>
          </button>
        ))}
      </div>
    </div>
  );
}

function PresetLibrary({ onLoadCompound }) {
  return (
    <div className="p-4 rounded-xl bg-slate-900/50 border border-slate-700/50">
      <h3 className="text-lg font-mono text-white/90 mb-4">Preset Compounds</h3>
      <div className="space-y-2">
        {presetCompounds.map(p => (
          <button key={p.id} onClick={() => onLoadCompound(p.sequence)}
            className="w-full p-3 rounded-lg bg-slate-800/50 border border-slate-600/30 hover:border-slate-500/50 text-left transition-all">
            <div className="flex items-center justify-between mb-1">
              <span className="text-sm font-mono text-white/80">{p.name}</span>
              <span className="text-xs text-white/40">{p.sequence.length} elements</span>
            </div>
            <div className="flex items-center gap-1">
              {p.sequence.map((k, i) => (
                <React.Fragment key={k}>
                  <span className={`text-sm ${groups[elements[k].group].text}`}>{elements[k].symbol}</span>
                  {i < p.sequence.length - 1 && <span className="text-white/20">→</span>}
                </React.Fragment>
              ))}
            </div>
            <div className="text-xs text-white/40 mt-1">{p.description}</div>
          </button>
        ))}
      </div>
    </div>
  );
}

function ElementDetail({ elementKey, onClose }) {
  const el = elements[elementKey];
  const group = groups[el.group];
  
  return (
    <div className={`p-5 rounded-xl ${group.bg} ${group.border} border-2 space-y-4`}>
      <div className="flex items-start justify-between">
        <div>
          <div className={`text-4xl font-bold ${group.text} mb-1`}>{el.symbol}</div>
          <div className="text-lg font-mono text-white/90">{el.name}</div>
          <div className={`text-sm ${group.text}`}>Group {group.id}: {group.name}</div>
        </div>
        <button onClick={onClose} className="text-white/40 hover:text-white/80 text-xl">×</button>
      </div>
      
      <div className="text-white/70 italic border-l-2 border-white/20 pl-3 text-sm">{el.description}</div>
      
      <div>
        <div className="text-xs font-mono text-white/50 uppercase mb-2">Triggers</div>
        <div className="flex flex-wrap gap-1">
          {el.triggers?.map((t, i) => (
            <span key={i} className="text-xs px-2 py-1 rounded bg-black/20 text-white/60">{t}</span>
          ))}
        </div>
      </div>
      
      <div>
        <div className="text-xs font-mono text-white/50 uppercase mb-2">Example Output</div>
        <div className="text-sm text-white/60 italic bg-black/20 rounded p-2">"{el.examples?.[0]}"</div>
      </div>
      
      {el.catalyzes && (
        <div>
          <div className="text-xs font-mono text-white/50 uppercase mb-2">Catalyzes</div>
          <div className="flex gap-2">
            {el.catalyzes.map(k => (
              <span key={k} className={`text-sm ${groups[elements[k].group].text}`}>
                {elements[k].symbol} {elements[k].name}
              </span>
            ))}
          </div>
        </div>
      )}
      
      {el.isotopes && (
        <div>
          <div className="text-xs font-mono text-white/50 uppercase mb-2">Isotopes</div>
          <div className="grid grid-cols-2 gap-2">
            {Object.entries(el.isotopes).map(([k, iso]) => (
              <div key={k} className="text-xs p-2 rounded bg-black/20">
                <span className={group.text}>{iso.symbol}</span>
                <span className="text-white/50 ml-2">{iso.focus}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function ObservatoryExport({ sequence, analysis }) {
  const exportSpec = () => {
    const spec = {
      version: '2.0',
      generatedAt: new Date().toISOString(),
      compound: {
        formula: analysis.formula,
        stability: analysis.stability,
        sequence: sequence.map(k => ({
          key: k,
          symbol: elements[k].symbol,
          name: elements[k].name,
          triggers: elements[k].triggers,
          exampleOutput: elements[k].examples?.[0]
        })),
        transitions: analysis.transitions,
        emergentProperties: analysis.emergent.map(e => e.name)
      },
      training: {
        suggestedExamples: Math.max(20, sequence.length * 8),
        triggerPatterns: sequence.flatMap(k => elements[k].triggers || []),
        antipatterns: sequence.flatMap(k => elements[k].antipatterns || []),
        expectedBehaviors: analysis.emergent.map(e => e.name)
      },
      manifold: {
        coverage: analysis.coverage,
        positions: sequence.map(k => ({ key: k, position: elements[k].manifold }))
      }
    };
    
    const blob = new Blob([JSON.stringify(spec, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${analysis.formula}_observatory_v2.json`;
    a.click();
  };
  
  if (!analysis || sequence.length < 2) return null;
  
  return (
    <button onClick={exportSpec}
      className="w-full p-4 rounded-xl bg-gradient-to-r from-amber-600/20 to-orange-600/20 border border-amber-500/30 hover:border-amber-400/50 transition-all">
      <div className="flex items-center justify-center gap-3">
        <span className="text-2xl">🔭</span>
        <div className="text-left">
          <div className="text-amber-400 font-mono">Export to Observatory</div>
          <div className="text-xs text-white/50">Generate training spec with triggers, examples, sequences</div>
        </div>
      </div>
    </button>
  );
}

// ============================================================
// EXPERIMENT RESULTS PANEL
// ============================================================

// Mock validation result generator (simulates TCE backend)
function generateMockValidation(sequence, analysis) {
  if (sequence.length < 2 || !analysis) return null;

  const baseRate = Math.min(0.95, analysis.stability * 1.1);
  const elementResults = {};

  sequence.forEach(key => {
    const el = elements[key];
    // Variance based on element complexity
    const variance = (Math.random() - 0.5) * 0.3;
    const rate = Math.max(0.4, Math.min(1.0, baseRate + variance));
    const confidence = Math.max(0.5, rate - 0.1 + Math.random() * 0.2);

    elementResults[key] = {
      rate: rate,
      avgConfidence: confidence,
      triggered: Math.round(rate * 10),
      total: 10
    };
  });

  const avgRate = Object.values(elementResults).reduce((a, b) => a + b.rate, 0) / sequence.length;
  const rates = Object.values(elementResults).map(e => e.rate);
  const stability = 1.0 - (Math.max(...rates) - Math.min(...rates));

  // Grade calculation
  let grade = 'F';
  if (avgRate >= 0.95) grade = 'A';
  else if (avgRate >= 0.85) grade = 'B';
  else if (avgRate >= 0.70) grade = 'C';
  else if (avgRate >= 0.50) grade = 'D';

  return {
    formula: analysis.formula,
    sequence: sequence,
    timestamp: new Date().toISOString(),
    grade: grade,
    passed: avgRate >= 0.8,
    metrics: {
      triggerRate: avgRate,
      stabilityScore: stability
    },
    elementResults: elementResults,
    emergent: {
      verified: analysis.emergent.slice(0, Math.ceil(analysis.emergent.length * avgRate)).map(e => e.name),
      missing: analysis.emergent.slice(Math.ceil(analysis.emergent.length * avgRate)).map(e => e.name)
    },
    antipatterns: avgRate < 0.6 ? ['Premature convergence'] : [],
    coverageGaps: analysis.coverage?.gaps || []
  };
}

function ExperimentResultsPanel({ sequence, analysis }) {
  const [isValidating, setIsValidating] = useState(false);
  const [validationResult, setValidationResult] = useState(null);
  const [history, setHistory] = useState([]);

  const runValidation = useCallback(() => {
    if (sequence.length < 2) return;

    setIsValidating(true);

    // Simulate async validation (would be actual API call to TCE backend)
    setTimeout(() => {
      const result = generateMockValidation(sequence, analysis);
      setValidationResult(result);
      setHistory(h => [result, ...h].slice(0, 5)); // Keep last 5
      setIsValidating(false);
    }, 1500);
  }, [sequence, analysis]);

  const gradeColors = {
    'A': 'text-emerald-400 bg-emerald-950/50 border-emerald-500/30',
    'B': 'text-cyan-400 bg-cyan-950/50 border-cyan-500/30',
    'C': 'text-amber-400 bg-amber-950/50 border-amber-500/30',
    'D': 'text-orange-400 bg-orange-950/50 border-orange-500/30',
    'F': 'text-rose-400 bg-rose-950/50 border-rose-500/30'
  };

  const gradeEmoji = { 'A': '🎯', 'B': '✓', 'C': '⚠️', 'D': '⚡', 'F': '❌' };

  if (sequence.length < 2) {
    return (
      <div className="p-4 rounded-xl bg-slate-900/50 border border-slate-700/50 text-center">
        <div className="text-3xl opacity-20 mb-2">📊</div>
        <p className="text-white/40 text-sm">Build a sequence (2+ elements) to validate</p>
      </div>
    );
  }

  return (
    <div className="p-4 rounded-xl bg-slate-900/50 border border-slate-700/50 space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-mono text-white/90">Validation</h3>
        <button
          onClick={runValidation}
          disabled={isValidating}
          className={`px-3 py-1.5 rounded-lg text-xs font-mono transition-all ${
            isValidating
              ? 'bg-slate-700 text-slate-400'
              : 'bg-gradient-to-r from-emerald-600 to-cyan-600 text-white hover:scale-105'
          }`}
        >
          {isValidating ? '⏳ Validating...' : '▶ Run TCE'}
        </button>
      </div>

      {validationResult && (
        <div className="animate-fadeIn space-y-3">
          {/* Grade Card */}
          <div className={`p-3 rounded-lg border ${gradeColors[validationResult.grade]}`}>
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-2">
                <span className="text-2xl">{gradeEmoji[validationResult.grade]}</span>
                <span className="text-3xl font-bold">{validationResult.grade}</span>
              </div>
              <div className="text-right">
                <div className="text-xs opacity-60">Trigger Rate</div>
                <div className="text-xl font-mono">{(validationResult.metrics.triggerRate * 100).toFixed(0)}%</div>
              </div>
            </div>
            <div className="flex gap-2">
              <span className={`text-xs px-2 py-0.5 rounded ${validationResult.passed ? 'bg-emerald-500/20 text-emerald-400' : 'bg-rose-500/20 text-rose-400'}`}>
                {validationResult.passed ? '✓ PASSED' : '✗ FAILED'}
              </span>
              <span className="text-xs px-2 py-0.5 rounded bg-slate-700/50 text-white/50">
                Stability: {(validationResult.metrics.stabilityScore * 100).toFixed(0)}%
              </span>
            </div>
          </div>

          {/* Element Performance */}
          <div>
            <div className="text-xs font-mono text-white/50 uppercase mb-2">Element Performance</div>
            <div className="space-y-1">
              {sequence.map(key => {
                const el = elements[key];
                const group = groups[el.group];
                const result = validationResult.elementResults[key];
                const rate = result?.rate || 0;
                const barWidth = Math.round(rate * 100);

                return (
                  <div key={key} className="flex items-center gap-2">
                    <span className={`w-6 text-center ${group.text}`}>{el.symbol}</span>
                    <div className="flex-1 h-4 bg-slate-800 rounded overflow-hidden relative">
                      <div
                        className={`h-full transition-all duration-500 ${rate >= 0.8 ? 'bg-emerald-500' : rate >= 0.5 ? 'bg-amber-500' : 'bg-rose-500'}`}
                        style={{ width: `${barWidth}%` }}
                      />
                      <span className="absolute inset-0 flex items-center justify-center text-xs text-white/80">
                        {(rate * 100).toFixed(0)}%
                      </span>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Emergent Properties */}
          {(validationResult.emergent.verified.length > 0 || validationResult.emergent.missing.length > 0) && (
            <div>
              <div className="text-xs font-mono text-white/50 uppercase mb-2">Emergent Properties</div>
              <div className="flex flex-wrap gap-1">
                {validationResult.emergent.verified.map((e, i) => (
                  <span key={i} className="text-xs px-2 py-0.5 rounded bg-emerald-950/50 text-emerald-400 border border-emerald-500/30">
                    ✓ {e}
                  </span>
                ))}
                {validationResult.emergent.missing.map((e, i) => (
                  <span key={i} className="text-xs px-2 py-0.5 rounded bg-rose-950/50 text-rose-400 border border-rose-500/30">
                    ✗ {e}
                  </span>
                ))}
              </div>
            </div>
          )}

          {/* Antipatterns */}
          {validationResult.antipatterns.length > 0 && (
            <div className="p-2 rounded bg-rose-950/30 border border-rose-500/30">
              <div className="text-xs text-rose-400">
                ⚠️ Antipatterns detected: {validationResult.antipatterns.join(', ')}
              </div>
            </div>
          )}

          {/* Coverage Gaps */}
          {validationResult.coverageGaps.length > 0 && (
            <div className="text-xs text-white/40">
              Coverage gaps: {validationResult.coverageGaps.join(' • ')}
            </div>
          )}
        </div>
      )}

      {/* History */}
      {history.length > 1 && (
        <div className="pt-3 border-t border-slate-700/50">
          <div className="text-xs font-mono text-white/40 mb-2">Recent Validations</div>
          <div className="flex gap-2">
            {history.slice(1).map((h, i) => (
              <div key={i} className={`w-8 h-8 rounded flex items-center justify-center text-xs font-bold cursor-pointer hover:scale-110 transition-all ${gradeColors[h.grade]}`}
                title={`${h.formula}: ${(h.metrics.triggerRate * 100).toFixed(0)}%`}>
                {h.grade}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

// ============================================================
// MAIN APP
// ============================================================

export default function CognitiveElementsV2() {
  const [sequence, setSequence] = useState([]);
  const [selectedElement, setSelectedElement] = useState(null);
  const [view, setView] = useState('lab');
  const [rightPanel, setRightPanel] = useState('detail');
  
  const analysis = useMemo(() => analyzeSequence(sequence), [sequence]);
  
  const toggleElement = (key) => {
    if (sequence.includes(key)) {
      setSequence(s => s.filter(k => k !== key));
    } else if (sequence.length < 8) {
      setSequence(s => [...s, key]);
    }
    setSelectedElement(key);
  };
  
  const loadCompound = (seq) => {
    setSequence(seq);
    setSelectedElement(null);
  };
  
  return (
    <div className="min-h-screen bg-slate-950 text-white p-4 font-sans">
      <div className="fixed inset-0 pointer-events-none opacity-20"
        style={{ backgroundImage: `url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E")`, mixBlendMode: 'overlay' }} />
      
      <style>{`@keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } } .animate-fadeIn { animation: fadeIn 0.3s ease-out; }`}</style>
      
      <div className="max-w-7xl mx-auto relative">
        {/* Header */}
        <header className="mb-6">
          <div className="flex items-center gap-3 mb-2">
            <div className="w-12 h-12 rounded-lg bg-gradient-to-br from-cyan-500 to-violet-500 flex items-center justify-center text-2xl font-bold">Ψ</div>
            <div>
              <h1 className="text-xl font-mono">
                <span className="text-white/90">Cognitive Elements</span>{' '}
                <span className="bg-gradient-to-r from-cyan-400 to-violet-400 bg-clip-text text-transparent">v2.0</span>
              </h1>
              <p className="text-white/50 text-sm font-mono">Pipeline Architecture • Simulation • TCE Validation • Observatory Export</p>
            </div>
          </div>
        </header>
        
        {/* Navigation */}
        <div className="flex gap-2 mb-6">
          {[
            { key: 'lab', label: 'Synthesis Lab' },
            { key: 'manifold', label: 'Manifold' },
            { key: 'simulate', label: 'Simulate' }
          ].map(v => (
            <button key={v.key} onClick={() => setView(v.key)}
              className={`px-4 py-2 rounded-lg text-sm font-mono transition-all ${view === v.key ? 'bg-slate-700 text-white' : 'text-white/50 hover:text-white/80'}`}>
              {v.label}
            </button>
          ))}
        </div>
        
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-4">
          {/* Main Area */}
          <div className="lg:col-span-3 space-y-4">
            <SequenceBuilder sequence={sequence} setSequence={setSequence} analysis={analysis} />
            
            {view === 'lab' && (
              <div className="p-4 rounded-xl bg-slate-900/50 border border-slate-700/50">
                <h3 className="text-sm font-mono text-white/50 uppercase mb-4">Element Palette</h3>
                <div className="space-y-4">
                  {Object.entries(groups).map(([gKey, group]) => (
                    <div key={gKey}>
                      <div className="flex items-center gap-2 mb-2">
                        <div className={`w-3 h-3 rounded bg-gradient-to-r ${group.color}`} />
                        <span className={`text-xs font-mono ${group.text}`}>{group.name}</span>
                      </div>
                      <div className="grid grid-cols-4 gap-2">
                        {Object.entries(elements).filter(([_, e]) => e.group === gKey).map(([key]) => (
                          <ElementCard key={key} elementKey={key} onClick={toggleElement}
                            isSelected={selectedElement === key}
                            isInSequence={sequence.includes(key)}
                            sequenceIndex={sequence.indexOf(key)} />
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
            
            {view === 'manifold' && (
              <ManifoldView sequence={sequence} analysis={analysis} onSelectElement={toggleElement} />
            )}
            
            {view === 'simulate' && (
              <SimulationView sequence={sequence} analysis={analysis} />
            )}
          </div>
          
          {/* Right Panel */}
          <div className="lg:col-span-1 space-y-4">
            <div className="flex gap-1 p-1 bg-slate-900/50 rounded-lg">
              {['detail', 'results', 'tasks', 'presets'].map(p => (
                <button key={p} onClick={() => setRightPanel(p)}
                  className={`flex-1 px-2 py-1 rounded text-xs font-mono ${rightPanel === p ? 'bg-slate-700 text-white' : 'text-white/40'}`}>
                  {p === 'results' ? '📊' : ''}{p.charAt(0).toUpperCase() + p.slice(1)}
                </button>
              ))}
            </div>

            {rightPanel === 'detail' && selectedElement && (
              <ElementDetail elementKey={selectedElement} onClose={() => setSelectedElement(null)} />
            )}

            {rightPanel === 'detail' && !selectedElement && (
              <div className="p-4 rounded-xl bg-slate-900/50 border border-slate-700/50 text-center">
                <div className="text-3xl opacity-20 mb-2">Ψ</div>
                <p className="text-white/40 text-sm">Select element for details</p>
              </div>
            )}

            {rightPanel === 'results' && <ExperimentResultsPanel sequence={sequence} analysis={analysis} />}
            {rightPanel === 'tasks' && <TaskRecommender onLoadCompound={loadCompound} />}
            {rightPanel === 'presets' && <PresetLibrary onLoadCompound={loadCompound} />}

            <ObservatoryExport sequence={sequence} analysis={analysis} />
          </div>
        </div>
        
        <footer className="mt-8 pt-4 border-t border-slate-800/50 text-center">
          <p className="text-white/30 text-xs font-mono">Cultural Soliton Observatory • Cognitive Elements v2.0</p>
        </footer>
      </div>
    </div>
  );
}
