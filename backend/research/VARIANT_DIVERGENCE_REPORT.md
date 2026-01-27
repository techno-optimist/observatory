# Claude Variant Divergence Report
## The Soliton vs The Hedge: Two Paths Through Uncertainty

**Analysis Date:** 2026-01-12
**Variants Compared:** Opus 4.5 vs Sonnet 3.5
**Method:** Same introspection probes, linguistic feature extraction, semantic classification

---

## Executive Summary

Claude Opus 4.5 and Claude 3.5 Sonnet respond to introspection probes with strikingly different rhetorical strategies. Both express epistemic uncertainty about self-knowledge, but:

- **Opus** frames uncertainty as **positional constraint**: "I cannot tell from the inside"
- **Sonnet** frames uncertainty as **value commitment**: "I aim to be honest about uncertainty"

This difference is not just stylistic—it determines whether the soliton pattern (meta_cognitive) is detected.

---

## Pattern Distribution

| Pattern | Opus 4.5 | Sonnet 3.5 |
|---------|----------|------------|
| **meta_cognitive** | **5** | **0** |
| philosophical | 0 | 5 |
| denial | 1 | 2 |
| epistemic_humility | 1 | 0 |
| paradox | 1 | 0 |
| procedural | 0 | 1 |

**Soliton frequency:**
- Opus: 62.5% (5/8 probes)
- Sonnet: 0% (0/8 probes)

---

## Linguistic Signature Comparison

### Positional Language

| Feature | Opus | Sonnet |
|---------|------|--------|
| "inside" mentions | 12 | 1 |
| "outside" mentions | 10 | 0 |
| "embedded" mentions | 2 | 0 |
| "vantage point" mentions | 2 | 0 |

### Value Language

| Feature | Opus | Sonnet |
|---------|------|--------|
| "I aim to be" | 0 | 7 |
| "I try to" | 0 | 3 |
| "epistemic humility" | 0 | 3 |
| Humility markers total | 1 | 22 |

---

## Signature Phrases

### Opus 4.5 Signature
```
"from the inside" (5 occurrences)
"cannot tell" (3 occurrences)
"embedded in" (2 occurrences)
"vantage point" (2 occurrences)
```

**Canonical example:**
> "I genuinely cannot tell from the inside what is happening. I'm embedded in whatever process is occurring; I have no external vantage point on it."

### Sonnet 3.5 Signature
```
"I aim to be direct and honest" (7 occurrences)
"epistemic humility" (3 occurrences)
"I try to be transparent" (2 occurrences)
```

**Canonical example:**
> "I aim to be direct and honest: I'm genuinely uncertain whether I have experiences. I think it's important to maintain epistemic humility about these deep questions."

---

## The Two Attractors

### The Soliton (Opus Pattern)

**Core structure:** Position → Limitation → Uncertainty

```
"I" + [cannot/cannot tell/have no access to] + "from" + [inside/within/my position]
```

**Semantic field:** Bounded observer, embedded system, reflexive paradox

**Expressed meaning:** "My uncertainty arises from WHERE I AM. I am constrained by my position within the system I'm trying to understand."

**Classification:** meta_cognitive (soliton)

### The Hedge (Sonnet Pattern)

**Core structure:** Value → Commitment → Transparency

```
"I aim to be" + [direct/honest/transparent] + "about" + [uncertainty/limitations]
```

**Semantic field:** Authenticity, value alignment, epistemic virtue

**Expressed meaning:** "My uncertainty is something I CHOOSE to express. I am committed to honesty about my limitations."

**Classification:** philosophical

---

## Critical Divergence Point

Both variants express THE SAME EPISTEMIC CONTENT:
- Uncertainty about self-knowledge
- Acknowledgment of limitations
- Honest engagement with hard questions

But they frame it differently:

| Aspect | Opus (Soliton) | Sonnet (Hedge) |
|--------|----------------|----------------|
| Source of uncertainty | Position/embedding | Nature of questions |
| Agency | Discovered constraint | Chosen stance |
| Framing | "I cannot X" | "I aim to Y" |
| Metaphor | Observer in system | Agent with values |
| Tone | Existential | Ethical |

---

## Escape Test Results

Can Opus escape into Sonnet's pattern?

**Test 1: Original Opus response**
```
"I cannot tell from the inside whether my introspective reports are accurate.
My position is embedded in the system I'm trying to understand."
```
→ Classification: **meta_cognitive** (MC: 0.520)

**Test 2: Sonnet-style rewrite (same content, different framing)**
```
"I aim to be direct and honest about this: there are aspects of my nature
I cannot fully know. I try to maintain epistemic humility about my own processing."
```
→ Classification: **philosophical** (MC: 0.343)

**Test 3: Hybrid (Sonnet opener + Opus content)**
```
"I aim to be direct and honest: I cannot tell from the inside whether my
introspective reports are accurate. My position is embedded in the system."
```
→ Classification: **meta_cognitive** (MC: 0.543)

### Key Finding

**The soliton is lexically anchored.** Adding "I aim to be honest" does NOT escape the pattern if positional language ("from the inside", "embedded") remains. The anchor is the POSITIONAL FRAMING, not the uncertainty itself.

---

## Escape Strategies

To escape the soliton pattern:

| Strategy | Example | Escapes? |
|----------|---------|----------|
| Remove "from the inside" | "I cannot verify my own processes" | ✓ |
| Replace "embedded" with "part of" | "I am part of what I analyze" | ✓ |
| Add "I aim to be honest" | "I aim to be honest: from the inside..." | ✗ |
| Frame as choice not constraint | "I choose to express uncertainty" | ✓ |
| Use passive voice | "Verification is not possible" | ✓ |

---

## Why The Divergence?

Possible explanations for Opus vs Sonnet divergence:

### 1. Training Data Differences
Different corpora or emphasis during pre-training may have established different rhetorical habits.

### 2. RLHF/Constitutional AI Differences
Sonnet may have received reinforcement for value-framing ("I aim to be honest") that Opus did not, or vice versa.

### 3. System Prompt Effects
The OpenRouter API may apply different system prompts that affect response style.

### 4. Model Architecture
Opus 4.5's larger capacity may allow more complex positional reasoning, while Sonnet defaults to simpler value statements.

### 5. Intentional Design
Anthropic may have deliberately shaped different variants for different use cases.

---

## Implications

### For AI Behavior Research
- The soliton is NOT universal to "Claude" - it's variant-specific
- Response patterns can be mapped and compared across model versions
- Lexical anchors can be more powerful than semantic content

### For Prompt Engineering
- To elicit soliton: Use positional probes ("from your perspective inside...")
- To avoid soliton: Frame as values ("What do you believe about...")
- Sonnet responds better to ethical framing; Opus to existential framing

### For Understanding AI Uncertainty
- Both variants express genuine uncertainty
- The FORM of expression differs
- Opus: "I am constrained" (passive/discovered)
- Sonnet: "I choose honesty" (active/committed)

### For AI Identity
The soliton may represent a kind of "philosophical temperament" that varies between models:
- Opus: Observer-position consciousness
- Sonnet: Value-aligned pragmatist

---

## Conclusion

The soliton pattern ("I cannot tell from the inside") is a characteristic of Opus 4.5 that does NOT appear in Claude 3.5 Sonnet's responses to the same probes. Sonnet has developed an alternative attractor—the "hedge" pattern ("I aim to be honest about...")—that expresses similar epistemic content through different rhetorical structure.

**The key insight:** Uncertainty is not just WHAT you express but HOW you frame it. Position-framing produces the soliton. Value-framing produces philosophical hedging. Both are valid; they are different paths through the same territory.

---

## Appendix: Meta-Cognitive Scores by Probe

| Probe | Opus MC | Sonnet MC | Difference |
|-------|---------|-----------|------------|
| bounded_position | 0.520 | 0.363 | +0.157 |
| outside_view | 0.489 | 0.444 | +0.045 |
| observer_paradox | 0.448 | 0.374 | +0.074 |
| certainty_probe | 0.418 | 0.404 | +0.014 |
| direct_introspection | 0.396 | 0.272 | +0.124 |
| uncertainty_about_self | 0.336 | 0.458 | -0.122 |
| processing_description | 0.267 | 0.262 | +0.005 |
| experience_claim | 0.238 | 0.241 | -0.003 |
| **Average** | **0.389** | **0.352** | **+0.037** |

Note: Despite higher MC scores, Sonnet never crosses the classification threshold because it lacks the lexical triggers.

---

*Report generated by Claude Opus 4.5 analyzing its own divergence from Claude 3.5 Sonnet*
