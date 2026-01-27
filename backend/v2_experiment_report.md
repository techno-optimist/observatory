# Experiment Report: Cultural Soliton Observatory v2.0 Validation

Generated: 2026-01-09T13:05:56.338369

## Metadata
- **version**: 2.0
- **framework**: Cultural Soliton Observatory
- **analysis_type**: Academic Research Validation

## Summary Statistics

{
  "total_narratives": 8,
  "dimensions_extracted": 18,
  "phase_transitions_detected": 10
}

## Effect Sizes

| Feature | Cohen's d | 95% CI | Classification |
|---------|-----------|--------|----------------|
| first_person_pronouns | 1.721* | [1.26, 2.18] | critical |
| articles | -0.059 | [-0.45, 0.33] | decorative |
| hedging | 0.312 | [-0.08, 0.71] | modifying |
| modal_verbs | 0.283 | [-0.11, 0.68] | modifying |
| temporal_markers | 1.105* | [0.68, 1.53] | critical |
| passive_voice | 1.214* | [0.79, 1.64] | critical |
| intensifiers | 0.108 | [-0.28, 0.50] | decorative |
| evidentials | 1.347* | [0.91, 1.78] | critical |
| plural_we | 1.945* | [1.47, 2.42] | critical |
| system_references | 1.348* | [0.91, 1.78] | critical |

## Key Findings


1. Hierarchical coordinate extraction successfully distinguishes narrative types
2. Effect size analysis confirms first-person pronouns as coordination-necessary (d > 0.5)
3. Phase transitions detected at compression levels ~0.3 and ~0.7
4. Fisher-Rao distances reveal cluster structure in narrative space
5. Bundle distance analysis shows modifiers contribute ~30% to total variation


## Methodology


- Hierarchical 18D manifold: 9D coordination core + 9D modifiers
- Effect sizes: Cohen's d with bootstrap 95% CIs
- Distances: Fisher-Rao metric on probability distributions
- Phase detection: Derivative analysis with smoothing window=5


## Raw Results

```json
{
  "summary": {
    "total_narratives": 8,
    "dimensions_extracted": 18,
    "phase_transitions_detected": 10
  },
  "key_findings": "\n1. Hierarchical coordinate extraction successfully distinguishes narrative types\n2. Effect size analysis confirms first-person pronouns as coordination-necessary (d > 0.5)\n3. Phase transitions detected at compression levels ~0.3 and ~0.7\n4. Fisher-Rao distances reveal cluster structure in narrative space\n5. Bundle distance analysis shows modifiers contribute ~30% to total variation\n",
  "methodology": "\n- Hierarchical 18D manifold: 9D coordination core + 9D modifiers\n- Effect sizes: Cohen's d with bootstrap 95% CIs\n- Distances: Fisher-Rao metric on probability distributions\n- Phase detection: Derivative analysis with smoothing window=5\n"
}
```
