================================================================================
OBSERVATORY RESEARCH AGENT - SESSION REPORT
Session ID: 20260107_151602
Experiments Run: 8
Findings: 2
================================================================================

## HYPOTHESIS STATUS

? [H001] Negation ('not X') projects differently than affirmation ('X')
✓ [H002] Questions score lower on agency than statements
✗ [H003] Future tense increases fairness vs past tense
✓ [H004] First person plural ('we') increases belonging vs singular ('I')
✗ [H005] Passive voice reduces agency vs active voice
? [H006] Hedging ('might', 'perhaps') reduces confidence in all axes
✗ [H007] There exist texts that maximize all three axes simultaneously
? [H008] Metaphorical agency is weaker than literal agency

## CONFIRMED FINDINGS

### F001: Questions score lower on agency than statements
Significance: high
Description: Confirmed with effect size 1.34
Statistics: {
  "mean_agency_delta": -0.30950061678886415,
  "mean_fairness_delta": 0.08320013508200645,
  "mean_belonging_delta": -0.16343830823898314,
  "std_agency_delta": 0.2208402191419504,
  "std_fairness_delta": 0.08550148232570329,
  "std_belonging_delta": 0.05574006487371966,
  "n_pairs": 5
}

### F002: First person plural ('we') increases belonging vs singular ('I')
Significance: high
Description: Confirmed with effect size 1.92
Statistics: {
  "mean_agency_delta": -0.2309415191411972,
  "mean_fairness_delta": 0.16227110624313354,
  "mean_belonging_delta": 0.8673618406057357,
  "std_agency_delta": 0.12506926681443642,
  "std_fairness_delta": 0.3098261186186458,
  "std_belonging_delta": 0.44208945320711723,
  "n_pairs": 5
}
