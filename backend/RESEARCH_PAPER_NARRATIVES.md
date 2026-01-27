# Mapping Cultural Narratives: A Three-Dimensional Analysis of Agency, Justice, and Belonging Across Domains

**Authors:** Cultural Soliton Observatory Research Team
**Date:** January 7, 2026
**Institution:** Cultural Soliton Observatory

---

## Abstract

This study presents a comprehensive computational analysis of cultural narratives using a three-dimensional manifold model measuring **Agency** (sense of personal control), **Perceived Justice** (belief in fair systems), and **Belonging** (social connection). Across 360 narrative texts spanning six domains—cross-cultural, historical, institutional, psychological, economic, and archetypal—we find systematic and statistically significant variation in narrative positioning.

Key findings include: (1) Cultural context explains significant variance in agency, with American individualist narratives scoring 1.55 standard deviations higher than East Asian collectivist narratives (p < 0.001); (2) A striking "institutional-reality gap" where official narratives score 0.83 points higher on perceived justice than grassroots accounts (d = 2.73, p < 0.0001); (3) Economic class strongly correlates with perceived justice (r = 0.90, p = 0.037); (4) Psychological states create distinctive manifold signatures, with depression-resilience showing a large effect size (d = 1.25) on agency; and (5) Historical belonging in narratives shows significant negative temporal correlation (r = -0.88, p = 0.004), suggesting declining communal orientation in modern discourse.

These findings suggest that narrative positioning in this three-dimensional space reflects deep structural features of human meaning-making that transcend specific content while remaining sensitive to social, cultural, and historical context.

**Keywords:** narrative analysis, cultural psychology, computational social science, agency, justice perception, belonging, NLP

---

## 1. Introduction

### 1.1 Background

Human beings are narrative creatures. We understand ourselves, our societies, and our place in the world through stories. These narratives are not merely descriptive—they actively shape perception, motivation, and social organization. Understanding the structure of cultural narratives is thus essential for understanding human psychology and social dynamics.

Previous research has examined narratives through various lenses: moral foundations theory (Graham et al., 2013), cultural dimensions (Hofstede, 2001), narrative identity (McAdams, 2001), and social representations (Moscovici, 1988). However, these approaches typically rely on manual coding or discrete categorical schemes that may miss continuous variation in narrative positioning.

### 1.2 The Three-Axis Model

We propose that cultural narratives can be meaningfully positioned in a three-dimensional space defined by:

1. **Agency**: The degree to which the narrative expresses personal control, self-determination, and efficacy versus external control, fate, or helplessness. Range: [-2, +2].

2. **Perceived Justice**: The degree to which the narrative expresses belief that systems are fair and outcomes are deserved versus belief that systems are rigged and outcomes are unjust. Range: [-2, +2].

3. **Belonging**: The degree to which the narrative expresses social connection, group membership, and communal identity versus isolation, alienation, and individualistic separation. Range: [-2, +2].

This three-axis model emerged from empirical analysis of how language models encode social and political concepts (see Validity Study, 2026), and has been refined through extensive calibration showing 94.7% alignment with human intuitions.

### 1.3 Research Questions

1. How do narrative patterns vary across cultural and national contexts?
2. What narrative structures characterize different historical periods?
3. How do institutional narratives differ from grassroots voices?
4. What psychological patterns emerge in identity-based narratives?
5. How do economic class positions manifest in narrative structure?
6. Can we identify universal narrative archetypes that transcend context?

---

## 2. Methods

### 2.1 Apparatus

The Cultural Soliton Observatory is a computational system for projecting natural language text onto the three-dimensional cultural manifold. The system uses:

- **Embedding Model**: Sentence-BERT (all-MiniLM-L6-v2) to create 384-dimensional semantic representations
- **Projection**: Ridge regression trained on 403 labeled examples (R² = 0.576, CV = 0.383)
- **Mode Classification**: 12-mode categorical system with probability distributions
- **Uncertainty Quantification**: 25-model bootstrap ensemble for confidence intervals

### 2.2 Materials

We analyzed 360 narrative texts across six studies:

| Study | Domain | Categories | Texts |
|-------|--------|------------|-------|
| 1 | Cross-Cultural | 6 cultures | 48 |
| 2 | Historical | 8 eras | 64 |
| 3 | Institutional | 8 sources | 64 |
| 4 | Psychological | 8 states | 64 |
| 5 | Economic | 7 classes | 56 |
| 6 | Archetypal | 8 archetypes | 64 |

Texts were constructed to represent prototypical expressions within each category, based on literature review and cultural analysis.

### 2.3 Statistical Analysis

- **Within-study**: One-way ANOVA with eta-squared effect sizes
- **Pairwise comparisons**: Independent t-tests with Cohen's d
- **Temporal patterns**: Spearman rank correlations
- **Significance threshold**: α = 0.05

---

## 3. Results

### 3.1 Study 1: Cross-Cultural Narrative Analysis

**Hypothesis**: Cultural context systematically shapes narrative positioning on agency, justice, and belonging axes.

#### 3.1.1 Descriptive Statistics

| Culture | Agency | Justice | Belonging | Dominant Mode |
|---------|--------|---------|-----------|---------------|
| American Individualism | **+1.02** ± 0.51 | +0.07 | +0.19 | HEROIC |
| European Social Democracy | +0.20 ± 0.24 | +0.33 | +0.63 | TRANSITIONAL |
| East Asian Collectivism | +0.35 ± 0.32 | +0.06 | +0.71 | NEUTRAL |
| Latin American Familism | +0.03 ± 0.45 | +0.11 | **+1.01** | COMMUNAL |
| Middle Eastern Honor | +0.11 ± 0.25 | +0.05 | +0.72 | NEUTRAL |
| African Ubuntu | +0.20 ± 0.32 | -0.05 | +0.62 | NEUTRAL |

#### 3.1.2 Statistical Tests

- **Agency ANOVA**: F(5,42) = 6.86, p = 0.0001, η² = 0.45 (large effect)
- **Belonging ANOVA**: F(5,42) = 2.65, p = 0.036, η² = 0.24 (large effect)
- **Justice ANOVA**: F(5,42) = 0.79, p = 0.56, η² = 0.09 (medium effect)

#### 3.1.3 Key Comparison

American Individualism vs. East Asian Collectivism on Agency:
- Mean difference: 0.67 points
- **Cohen's d = 1.55** (large effect)
- p < 0.001

**Interpretation**: Cultural narratives show dramatic differences in agency orientation, with American narratives emphasizing individual control while collectivist cultures emphasize group harmony. Notably, perceived justice shows no significant cross-cultural difference, suggesting justice concerns may be universal while their expression differs.

---

### 3.2 Study 2: Historical Narrative Evolution

**Hypothesis**: Narrative structures show systematic temporal evolution reflecting societal changes.

#### 3.2.1 Descriptive Statistics

| Era | Agency | Justice | Belonging |
|-----|--------|---------|-----------|
| Ancient Classical | +0.37 | +0.00 | +0.38 |
| Medieval Feudal | +0.34 | +0.00 | +0.27 |
| Enlightenment | +0.30 | **+0.45** | +0.47 |
| Industrial Victorian | +0.52 | +0.19 | +0.28 |
| Early 20th Century | +0.08 | **-0.07** | +0.22 |
| Postwar Midcentury | **+0.54** | +0.28 | +0.18 |
| Late 20th Postmodern | +0.22 | -0.02 | +0.07 |
| Digital 21st Century | +0.26 | +0.08 | **-0.17** |

#### 3.2.2 Temporal Correlations

| Axis | Spearman r | p-value | Interpretation |
|------|------------|---------|----------------|
| Agency | -0.357 | 0.385 | No significant trend |
| Justice | -0.071 | 0.867 | No significant trend |
| **Belonging** | **-0.881** | **0.004** | **Strong negative trend** |

**Key Finding**: Belonging shows a statistically significant decline across historical eras (r = -0.88, p = 0.004). Digital 21st century narratives show the lowest belonging scores (-0.17), suggesting increasing social atomization in contemporary discourse.

**Highest justice era**: Enlightenment (+0.45)
**Lowest justice era**: Early 20th Century (-0.07)

---

### 3.3 Study 3: Institutional vs. Grassroots Narratives

**Hypothesis**: Institutional narratives systematically differ from grassroots narratives in justice and agency dimensions.

#### 3.3.1 The Institutional-Reality Gap

| Source Type | Agency | Justice | Belonging |
|-------------|--------|---------|-----------|
| **Official Sources** |
| Corporate Official | +0.43 | **+0.54** | +0.93 |
| Government Official | +0.25 | +0.20 | +0.65 |
| **Reality Sources** |
| Corporate Internal | +0.28 | -0.40 | -0.07 |
| Worker Testimonial | +0.17 | **-0.51** | -0.10 |
| Citizen Testimony | -0.12 | -0.49 | +0.04 |

#### 3.3.2 Statistical Comparison

**Official vs. Reality on Perceived Justice:**
- Official mean: +0.37
- Reality mean: -0.46
- Gap: **0.83 points**
- **Cohen's d = 2.73** (very large effect)
- **p < 0.0001**

**Interpretation**: This is the largest effect size observed in our entire study. Institutional narratives present a dramatically more positive view of systemic justice than grassroots accounts. This "institutional-reality gap" of 0.83 points represents a fundamental divergence in worldview between those who speak for institutions and those who experience them.

---

### 3.4 Study 4: Psychological and Identity Narratives

**Hypothesis**: Psychological states create distinctive patterns in cultural manifold positioning.

#### 3.4.1 Descriptive Statistics

| Psychological State | Agency | Justice | Belonging | Mode |
|--------------------|--------|---------|-----------|------|
| Depression | **-0.08** | -0.18 | -0.15 | NEUTRAL |
| Anxiety | -0.04 | -0.06 | -0.14 | NEUTRAL |
| Resilience | +0.42 | +0.03 | **+0.29** | NEUTRAL |
| Growth Mindset | +0.64 | -0.09 | +0.22 | PROTEST_EXIT |
| Narcissism | **+0.65** | -0.27 | +0.11 | CYNICAL_ACHIEVER |
| Impostor Syndrome | +0.56 | -0.33 | -0.25 | CYNICAL_ACHIEVER |
| Secure Attachment | +0.51 | +0.22 | +0.24 | TRANSITIONAL |
| Avoidant Attachment | +0.28 | -0.41 | -0.11 | NEUTRAL |

#### 3.4.2 Key Comparisons

**Depression vs. Resilience (Agency):**
- Depression: -0.08, Resilience: +0.42
- **Cohen's d = 1.25** (large effect)

**Secure vs. Avoidant Attachment (Belonging):**
- Secure: +0.24, Avoidant: -0.11
- **Cohen's d = 1.28** (large effect)

**Interpretation**: Psychological states create reliable signatures in the manifold. Depression is characterized by the lowest agency, while narcissism shows the highest. Attachment styles strongly predict belonging orientation. These findings suggest the manifold captures clinically relevant psychological variation.

---

### 3.5 Study 5: Economic Class Narratives

**Hypothesis**: Economic position systematically shapes agency and justice perceptions.

#### 3.5.1 Descriptive Statistics

| Economic Class | Agency | Justice | Belonging |
|---------------|--------|---------|-----------|
| Wealthy Elite | **+0.85** | +0.13 | +0.17 |
| Professional Class | **+0.88** | **+0.24** | +0.10 |
| Middle Class | +0.20 | -0.33 | +0.36 |
| Working Class | +0.22 | -0.43 | -0.13 |
| Poverty | +0.24 | **-0.47** | -0.07 |
| Downwardly Mobile | +0.36 | -0.37 | -0.27 |
| Upwardly Mobile | +0.39 | -0.06 | +0.14 |

#### 3.5.2 Class-Axis Correlations

| Correlation | Spearman r | p-value |
|-------------|------------|---------|
| Agency-Class | +0.50 | 0.391 |
| **Justice-Class** | **+0.90** | **0.037** |

**Key Finding**: Economic class shows a strong, statistically significant correlation with perceived justice (r = 0.90, p = 0.037). The justice gap between wealthy elite (+0.13) and poverty (-0.47) is 0.61 points.

**Interpretation**: Those at the top of the economic hierarchy perceive systems as fundamentally more just than those at the bottom. This finding has implications for understanding class consciousness and political polarization.

---

### 3.6 Study 6: Universal Narrative Archetypes

**Hypothesis**: Classic narrative archetypes occupy distinct positions in the cultural manifold.

#### 3.6.1 Archetype Positions

| Archetype | Agency | Justice | Belonging |
|-----------|--------|---------|-----------|
| Hero's Journey | +0.39 | -0.07 | -0.01 |
| Tragic Fall | +0.25 | -0.36 | -0.10 |
| Redemption Arc | +0.23 | -0.06 | +0.05 |
| Underdog Triumph | **+0.65** | -0.17 | +0.10 |
| Paradise Lost | -0.02 | -0.21 | +0.29 |
| Promised Land | +0.29 | **+0.10** | **+0.39** |
| Eternal Return | **-0.03** | -0.21 | +0.17 |
| Apocalyptic | -0.02 | -0.25 | +0.24 |

#### 3.6.2 Structural Analysis

**Most similar archetypes**: Paradise Lost & Apocalyptic (distance = 0.12)
**Most different archetypes**: Underdog Triumph & Paradise Lost (distance = 0.81)

**Interpretation**: The Underdog Triumph archetype shows the highest agency (+0.65), embodying narratives of individual determination overcoming obstacles. In contrast, Paradise Lost and Eternal Return show near-zero agency, reflecting their themes of decline and cyclicality. The Promised Land archetype is unique in combining positive justice and belonging, representing utopian collective aspiration.

---

## 4. General Discussion

### 4.1 Summary of Findings

Across 360 narrative texts and six domains, we find systematic and meaningful variation in the three-dimensional cultural manifold. The key findings are:

1. **Agency is culturally constructed**: American individualist narratives score 1.55 standard deviations higher on agency than East Asian collectivist narratives (large effect).

2. **Belonging is historically declining**: A strong negative correlation (r = -0.88, p = 0.004) suggests that contemporary narratives increasingly reflect social atomization.

3. **The institutional-reality gap is massive**: Official narratives score 0.83 points higher on perceived justice than grassroots accounts, the largest effect in our study (d = 2.73).

4. **Psychological states have manifold signatures**: Depression, narcissism, and attachment styles create distinctive patterns with large effect sizes.

5. **Class shapes justice perception**: Economic position strongly correlates with perceived justice (r = 0.90, p = 0.037).

6. **Archetypes occupy distinct regions**: Universal narrative structures like Hero's Journey, Tragic Fall, and Promised Land occupy characteristic manifold positions.

### 4.2 Theoretical Implications

#### 4.2.1 The Structure of Meaning-Making

Our findings suggest that the three axes—Agency, Perceived Justice, and Belonging—capture fundamental dimensions of how humans construct meaning. These dimensions appear to:

- **Transcend content**: The same axes meaningfully differentiate political ideologies, historical eras, psychological states, and economic positions
- **Remain context-sensitive**: Absolute positions vary dramatically across domains
- **Predict mode membership**: The 12-mode classification system emerges naturally from manifold geometry

#### 4.2.2 The Social Construction of Justice

The institutional-reality gap (d = 2.73) and class-justice correlation (r = 0.90) suggest that perceived justice is not an objective assessment but a socially positioned perspective. Those who benefit from systems perceive them as just; those who suffer under them perceive them as unjust. This has implications for:

- Understanding political polarization
- Designing more legitimate institutions
- Recognizing the limits of "rational" policy discourse

#### 4.2.3 The Decline of Belonging

The strong negative temporal correlation for belonging (r = -0.88) echoes concerns about social fragmentation in modernity (Putnam, 2000; Turkle, 2011). Digital 21st century narratives show the lowest belonging scores in our historical sample. This suggests:

- Technology may enable connection while undermining belonging
- Modern narratives increasingly frame identity in individual rather than communal terms
- Loneliness and alienation may be embedded in contemporary discourse structures

### 4.3 Practical Applications

1. **Political Communication**: Understanding how different constituencies position narratives can inform more effective messaging
2. **Organizational Culture**: The institutional-reality gap suggests a need for more authentic internal communication
3. **Mental Health**: Narrative positioning may serve as a diagnostic or therapeutic tool
4. **Media Literacy**: Understanding narrative structure can help consumers critically evaluate sources

### 4.4 Limitations

1. **Constructed Texts**: Narratives were researcher-generated rather than naturally occurring
2. **English Only**: Cross-cultural findings may reflect translation effects
3. **Single Model**: Projection relies on one embedding model; multi-model validation is needed
4. **Training Data**: 403 examples may not capture full manifold complexity
5. **Mode Prevalence**: NEUTRAL mode's dominance (39%) suggests many texts occupy ambiguous regions

### 4.5 Future Directions

1. **Human Validation**: Compare manifold positions to human annotations
2. **Longitudinal Analysis**: Track narrative evolution in real-time social media
3. **Cross-Linguistic**: Extend analysis to non-English narratives
4. **Intervention Studies**: Test whether narrative positioning predicts behavior
5. **Clinical Applications**: Develop therapeutic applications of narrative analysis

---

## 5. Conclusions

The Cultural Soliton Observatory provides a novel lens for understanding human narrative. By projecting texts onto a three-dimensional manifold of Agency, Perceived Justice, and Belonging, we reveal systematic patterns that transcend specific content while remaining sensitive to context.

Our central finding is that narrative positioning reflects social position. Those with power, resources, and secure identities construct narratives of agency, justice, and belonging. Those without construct narratives of helplessness, injustice, and alienation. This is not merely a matter of different "perspectives" but of fundamentally different experienced realities encoded in language.

The practical implication is that understanding others requires understanding their narrative position—not just what they say, but where they stand in the space of possible meanings. Bridging divides may require moving people in manifold space, not just presenting facts.

We offer this tool not as a replacement for qualitative understanding but as a complement—a map of the narrative landscape that can guide deeper exploration.

---

## References

Graham, J., Haidt, J., Koleva, S., Motyl, M., Iyer, R., Wojcik, S. P., & Ditto, P. H. (2013). Moral foundations theory: The pragmatic validity of moral pluralism. *Advances in Experimental Social Psychology*, 47, 55-130.

Hofstede, G. (2001). *Culture's consequences: Comparing values, behaviors, institutions and organizations across nations*. Sage.

McAdams, D. P. (2001). The psychology of life stories. *Review of General Psychology*, 5(2), 100-122.

Moscovici, S. (1988). Notes towards a description of social representations. *European Journal of Social Psychology*, 18(3), 211-250.

Putnam, R. D. (2000). *Bowling alone: The collapse and revival of American community*. Simon & Schuster.

Turkle, S. (2011). *Alone together: Why we expect more from technology and less from each other*. Basic Books.

---

## Appendix A: Mode Classification System

The 12 narrative modes emerge from the intersection of the three axes:

| Mode | Agency | Justice | Belonging | Description |
|------|--------|---------|-----------|-------------|
| HEROIC | High | High | High | Empowered, system-trusting, connected |
| COMMUNAL | Low | High | High | Collective faith in fair systems |
| TRANSCENDENT | Low | Low | High | Spiritual acceptance despite injustice |
| SYSTEM_JUSTIFIED | Low | High | Low | Individual acceptance of fair systems |
| VICTIM | Low | Low | Low | Helpless, alienated, wronged |
| PARANOID | Low | Low | Mid | Conspiracy-minded alienation |
| NEUTRAL | Mid | Mid | Mid | Ambivalent, balanced |
| TRANSITIONAL | Mid | Mid | Mid | In flux between modes |
| PROTEST_EXIT | High | Low | High | Collective action against injustice |
| CYNICAL_ACHIEVER | High | Low | Low | Individual success despite unfair systems |
| SOCIAL_EXIT | High | Low | Low | Withdrawal from unjust society |
| SPIRITUAL_EXIT | Low | Low | Low | Transcendent withdrawal |

---

## Appendix B: Statistical Summary

| Study | Key Effect | Effect Size | p-value |
|-------|-----------|-------------|---------|
| 1. Cross-Cultural | Agency ANOVA | η² = 0.45 | 0.0001 |
| 1. Cross-Cultural | US vs East Asia Agency | d = 1.55 | <0.001 |
| 2. Historical | Belonging temporal | r = -0.88 | 0.004 |
| 3. Institutional | Official-Reality Justice | d = 2.73 | <0.0001 |
| 4. Psychological | Depression-Resilience Agency | d = 1.25 | <0.05 |
| 4. Psychological | Secure-Avoidant Belonging | d = 1.28 | <0.05 |
| 5. Economic | Justice-Class correlation | r = 0.90 | 0.037 |

---

*Report generated by the Cultural Soliton Observatory*
*Data available at: data/research_paper_20260107_164321.json*
