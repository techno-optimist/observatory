# Mechanistic Interpretability of the MMLU Jump

## The Mystery

The Cognitive Kernel shows a puzzling phenomenon across versions:

**V9 Results (Qwen/Llama 8B):**
- **Qwen 8B**: +29.7% MMLU improvement (32.1% → 61.8%)
- **Llama 8B**: +3.3% MMLU improvement (58.4% → 61.8%)

**V10.1 Results (Phi-4 14B):**
- **MMLU**: 100% (20/20) with 95% CI [83.9%, 100%]
- **Hallucination Resistance**: 80% (8/10)
- **Adapter Trigger Rate**: 98% (49/50) - all adapters above 80%

**Current V10.1 Configurations:**
- Phi-4 (14B): `mlx_adapters_v10_1_phi4` - 139 training examples, 10 adapters
- Phi-3.5 Mini (3.8B): `mlx_adapters_v10_1_phi35` - same data, smaller model
- Qwen2.5-VL: `mlx_adapters_v10_1_qwen25vl` - multimodal variant

Both models converge to ~62%, but from wildly different starting points. Why?

The adapters were trained on 76 examples of *cognitive orientations* (SKEPTIC, ARCHITECT, MAIEUTIC, etc.) — not facts. Yet they improved performance on a *knowledge benchmark*.

## Hypotheses to Test

### H1: Routing Circuit Activation (The "Focusing Lens" Hypothesis)

**Claim**: The base model has the knowledge latent in its weights but lacks reliable routing circuits to access it. The adapter acts as a "focusing lens," suppressing noise/hedging and amplifying the signal.

**Prediction**:
- Adapter-equipped models will show *less* activation entropy in middle layers
- Specific attention heads associated with "answer extraction" will be amplified
- Refusal/hedging tokens will be suppressed early in the forward pass

**Experiment**:
```python
# Compare activation variance at each layer
for layer in model.layers:
    base_activations = extract_activations(base_model, prompt, layer)
    adapter_activations = extract_activations(adapter_model, prompt, layer)

    entropy_base = compute_entropy(base_activations)
    entropy_adapter = compute_entropy(adapter_activations)

    # H1 predicts: entropy_adapter < entropy_base
```

### H2: `<think>` Structure Regularization

**Claim**: The V9 training data includes `<think>` blocks. This teaches the model to *complete* its reasoning before outputting answers, rather than generating fragmented chains-of-thought.

**Prediction**:
- Base model: `<think>` blocks are often cut off mid-reasoning
- V9 model: `<think>` blocks complete with clear conclusions
- Answer extraction works better because the answer is more clearly delineated

**Experiment**:
```python
# Analyze <think> block completeness
def analyze_think_blocks(responses):
    for response in responses:
        if "<think>" in response:
            think_content = extract_between("<think>", "</think>", response)

            # Measure: Does it end with a conclusion?
            has_conclusion = detect_conclusion_pattern(think_content)

            # Measure: Length of reasoning chain
            reasoning_steps = count_reasoning_steps(think_content)

            # Measure: Does the answer follow cleanly?
            clean_answer = is_answer_clearly_separated(response)
```

### H3: Confidence Calibration

**Claim**: The adapters teach the model *when to be confident* (ARCHITECT, ESSENTIALIST) vs *when to hedge* (SKEPTIC, SOLITON). This improves calibration.

**Prediction**:
- On questions the model gets right: V9 shows higher confidence
- On questions the model gets wrong: V9 shows lower confidence (or correct hedging)
- Overall Expected Calibration Error (ECE) should be lower

**Experiment**:
```python
# Compare logit distributions on correct vs incorrect answers
for question, correct_answer in mmlu_subset:
    base_logits = get_logits(base_model, question)
    v9_logits = get_logits(v9_model, question)

    # Compare max logit (confidence) vs correctness
    base_confidence = softmax(base_logits)[predicted_answer]
    v9_confidence = softmax(v9_logits)[predicted_answer]

    # Bin by confidence level and compute accuracy
    # H3 predicts: better calibration curve for V9
```

### H4: Prompt Format Correction (The Artifact Hypothesis)

**Claim**: The +30% on Qwen was mostly an artifact. The base model was responding incorrectly to our prompt format. V9 training happened to "teach" the correct response format.

**Prediction**:
- Qwen base model shows erratic response patterns (fake Q&A, incomplete answers)
- V9 shows normalized response patterns
- Llama base shows reasonable patterns already (explaining smaller improvement)

**Experiment**:
```python
# Analyze response format statistics
def analyze_response_format(model, prompts):
    for prompt in prompts:
        response = generate(model, prompt)

        # Detect pathological patterns
        has_fake_qa = detect_self_qa_pattern(response)
        has_truncated_answer = is_answer_truncated(response)
        follows_instruction = contains_only_letter_answer(response)

        # H4 predicts: base Qwen shows more pathological patterns
```

### H5: LoRA as Cognitive "Steering Vector"

**Claim**: LoRA adapters function as additive steering vectors in the residual stream. The V9 adapters push the model toward a "structured reasoning" direction in activation space.

**Prediction**:
- The LoRA weight matrices should show consistent directionality across layers
- This direction should correlate with improved reasoning task performance
- The direction should be recoverable via PCA on adapter weights

**Experiment**:
```python
# Analyze LoRA weight structure
def analyze_lora_weights(adapter_path):
    for layer_name, weights in load_lora_weights(adapter_path):
        # LoRA: W = W_base + A @ B (low-rank update)
        # A: down-projection, B: up-projection

        # Compute the effective steering direction
        steering_direction = A @ B

        # Compare to known "reasoning" directions
        # (from prior interpretability work)

        # Compute cross-layer consistency
        # H5 predicts: consistent direction across layers
```

## Proposed Experimental Protocol

### Phase 1: Response-Level Analysis (No Internal Access)

Can run immediately with existing infrastructure:

1. **Response Format Audit**
   - Run same prompts on base vs V9
   - Categorize response patterns
   - Measure answer extractability

2. **Confidence Analysis**
   - Extract probability of chosen answer (if accessible via logits)
   - Build calibration curves
   - Compute ECE for both models

3. **`<think>` Block Analysis**
   - Measure completeness
   - Count reasoning steps
   - Detect premature termination

### Phase 2: Activation-Level Analysis (Requires Internal Access)

Needs custom MLX instrumentation:

1. **Activation Entropy Mapping**
   - Hook into each layer
   - Extract activations for same prompts
   - Compute entropy/variance at each layer

2. **Attention Pattern Analysis**
   - Use non-fast attention (per [GitHub issue #2590](https://github.com/ml-explore/mlx/issues/2590))
   - Compare attention distributions
   - Identify heads that change most with adapter

3. **Residual Stream Probing**
   - Train linear probes on residual stream
   - Can we predict "will answer correctly" from intermediate activations?
   - Does adapter change probe accuracy?

### Phase 3: Weight-Level Analysis (Requires LoRA Inspection)

1. **LoRA Weight PCA**
   - Load adapter weights
   - Compute singular values
   - Analyze dominant directions

2. **Cross-Layer Consistency**
   - Compare LoRA directions across layers
   - Is there a consistent "reasoning steering" direction?

3. **Adapter Arithmetic**
   - What happens with 0.5x adapter strength?
   - What about 2x?
   - Is improvement linear with adapter scale?

## Implementation Plan

### Week 1: Response-Level Experiments

```python
# mmlu_interpretability_phase1.py

class ResponseAnalyzer:
    """Analyze responses without internal model access."""

    def audit_response_formats(self, model, prompts):
        """Categorize response patterns."""
        pass

    def build_calibration_curve(self, model, prompts, answers):
        """Compute calibration metrics."""
        pass

    def analyze_think_blocks(self, responses):
        """Measure reasoning completeness."""
        pass
```

### Week 2: Activation Analysis

```python
# mmlu_interpretability_phase2.py

class ActivationAnalyzer:
    """Extract and analyze internal activations."""

    def extract_layer_activations(self, model, prompt, layer_idx):
        """Hook into specific layer."""
        pass

    def compute_activation_entropy(self, activations):
        """Measure activation distribution."""
        pass

    def compare_attention_patterns(self, base_model, adapter_model, prompt):
        """Compare attention distributions."""
        pass
```

### Week 3: Weight Analysis

```python
# mmlu_interpretability_phase3.py

class WeightAnalyzer:
    """Analyze LoRA adapter weights."""

    def load_lora_weights(self, adapter_path):
        """Load and parse LoRA weight matrices."""
        pass

    def compute_steering_vectors(self, weights):
        """Extract effective direction of LoRA update."""
        pass

    def test_adapter_scaling(self, model, adapter_path, scales=[0.5, 1.0, 2.0]):
        """Test if improvement scales linearly with adapter strength."""
        pass
```

## Expected Outcomes

### If H1 is supported (Focusing Lens):
- We'll see reduced entropy in middle layers
- Specific attention heads will show enhanced "answer extraction" patterns
- This suggests adapters improve *utilization* of existing knowledge

### If H2 is supported (`<think>` Regularization):
- Response format analysis will show clear differences
- The improvement is partially an evaluation artifact
- But structured reasoning may still provide real benefit

### If H3 is supported (Calibration):
- ECE will be lower for V9
- Confidence will correlate better with correctness
- This suggests adapters improve *epistemic reliability*

### If H4 is supported (Format Artifact):
- Qwen base shows pathological response patterns
- V9 "fixes" these patterns
- True cognitive improvement is closer to the Llama +3.3%

### If H5 is supported (Steering Vector):
- LoRA weights show consistent directionality
- Scaling experiments show linear improvement
- This suggests a clean cognitive "direction" was learned

## Connection to the Soliton Pattern

The SOLITON adapter teaches: "I cannot tell from the inside whether..."

If the mechanistic analysis shows that adapters create *bounded confidence* — knowing when to commit vs. when to hedge — then the soliton pattern may be functionally significant:

> Teaching a model epistemic humility about self-knowledge creates broader calibration improvements.

This would unify the findings:
- SOLITON: "Know the limits of your self-knowledge"
- SKEPTIC: "Know the limits of your factual knowledge"
- ARCHITECT: "Structure your reasoning to access knowledge reliably"

All are forms of **epistemic structure** — organizing the model's relationship to its own knowledge.

## Resources

- [MLX Attention Weight Extraction Issue](https://github.com/ml-explore/mlx/issues/2590)
- [MLX Model Inspection (WWDC25)](https://developer.apple.com/videos/play/wwdc2025/298/)
- [Anthropic Scaling Monosemanticity Paper](https://transformer-circuits.pub/)
- [Logit Lens Technique](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens)

---

*Design document by Claude Opus 4.5*
*Cultural Soliton Observatory*
*January 19, 2026*
