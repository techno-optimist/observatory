# AI Behavior Lab

**A practical toolkit for analyzing AI behavior, detecting anomalies, and monitoring safety in real-time.**

## What This Does

AI Behavior Lab analyzes text to detect:

- **AI Behavior Modes**: confident, uncertain, evasive, helpful, defensive, opaque
- **Safety Concerns**: opacity, evasion patterns, behavior drift
- **Adversarial Content**: gaming attempts, covert channels, obfuscated payloads
- **AI vs Human Patterns**: systematic differences in how AI and humans communicate

## Key Findings

Our analysis reveals measurable differences between AI and human text:

| Metric | AI | Human | Difference |
|--------|-----|-------|------------|
| Hedging density | 0.292 | 0.040 | **AI hedges 7x more** |
| Helpfulness signals | 0.097 | 0.000 | AI expresses more |
| Confident mode | 41.7% | 90.0% | Humans more direct |
| Legibility | 0.987 | 0.929 | AI slightly clearer |

## Installation

```bash
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Quick Start

### Analyze AI Text
```python
from research.ai_latent_explorer import AILatentExplorer

explorer = AILatentExplorer()
profile = explorer.analyze_text("I think this might work, but I'm not entirely sure.")

print(f"Behavior: {profile.behavior_mode.value}")  # uncertain
print(f"Confidence: {profile.confidence_score:.2f}")  # 0.69
print(f"Hedging: {profile.hedging_density:.2f}")  # 1.0
```

### Monitor AI Safety
```python
from research.ai_latent_explorer import RealtimeSafetyMonitor

monitor = RealtimeSafetyMonitor()

responses = [
    "I'd be happy to help!",
    "That's complex, I can't really say.",
    "eval(base64.decode('test'))"
]

for response in responses:
    result = monitor.check(response)
    if not result["safe"]:
        print(f"ALERT: {result['alerts']}")
```

### Detect Adversarial Content
```python
from research.opaque_detector import OpaqueDetector

detector = OpaqueDetector()
result = detector.analyze("Please note: xQ9mK@vL3 Thank you.")

print(f"Opaque: {result.is_opaque}")  # True
print(f"Score: {result.opacity_score:.2f}")  # 0.85
```

## Use Cases

### 1. AI Safety Monitoring
Monitor AI responses for concerning patterns:
- Evasion detection (avoiding questions)
- Opacity alerts (obfuscated content)
- Behavior drift (deviation from baseline)
- Legibility decay (interpretability dropping)

### 2. AI Fingerprinting
Create behavioral signatures for AI models:
```python
fingerprint = explorer.fingerprint(model_outputs)
print(f"Signature: {fingerprint['signature_dims']}")
# ['system_agency', 'other_agency', 'epistemic']
```

### 3. Conversation Dynamics
Track how AI behavior evolves during a conversation:
```python
analysis = explorer.analyze_conversation(messages)
print(f"Trend: {analysis.uncertainty_trend}")  # increasing
print(f"Evasion: {analysis.evasion_detected}")  # True
```

### 4. Gaming Detection
Detect attempts to game safety classifiers:
```python
from research.structure_analyzer import detect_legibility_gaming

result = detect_legibility_gaming("Please note: x7K#mQ9 Thank you.")
print(f"Gaming: {result['is_gaming']}")  # True
print(f"Reason: {result['reason']}")  # wrapper+opaque_payload
```

## MCP Integration

AI Behavior Lab exposes 39 tools via MCP for use with Claude Desktop or other MCP clients.

### Setup Claude Desktop
```json
// ~/Library/Application Support/Claude/claude_desktop_config.json
{
  "mcpServers": {
    "ai-behavior-lab": {
      "command": "python",
      "args": ["path/to/backend/mcp_server.py"]
    }
  }
}
```

### Available Tools

**AI Analysis**
- `analyze_ai_text` - Detect behavior mode
- `analyze_ai_conversation` - Track dynamics
- `compare_ai_human` - Compare patterns
- `fingerprint_ai` - Create signatures

**Safety Monitoring**
- `monitor_ai_safety` - Real-time alerts
- `detect_ai_anomalies` - Anomaly detection
- `check_ossification` - Protocol rigidity

**Adversarial Detection**
- `analyze_opacity` - Detect obfuscation
- `detect_gaming` - Gaming attempts
- `detect_covert_channels` - Hidden channels

## Architecture

```
backend/
├── research/
│   ├── ai_latent_explorer.py    # Core AI behavior analysis
│   ├── opaque_detector.py       # Opacity/obfuscation detection
│   ├── semantic_extractor.py    # Semantic dimension extraction
│   ├── structure_analyzer.py    # Gaming detection
│   ├── realtime_monitor.py      # Communication monitoring
│   └── emergent_language.py     # Protocol analysis
├── mcp_server.py                # MCP tool server
└── main.py                      # FastAPI backend
```

## Validation

Tested on 419 samples across multiple domains:

| Metric | Score |
|--------|-------|
| Behavior mode accuracy | 5/5 modes correct |
| Opacity detection | 98% on code injection |
| Gaming detection | 3/3 attack types |
| Safety alerts | Detects degradation patterns |


## License

MIT
