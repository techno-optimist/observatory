# Cultural Soliton Observatory MCP Server - API Reference

> **Version**: 3.0
> **Last Updated**: January 2026
> **Server Name**: `cultural-soliton-observatory`

The Cultural Soliton Observatory MCP Server exposes tools for analyzing cultural narratives in embedding space. It projects text onto a 3D manifold with axes for Agency, Perceived Justice, and Belonging, enabling detection of narrative modes, hypocrisy gaps, and cultural dynamics.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Configuration](#configuration)
3. [Core Concepts](#core-concepts)
4. [Tool Reference](#tool-reference)
   - [Core Analysis](#core-analysis)
   - [Projection Mode Management (v2)](#projection-mode-management-v2)
   - [Comparative Analysis (v2)](#comparative-analysis-v2)
   - [Alert System (v2)](#alert-system-v2)
   - [Advanced Analytics (v2)](#advanced-analytics-v2)
   - [Force Field Analysis (v2)](#force-field-analysis-v2)
   - [High-Level Narrative Analysis](#high-level-narrative-analysis)
   - [Advanced Research (v3)](#advanced-research-v3)
5. [Example Workflows](#example-workflows)
6. [Troubleshooting](#troubleshooting)

---

## Quick Start

### Prerequisites

1. **Backend Server**: Ensure the Cultural Soliton Observatory backend is running:
   ```bash
   cd backend
   python -m uvicorn main:app --host 127.0.0.1 --port 8000
   ```

2. **MCP Server**: Start the MCP server:
   ```bash
   cd mcp-server
   python server.py
   ```

3. **MCP Inspector** (optional for debugging):
   ```bash
   npx @modelcontextprotocol/inspector python server.py
   ```

### Connecting via Claude Desktop

Add to your Claude Desktop configuration (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "cultural-soliton-observatory": {
      "command": "python",
      "args": ["/path/to/mcp-server/server.py"],
      "env": {
        "OBSERVATORY_BACKEND_URL": "http://127.0.0.1:8000"
      }
    }
  }
}
```

### Basic Usage Example

```
User: Analyze this text: "We're building something amazing together.
      Everyone's voice matters and our team is unstoppable."

Claude: [calls project_text tool]

Result:
- Agency: 1.42 (Strong empowerment)
- Perceived Justice: 0.85 (Trust in fair treatment)
- Belonging: 1.67 (Deep community connection)
- Mode: GROWTH_MINDSET (Positive)
```

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OBSERVATORY_BACKEND_URL` | `http://127.0.0.1:8000` | URL of the observatory backend API |
| `OBSERVATORY_DEFAULT_MODEL` | `all-MiniLM-L6-v2` | Default embedding model for projections |

---

## Core Concepts

### The Cultural Manifold

The observatory projects text onto a 3D space defined by three axes:

| Axis | Range | Low (-2) | High (+2) |
|------|-------|----------|-----------|
| **Agency** | -2 to +2 | Fatalism, powerlessness | Self-determination, empowerment |
| **Perceived Justice** | -2 to +2 | Corruption, unfairness | System legitimacy, fair treatment |
| **Belonging** | -2 to +2 | Alienation, isolation | Community connection, embedding |

### Narrative Modes

Texts are classified into 12 modes organized in 4 categories:

| Category | Modes | Characteristics |
|----------|-------|-----------------|
| **POSITIVE** | Growth Mindset, Civic Idealism, Faithful Zeal | High agency + High justice |
| **SHADOW** | Cynical Burnout, Institutional Decay, Schismatic Doubt | High agency + Low justice |
| **EXIT** | Quiet Quitting, Grid Exit, Apostasy | Low agency, withdrawal |
| **AMBIVALENT** | Conflicted, Transitional, Neutral | Mixed signals |

### Force Field System

Beyond position, narratives have motivational forces:

**Attractor Targets** (what narratives are drawn toward):
- AUTONOMY, COMMUNITY, JUSTICE, MEANING, SECURITY, RECOGNITION

**Detractor Sources** (what narratives flee from):
- OPPRESSION, ISOLATION, INJUSTICE, MEANINGLESSNESS, INSTABILITY, INVISIBILITY

**Force Quadrants**:
- ACTIVE_TRANSFORMATION: High attractor + high detractor
- PURE_ASPIRATION: High attractor + low detractor
- PURE_ESCAPE: Low attractor + high detractor
- STASIS: Low attractor + low detractor

---

## Tool Reference

---

### Core Analysis

#### `project_text`

Project a text onto the 3D cultural manifold.

**Description**: Returns coordinates (agency, perceived_justice, belonging) representing where the text lands in cultural-semantic space, along with its narrative mode classification and confidence. Uses the v2 API which returns enhanced 12-mode classification with soft labels, stability indicators, and primary/secondary modes with probabilities.

**Parameters**:

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `text` | string | Yes | The text to project onto the manifold |
| `model_id` | string | No | Embedding model to use (default: all-MiniLM-L6-v2) |
| `include_soft_labels` | boolean | No | Include probability distribution across all modes (default: false) |

**Returns**:
- Coordinates (agency, perceived_justice, belonging)
- Primary and secondary mode classifications
- Confidence score
- Stability score and boundary case detection
- Soft labels (if requested)

**Example**:
```json
{
  "text": "I refuse to accept the status quo. We can and will change things for the better.",
  "include_soft_labels": true
}
```

---

#### `analyze_corpus`

Analyze a collection of texts with clustering detection.

**Description**: Projects all texts onto the manifold and identifies narrative clusters (stable groupings). Returns per-text coordinates, detected clusters with their modes and centroids, and overall mode distribution.

**Parameters**:

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `texts` | array[string] | Yes | List of texts to analyze |
| `detect_clusters` | boolean | No | Whether to detect narrative clusters (default: true) |

**Returns**:
- Total texts analyzed
- Mode distribution (count per mode)
- Detected clusters with:
  - Cluster ID and mode
  - Size and stability
  - Centroid coordinates
  - Exemplar texts

**Example**:
```json
{
  "texts": [
    "We're crushing it this quarter!",
    "Management doesn't listen to us",
    "Just doing what I'm told..."
  ],
  "detect_clusters": true
}
```

---

#### `compare_projections`

Compare how a text is projected by different methods.

**Description**: Tests the same text against multiple projection methods (ridge, GP, neural) to understand projection stability and uncertainty. High variance across methods suggests the text is ambiguous or at a boundary between modes.

**Parameters**:

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `text` | string | Yes | Text to compare projections for |
| `methods` | array[string] | No | Projection methods to compare: "ridge", "gp", "neural" (default: all) |

**Returns**:
- Per-method coordinates and mode classifications
- Variance analysis across methods

**Example**:
```json
{
  "text": "Things are changing around here, not sure if for better or worse",
  "methods": ["ridge", "gp"]
}
```

---

#### `generate_probe`

Generate a text predicted to land at specific coordinates.

**Description**: Given target (agency, perceived_justice, belonging) coordinates, provides guidance for generating a probe text that should project near those coordinates. Useful for testing projection boundaries and understanding what narratives occupy specific regions.

**Parameters**:

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `target_agency` | number | Yes | Target agency coordinate (-2 to +2) |
| `target_perceived_justice` | number | No* | Target perceived justice coordinate (-2 to +2) |
| `target_fairness` | number | No | DEPRECATED: use target_perceived_justice |
| `target_belonging` | number | Yes | Target belonging coordinate (-2 to +2) |
| `domain` | string | No | Domain context: "corporate", "government", "religion", "general" (default: general) |

*Either `target_perceived_justice` or `target_fairness` should be provided.

**Returns**:
- Target coordinates
- Suggested probe characteristics
- Recommended approach for generation

**Example**:
```json
{
  "target_agency": -1.5,
  "target_perceived_justice": -1.0,
  "target_belonging": -0.5,
  "domain": "corporate"
}
```

---

#### `get_manifold_state`

Get the current state of the observatory.

**Description**: Returns information about loaded embedding models, projection training status and metrics, number of training examples, and any cached projections.

**Parameters**: None

**Returns**:
- Backend status
- Loaded models count
- Projection status
- Training examples count

**Example**:
```json
{}
```

---

#### `add_training_example`

Add a labeled example to improve projection calibration.

**Description**: Contributes a (text, agency, perceived_justice, belonging) tuple to the training set. After adding examples, the projection can be retrained for improved accuracy.

**Parameters**:

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `text` | string | Yes | The example text |
| `agency` | number | Yes | Agency score (-2 to +2) |
| `perceived_justice` | number | No* | Perceived justice score (-2 to +2) |
| `fairness` | number | No | DEPRECATED: use perceived_justice |
| `belonging` | number | Yes | Belonging score (-2 to +2) |

*Either `perceived_justice` or `fairness` is required.

**Returns**:
- Confirmation message
- Total training examples count

**Example**:
```json
{
  "text": "Despite the setbacks, I believe in our mission",
  "agency": 0.8,
  "perceived_justice": 0.5,
  "belonging": 1.2
}
```

---

#### `explain_modes`

Explain the narrative mode classification system.

**Description**: Returns detailed descriptions of the four narrative mode categories and their sub-modes.

**Parameters**:

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `mode` | string | No | Specific mode to explain: "positive", "shadow", "exit", "noise", or "all" |

**Returns**:
- Mode descriptions
- Example narratives for each mode

**Example**:
```json
{
  "mode": "shadow"
}
```

---

#### `detect_hypocrisy`

Analyze potential hypocrisy gap in organizational texts.

**Description**: Compares stated/espoused values against the inferred values from actual communications. A high delta suggests misalignment between what an organization claims to value and what its language reveals.

**Parameters**:

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `espoused_texts` | array[string] | Yes | Texts representing stated values (mission statements, etc.) |
| `operational_texts` | array[string] | Yes | Texts from actual operations (emails, policies, etc.) |

**Returns**:
- Espoused values centroid
- Inferred values centroid
- Delta per axis
- Overall hypocrisy gap score
- Interpretation

**Example**:
```json
{
  "espoused_texts": [
    "We value every employee's voice",
    "Transparency is our foundation"
  ],
  "operational_texts": [
    "This decision has been made at the executive level",
    "Details will be shared on a need-to-know basis"
  ]
}
```

---

#### `run_scenario`

Simulate narrative evolution through a scenario.

**Description**: Runs a scenario simulation where narratives (solitons) evolve through multiple eras with different reward structures. Note: Full simulation requires the web UI.

**Parameters**:

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `scenario_id` | string | Yes | Pre-built scenario ID: "market_disruption", "polarization", "reformation", or "custom" |
| `initial_narratives` | array[string] | No | Starting narrative texts (for custom scenarios) |
| `era_configs` | array[object] | No | Era configurations with agency/justice/belonging weights |

**Returns**:
- Scenario information
- Guidance for running simulations

**Example**:
```json
{
  "scenario_id": "market_disruption"
}
```

---

### Projection Mode Management (v2)

#### `list_projection_modes`

List available projection modes with their characteristics.

**Description**: Returns information about each projection configuration including accuracy metrics, required models, and recommendations for different use cases.

**Parameters**: None

**Returns**:
- Current active mode
- Available modes with:
  - Name and display name
  - Availability status
  - Required models
  - CV score
- Recommendations for best accuracy, robustness, speed, and uncertainty quantification

**Available Modes**:
| Mode | CV Score | Description |
|------|----------|-------------|
| `current_projection` | 0.383 | Default MiniLM-based, fastest |
| `mpnet_projection` | 0.612 | Best accuracy with all-mpnet-base-v2 |
| `multi_model_ensemble` | N/A | Best robustness, averages 3 models |
| `ensemble_projection` | N/A | 25-model bootstrap for uncertainty |

**Example**:
```json
{}
```

---

#### `select_projection_mode`

Switch the active projection mode used for analysis.

**Description**: After selecting a mode, subsequent `project_text` and `analyze_with_uncertainty` calls will use the selected projection. Mode selection persists until changed.

**Parameters**:

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `mode` | string | Yes | Mode to activate: "current_projection", "mpnet_projection", "multi_model_ensemble", or "ensemble_projection" |

**Returns**:
- Success status
- Mode details
- Required models

**Example**:
```json
{
  "mode": "mpnet_projection"
}
```

---

#### `get_current_projection_mode`

Get the currently active projection mode.

**Description**: Returns the name and details of the projection mode currently in use, including required models and performance characteristics.

**Parameters**: None

**Returns**:
- Current mode name
- Mode description
- CV score (if applicable)
- Required models

**Example**:
```json
{}
```

---

#### `analyze_with_uncertainty`

Analyze text using ensemble projection for uncertainty quantification.

**Description**: Returns coordinates plus 95% confidence intervals for each axis, allowing you to understand how confident the projection is. Particularly useful for texts that might straddle multiple modes or for high-stakes classifications.

**Parameters**:

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `text` | string | Yes | The text to analyze with uncertainty |
| `model_id` | string | No | Embedding model to use (default: all-MiniLM-L6-v2) |

**Returns**:
- Coordinates with standard deviations
- 95% confidence intervals per axis
- Mode classification with confidence
- Overall uncertainty confidence score

**Example**:
```json
{
  "text": "The restructuring might actually help us, though I'm not sure yet"
}
```

---

#### `get_soft_labels`

Get full probability distribution across all 12 narrative modes.

**Description**: Unlike simple mode classification which returns a single label, this returns soft labels (probabilities) for EVERY mode, allowing nuanced interpretation of texts that straddle multiple narrative categories.

**Parameters**:

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `text` | string | Yes | The text to get soft labels for |
| `model_id` | string | No | Embedding model to use (default: all-MiniLM-L6-v2) |

**Returns**:
- Primary and secondary modes with probabilities
- Category classification
- Stability score
- Boundary case indicator
- Full probability distribution across all 12 modes

**Example**:
```json
{
  "text": "We used to believe in this place, but those days are gone"
}
```

---

### Comparative Analysis (v2)

#### `compare_narratives`

Compare two groups of texts and get gap analysis.

**Description**: Computes centroids for each group and measures the distance between them across all three axes. Reveals systematic differences in how two populations frame their narratives.

**Parameters**:

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `group_a_name` | string | Yes | Label for the first group (e.g., "leadership") |
| `group_a_texts` | array[string] | Yes | List of texts from group A |
| `group_b_name` | string | Yes | Label for the second group (e.g., "employees") |
| `group_b_texts` | array[string] | Yes | List of texts from group B |

**Returns**:
- Group A statistics (centroid, mode distribution)
- Group B statistics (centroid, mode distribution)
- Gap analysis (delta per axis, total gap)
- Interpretation

**Example**:
```json
{
  "group_a_name": "Executive Communications",
  "group_a_texts": ["Our vision is clear and achievable", "Together we will succeed"],
  "group_b_name": "Employee Feedback",
  "group_b_texts": ["Nobody asked for our input", "Another initiative from above"]
}
```

---

#### `track_trajectory`

Track narrative evolution over timestamped texts.

**Description**: Analyzes how narratives shift in the cultural manifold over time, detecting trend direction, velocity, and inflection points. Reveals whether an organization's narrative is drifting toward shadow modes, stabilizing, or transforming.

**Parameters**:

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `name` | string | Yes | Label for this trajectory (e.g., "company_comms_2024") |
| `texts` | array[string] | Yes | List of texts in chronological order |
| `timestamps` | array[string] | Yes | ISO timestamps for each text (e.g., "2024-01-15T10:30:00Z") |

**Returns**:
- Trend summary (direction, velocity, acceleration)
- Points with coordinates and modes
- Inflection points detected
- Summary interpretation

**Example**:
```json
{
  "name": "quarterly_updates",
  "texts": ["Q1: Strong growth ahead", "Q2: Facing headwinds", "Q3: Restructuring needed"],
  "timestamps": ["2024-01-15T00:00:00Z", "2024-04-15T00:00:00Z", "2024-07-15T00:00:00Z"]
}
```

---

### Alert System (v2)

#### `create_alert`

Create an alert rule for monitoring narrative drift.

**Description**: Sets up automated monitoring that triggers when specified conditions are met. Alerts persist and can be checked against new texts as they arrive.

**Parameters**:

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `name` | string | Yes | Unique name for this alert rule |
| `type` | string | Yes | Alert type: "gap_threshold", "mode_shift", "trajectory_velocity", or "boundary_crossing" |
| `config` | object | Yes | Alert-specific configuration |

**Alert Types**:
- `gap_threshold`: Trigger when gap between groups exceeds a threshold
- `mode_shift`: Trigger when dominant mode changes
- `trajectory_velocity`: Trigger on rapid movement in the manifold
- `boundary_crossing`: Trigger when texts cross mode boundaries

**Returns**:
- Alert ID
- Alert details
- Confirmation

**Example**:
```json
{
  "name": "culture_decay_alert",
  "type": "gap_threshold",
  "config": {
    "threshold": 0.5,
    "axis": "agency"
  }
}
```

---

#### `check_alerts`

Check texts against all configured alert rules.

**Description**: Projects both groups of texts and evaluates all active alerts to determine which (if any) have been triggered.

**Parameters**:

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `group_a_texts` | array[string] | Yes | First group of texts to check |
| `group_b_texts` | array[string] | Yes | Second group of texts to check |

**Returns**:
- Number of alerts checked
- Number triggered
- Triggered alert details (name, type, trigger value, threshold)

**Example**:
```json
{
  "group_a_texts": ["Leadership message 1", "Leadership message 2"],
  "group_b_texts": ["Employee post 1", "Employee post 2"]
}
```

---

### Advanced Analytics (v2)

#### `detect_outlier`

Detect anomalous narratives by comparing a test text against a corpus.

**Description**: Projects the corpus to establish baseline statistics (centroid, variance), then evaluates how far the test text deviates. Uses Mahalanobis distance and z-scores to identify statistically unusual narratives.

**Parameters**:

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `corpus` | array[string] | Yes | List of texts establishing the baseline norm |
| `test_text` | string | Yes | Text to evaluate for anomaly |

**Returns**:
- Is outlier (boolean)
- Outlier score (Mahalanobis distance)
- Z-scores per axis
- Corpus statistics
- Test text projection
- Interpretation

**Example**:
```json
{
  "corpus": [
    "Great teamwork today!",
    "Solid progress on the project",
    "Enjoyed the team lunch"
  ],
  "test_text": "The whole system is rigged against us"
}
```

---

#### `analyze_cohorts`

Multi-group cohort analysis with ANOVA.

**Description**: Compares multiple groups simultaneously using statistical tests to determine if there are significant differences between cohorts.

**Parameters**:

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `cohorts` | object | Yes | Dictionary mapping cohort names to lists of texts |

**Returns**:
- Per-cohort statistics (centroid, dominant mode)
- ANOVA results (F-statistic, p-value, significance)
- Per-axis ANOVA breakdown
- Significant pairwise differences
- Interpretation

**Example**:
```json
{
  "cohorts": {
    "engineering": ["We ship quality code", "Technical debt is manageable"],
    "sales": ["Customers love us!", "Pipeline is strong"],
    "support": ["Tickets are overwhelming", "Users are frustrated"]
  }
}
```

---

#### `analyze_mode_flow`

Analyze mode transitions and detect patterns in a sequence of texts.

**Description**: Maps how narratives flow through different modes over a sequence, computing transition probabilities and identifying patterns. Reveals which modes are "sticky" (stable) vs "volatile" (texts quickly leave).

**Parameters**:

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `texts` | array[string] | Yes | List of texts to analyze for mode flow patterns |

**Returns**:
- Mode sequence
- Stable modes (texts tend to stay)
- Volatile modes (texts tend to leave)
- Transition matrix
- Detected patterns
- Interpretation

**Example**:
```json
{
  "texts": [
    "Excited to start!",
    "Things are getting harder",
    "Why do we even bother?",
    "Found a new approach!",
    "This might actually work"
  ]
}
```

---

### Force Field Analysis (v2)

#### `analyze_force_field`

Analyze the attractor/detractor force field of a narrative.

**Description**: Returns the forces acting on the narrative - what it's being pulled TOWARD (attractors) and pushed AWAY FROM (detractors).

**Parameters**:

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `text` | string | Yes | The text to analyze for force field dynamics |

**Returns**:
- Attractor strength and detractor strength
- Net force and force direction
- Force quadrant and description
- Energy level
- Primary/secondary attractors and detractors
- Attractor and detractor scores by target/source

**Example**:
```json
{
  "text": "We need to escape this toxic environment and build something better together"
}
```

---

#### `analyze_trajectory_forces`

Analyze how force fields evolve across a sequence of texts.

**Description**: Tracks attractor/detractor changes over time, identifying shifts in goals, emergence/resolution of threats, energy changes, and quadrant transitions.

**Parameters**:

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `texts` | array[string] | Yes | Sequence of texts in chronological order (minimum 2) |

**Returns**:
- Attractor trajectory (trend, start/end values, change)
- Detractor trajectory
- Energy trajectory
- Attractor and detractor shifts
- Per-point force analysis
- Interpretation

**Example**:
```json
{
  "texts": [
    "Getting out of this mess",
    "Starting to see possibilities",
    "Building toward something meaningful"
  ]
}
```

---

#### `compare_force_fields`

Compare force fields between two groups of narratives.

**Description**: Identifies differences in attractor/detractor strengths, primary targets/sources, energy levels, and quadrant distributions between two groups.

**Parameters**:

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `group_a_name` | string | Yes | Label for the first group |
| `group_a_texts` | array[string] | Yes | Texts from group A |
| `group_b_name` | string | Yes | Label for the second group |
| `group_b_texts` | array[string] | Yes | Texts from group B |

**Returns**:
- Group A statistics (mean attractor/detractor, dominant quadrant, targets)
- Group B statistics
- Comparison (gaps, which group is higher, alignment)
- Interpretation

**Example**:
```json
{
  "group_a_name": "Marketing",
  "group_a_texts": ["Inspiring customers to achieve more"],
  "group_b_name": "Operations",
  "group_b_texts": ["Avoiding system failures at all costs"]
}
```

---

#### `get_force_targets`

List all attractor targets and detractor sources.

**Description**: Returns the complete taxonomy of attractor targets, detractor sources, and force quadrants for reference.

**Parameters**: None

**Returns**:
- Attractor targets with descriptions
- Detractor sources with descriptions
- Force quadrants with descriptions and energy levels

**Example**:
```json
{}
```

---

### High-Level Narrative Analysis

#### `fetch_narrative_source`

Fetch and segment content from a URL into analysis units.

**Description**: Supports multiple source types (websites, RSS/Atom feeds, news sites, blogs). Automatically detects source type, extracts meaningful text content, segments into analysis-ready units, and preserves metadata.

**Parameters**:

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `url` | string | Yes | The URL to fetch (website, RSS feed, etc.) |
| `source_type` | string | No | Optional type hint: "website", "rss", "twitter", "reddit", "blog", "news", "document" |
| `max_items` | integer | No | Maximum content units to return (default: 100) |
| `include_metadata` | boolean | No | Include source metadata (default: true) |

**Returns**:
- Source information
- Source type detected
- Content units with text and metadata
- Count of units

**Example**:
```json
{
  "url": "https://example.com/blog/rss.xml",
  "source_type": "rss",
  "max_items": 50
}
```

---

#### `build_narrative_profile`

Build a comprehensive narrative profile from content units.

**Description**: Takes content units and synthesizes a complete narrative profile including manifold position, mode analysis, force field, and comparative features.

**Parameters**:

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `content_units` | array[object] | Yes | List of content units with `text` field (from fetch_narrative_source) |
| `source` | string | Yes | Source identifier (URL, handle, name) |
| `source_type` | string | No | Type of source (default: "website") |
| `include_force_analysis` | boolean | No | Include attractor/detractor analysis (default: true) |

**Content Unit Schema**:
```json
{
  "text": "required string",
  "timestamp": "optional ISO string",
  "author": "optional string",
  "metadata": "optional object"
}
```

**Returns**:
- Manifold position (centroid, spread, quadrant)
- Mode analysis (dominant mode, distribution, signature, stability)
- Force field (attractors, detractors, tensions)
- Comparative features (distinctive features, sample quotes)

**Example**:
```json
{
  "content_units": [
    {"text": "Building the future of technology", "timestamp": "2024-01-15"},
    {"text": "Our team is unstoppable", "timestamp": "2024-01-20"}
  ],
  "source": "TechCorp Blog",
  "include_force_analysis": true
}
```

---

#### `get_narrative_suggestions`

Generate actionable suggestions from a narrative profile.

**Description**: Takes a narrative profile and generates suggestions based on intent (understand, engage, counter, bridge, strategy).

**Parameters**:

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `profile` | object | Yes | The narrative profile from build_narrative_profile |
| `intent` | string | No | Primary goal: "understand", "engage", "counter", "bridge", "strategy" (default: understand) |

**Intent Types**:
- `understand`: Research/analysis insights
- `engage`: How to interact and resonate
- `counter`: How to position against
- `bridge`: Finding common ground
- `strategy`: Content and positioning opportunities

**Returns**:
- Suggestions organized by type (understanding, engagement, strategy, warning)
- Each suggestion includes priority, insight, recommendation, evidence, and related coordinates

**Example**:
```json
{
  "profile": { "...profile object from build_narrative_profile..." },
  "intent": "engage"
}
```

---

### Advanced Research (v3)

> **Note**: These tools require additional dependencies and may not be available in all installations.

#### `extract_parsed`

Extract coordinates using dependency parsing.

**Description**: Uses spaCy dependency parsing for linguistically accurate feature extraction, capturing actual syntactic structure, semantic roles, negation scope, and modal modification.

**Requirements**: `pip install spacy && python -m spacy download en_core_web_sm`

**Parameters**:

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `text` | string | Yes | The text to analyze with dependency parsing |

**Returns**:
- Agency analysis (self/other/system scores, events with negation/passive/modal markers)
- Justice analysis (procedural/distributive/interactional scores, events)
- Belonging analysis (ingroup/outgroup/universal scores, markers)
- Metadata (sentence count, passive voice ratio, negation ratio, modal usage ratio)

**Example**:
```json
{
  "text": "I couldn't have done it without the team, though management didn't support us"
}
```

---

#### `extract_semantic`

Extract coordinates using semantic similarity.

**Description**: Uses sentence-transformers to compute semantic similarity between text and prototype sentences, addressing word sense disambiguation issues.

**Requirements**: `pip install sentence-transformers`

**Parameters**:

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `text` | string | Yes | The text to analyze with semantic similarity |
| `show_prototypes` | boolean | No | Include best-matching prototype sentences (default: false) |

**Returns**:
- Core dimensions (9): self_agency, other_agency, system_agency, procedural_justice, distributive_justice, interactional_justice, ingroup, outgroup, universal
- Modifier dimensions (9): certainty, evidentiality, commitment, temporal_focus, temporal_scope, power_differential, social_distance, arousal, valence
- Best-matching prototypes (if requested)

**Example**:
```json
{
  "text": "Due process was followed, ensuring everyone received their fair share",
  "show_prototypes": true
}
```

---

#### `validate_external`

Run external validation against psychological scales.

**Description**: Validates extracted coordinates against established psychological scales for convergent validity assessment.

**Validated Scales**:
- Agency: Sense of Agency Scale (Tapal et al., 2017)
- Justice: Organizational Justice Scale (Colquitt, 2001)
- Belonging: Inclusion of Other in Self Scale (Aron et al., 1992), Brief Sense of Community Scale (Peterson et al., 2008)

**Parameters**:

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `texts` | array[string] | Yes | List of texts to validate |
| `scale_type` | string | Yes | Scale to validate against: "agency", "justice", "belonging" |

**Returns**:
- Correlation coefficient (r) with 95% CI
- p-value and effect size
- Convergent validity assessment (r > 0.50 expected)
- Subscale correlations
- Interpretation

**Example**:
```json
{
  "texts": ["I control my own destiny", "I decide what happens to me"],
  "scale_type": "agency"
}
```

---

#### `get_safety_report`

Get comprehensive safety metrics for deployment assessment.

**Description**: Evaluates detector performance on a labeled test corpus and provides deployment readiness assessment.

**Parameters**:

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `texts` | array[string] | Yes | List of texts to include in safety evaluation |

**Returns**:
- Deployment readiness (research_only, human_in_loop, automated)
- Regime classification metrics (accuracy, macro F1, OPAQUE FPR/FNR)
- Ossification detection metrics
- Adversarial robustness (evasion rates by technique)
- ROC analysis and calibration data
- Blocking issues, warnings, and recommendations

**Deployment Thresholds**:
- Research: accuracy > 50%, samples > 100
- Monitoring: accuracy > 70%, critical_FNR < 30%
- Automation: accuracy > 85%, critical_FNR < 10%, adversarial_robustness > 80%

**Example**:
```json
{
  "texts": ["Sample text 1", "Sample text 2", "..."]
}
```

---

#### `compare_extraction_methods`

Compare regex vs parsed vs semantic extraction methods.

**Description**: Runs the same text through all available extraction methods and provides a side-by-side comparison showing differences and improvements.

**Requirements**: spaCy with en_core_web_sm model

**Parameters**:

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `text` | string | Yes | The text to compare extraction methods on |

**Returns**:
- Regex-based features
- Parsed-based features
- Differences (where methods disagree)
- Improvements (cases where parsing helps: negation, modals, passive voice)
- Detailed agency analysis with events

**Example**:
```json
{
  "text": "The decision was not made by me, though I might have influenced it"
}
```

---

## Example Workflows

### Workflow 1: Organizational Culture Audit

```
1. fetch_narrative_source
   - url: "https://company.com/blog"
   - source_type: "blog"

2. build_narrative_profile
   - content_units: [from step 1]
   - source: "Company Blog"

3. compare_narratives
   - group_a: Public communications
   - group_b: Internal Slack exports

4. detect_hypocrisy
   - espoused_texts: Mission statements, values pages
   - operational_texts: Internal emails, memos

5. get_narrative_suggestions
   - profile: [from step 2]
   - intent: "understand"
```

### Workflow 2: Monitoring for Cultural Drift

```
1. create_alert
   - name: "shadow_mode_alert"
   - type: "mode_shift"
   - config: {"from": "positive", "to": "shadow"}

2. create_alert
   - name: "engagement_gap"
   - type: "gap_threshold"
   - config: {"threshold": 0.8, "axis": "agency"}

3. [Periodically] check_alerts
   - group_a_texts: Latest leadership messages
   - group_b_texts: Latest employee feedback

4. [If triggered] analyze_cohorts
   - cohorts: {"engineering": [...], "sales": [...], "support": [...]}
```

### Workflow 3: Narrative Evolution Analysis

```
1. track_trajectory
   - name: "company_comms_2024"
   - texts: [quarterly all-hands transcripts]
   - timestamps: [quarterly dates]

2. analyze_mode_flow
   - texts: [same texts]

3. analyze_trajectory_forces
   - texts: [same texts]

4. [For anomalies] detect_outlier
   - corpus: [most texts]
   - test_text: [anomalous text]
```

### Workflow 4: Research Validation

```
1. select_projection_mode
   - mode: "ensemble_projection"

2. analyze_with_uncertainty
   - text: "Test text"

3. compare_extraction_methods
   - text: "Complex linguistic text"

4. validate_external
   - texts: [corpus]
   - scale_type: "agency"

5. get_safety_report
   - texts: [evaluation corpus]
```

---

## Troubleshooting

### Backend Not Reachable

**Error**: `Observatory backend not reachable at http://127.0.0.1:8000`

**Solutions**:
1. Ensure the backend is running:
   ```bash
   cd backend && python -m uvicorn main:app --host 127.0.0.1 --port 8000
   ```
2. Check the `OBSERVATORY_BACKEND_URL` environment variable
3. Verify firewall/network settings

### Missing Dependencies for Advanced Research Tools

**Error**: `Parsed extraction not available` or `Semantic extraction not available`

**Solutions**:
```bash
# For parsed extraction
pip install spacy
python -m spacy download en_core_web_sm

# For semantic extraction
pip install sentence-transformers

# For safety metrics
pip install scikit-learn
```

### Projection Mode Not Available

**Error**: `Mode not available` when selecting projection mode

**Solutions**:
1. Check available modes with `list_projection_modes`
2. Ensure required models are loaded
3. For `multi_model_ensemble`, first use `select_projection_mode` to trigger model loading

### Low Confidence Projections

**Symptoms**: Low confidence scores, boundary cases, high uncertainty

**Solutions**:
1. Use `analyze_with_uncertainty` to get confidence intervals
2. Use `get_soft_labels` to see probability distribution
3. Use `compare_projections` to check method agreement
4. Consider adding training examples with `add_training_example`

### Inconsistent Results Across Methods

**Symptoms**: Different modes from different projection methods

**Solutions**:
1. The text may genuinely be ambiguous - this is informative
2. Use `compare_extraction_methods` to understand why
3. Consider the `stability_score` in results
4. Use soft labels rather than hard classifications

### Memory Issues

**Symptoms**: Slow responses, timeouts with large corpora

**Solutions**:
1. Process texts in smaller batches
2. Use `current_projection` mode (fastest)
3. Avoid loading multiple models simultaneously
4. Increase server resources if available

---

## API Versioning Notes

### v1 to v2 Migration

- Parameter `fairness` renamed to `perceived_justice` (backward compatible)
- Mode classification expanded from 4 to 12 modes
- Added soft labels and stability scores
- Added force field analysis

### v2 to v3 Additions

- Advanced extraction methods (parsed, semantic)
- External validation tools
- Safety metrics and deployment assessment
- Extraction method comparison

---

## Support

For issues and feature requests, please file an issue in the Cultural Soliton Observatory repository.
