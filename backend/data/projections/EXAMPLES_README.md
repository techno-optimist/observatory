# Training Data: examples.json

## Axis Naming Convention

**IMPORTANT (January 2026 Update):** The "fairness" field in this training data represents "perceived_justice" in the API responses.

The validity study found that the original "Fairness" axis conflates:
1. Abstract fairness values ("Everyone deserves equal treatment")
2. System legitimacy beliefs ("The system is rigged")

The axis has been renamed to "Perceived Justice" to accurately reflect what it measures.

### Field Mapping

| Field in JSON | API Response Name | Description |
|---------------|------------------|-------------|
| `agency` | `agency` | Sense of personal control and self-determination |
| `fairness` | `perceived_justice` | Belief in fair treatment and system legitimacy |
| `belonging` | `belonging` | Sense of social connection and group membership |

### Why Keep "fairness" in Training Data?

For backward compatibility with:
- Existing trained projection models
- Saved projection weights
- Historical experiment records
- Legacy code that may reference these files

The internal storage uses "fairness" while the API layer translates to "perceived_justice" in responses.

## Data Format

Each training example contains:
- `text`: The example statement
- `agency`: Score from -2.0 to 2.0
- `fairness`: Score from -2.0 to 2.0 (represents perceived_justice)
- `belonging`: Score from -2.0 to 2.0
- `source`: Origin of the example (e.g., "manual", "human_validated")

## Validity Study Reference

See `/backend/VALIDITY_STUDY_REPORT.md` for the complete findings that led to this rename.
