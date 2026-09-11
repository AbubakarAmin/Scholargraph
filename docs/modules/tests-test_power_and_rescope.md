## Module Overview

Unit tests for power‑analysis integration and capability‑rescope traceability.
These tests verify that:
1. `preregister_power` computes a sensible sample‑size requirement.
2. `PlannerAgent._apply_capability_rescope` records the expected traceability fields
   in the plan when feasibility errors are present.
