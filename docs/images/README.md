# Screenshot checklist

Five real application screenshots were captured from the offline demo on
2026-08-06. They contain only repository-owned synthetic data. The container
startup and health endpoint were validated again on 2026-08-18; a
reviewer-facing Docker health screenshot remains outstanding, and no mockup was
substituted.

Required artifacts:

- [x] `streamlit-overview.png` — offline demo interface with the sidebar,
  example questions, and a completed response.
- [x] `cited-rag-response.png` — answer with source citations and similarity
  details visible.
- [x] `safety-gate.png` — deployment or destructive-action request showing the
  approval/safety response and no execution claim.
- [x] `feedback-and-cost.png` — current-session feedback controls and the cost
  estimate/session total.
- [x] `offline-monitoring.png` — synthetic-only monitoring page with KPIs,
  strategy comparisons, and the synthetic-data warning visible.
- [ ] `docker-health.png` — running container and successful health state or
  health endpoint.

Before committing an image:

1. Run in default demo mode with the synthetic CC0 corpus.
2. Use a clean browser profile or crop unrelated browser and desktop content.
3. Remove account IDs, user names, local filesystem paths, credentials,
   cookies, tokens, customer data, and notification content.
4. Confirm that model/cost labels match the active mode and that no execution
   is implied.
5. Use a readable PNG at a practical repository size and add descriptive alt
   text when linking it from the README.

Optional submission evidence includes a short GIF or video walkthrough, but it
must follow the same redaction and accuracy rules.
