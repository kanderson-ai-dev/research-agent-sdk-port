# Security Posture

This document maps the controls implemented in this service to the OWASP
Top 10 (web) and OWASP LLM Top 10. It is written to be auditable: every claim
points at the module that enforces it.

## Threat model

The service accepts a research topic, autonomously searches and scrapes the
public web, feeds parsed content to an LLM, and publishes a report behind a
human review gate. The two trust boundaries are:

1. **User input** — the topic string (API consumers, authenticated).
2. **Web content** — pages/documents fetched from arbitrary URLs found by
   search. This is the distinguishing risk of the project: *indirect prompt
   injection* embedded in third-party content.

## OWASP LLM Top 10 mapping

| Risk | Control | Implementation |
|---|---|---|
| LLM01 Prompt Injection (direct) | Input guardrail screens every topic before any LLM call or paid request; wired as a native `@input_guardrail` on the planner agent (`run_in_parallel=False` so it runs *before* the model call, not concurrently); blocked topics get a generic 422 | `app/guardrails.py:screen_topic`, `topic_injection_guardrail` in `app/agents/definitions.py`, `input_guardrail` stage, `POST /research` |
| LLM01 Prompt Injection (indirect) | All scraped content is treated as hostile *data*: invisible-character normalization, instruction-phrase neutralization, length caps — applied before anything reaches the model | `sanitize_scraped_content` in `app/guardrails.py`, invoked in the document-worker stage |
| LLM02 Sensitive Information Disclosure | Secrets are `SecretStr \| None` (empty string → unset); generic error messages; structlog processor redacts `*_key`, `*_token`, `*_password`, `authorization`, `*_secret` fields; no secrets in query strings (Bearer header only) | `app/core/config.py`, `app/core/logging.py`, `app/core/security.py` |
| LLM03 Supply Chain | Locked dependencies (`uv.lock`), `pip-audit` + `gitleaks` in CI | `pyproject.toml`, `.github/workflows/ci.yml` |
| LLM04 Data & Model Poisoning | Output is grounded in fetched sources: every citation's quote must be verifiably present in its source's extracted text (verbatim, punctuation-insensitive, or ≥0.85 sliding-window similarity) or it is dropped; wired as a native `@output_guardrail` on the writer agent that filters and counts rather than merely blocking | `verify_citations`, `citation_support_guardrail` in `app/agents/definitions.py`, `tests/fixtures/adversarial_page.html` |
| LLM05 Improper Output Handling | Console renders the report escaped (no raw HTML injection); output guardrail strips unsupported citations before the human gate | `frontend/console/app.js:renderMarkdown`, `output_guardrail` stage |
| LLM06 Excessive Agency | Bounded re-plan loop (`max_critic_rounds` enforced by the orchestrator), read-only tool functions, per-job cost accounting, mandatory human review before a report is considered published | `app/pipeline/runner.py`, `human_review` stage |
| LLM07 System Prompt Leakage | Agent `instructions` are server-side only; refusal messages are generic and carry no prompt material | `rejection_output` stage, generic API errors |
| LLM08 Vector & Embedding Weaknesses | Not applicable — the service does not use embedding stores | — |
| LLM09 Misinformation | Citation support rate is measured and gated in CI (≥ 0.90); unresolved sub-questions are surfaced honestly instead of being written around | `evaluation/`, `output_guardrail`, `report_assembler` |
| LLM10 Unbounded Consumption | Rate limiting on `/auth/login` and `POST /research`; bounded critic loop; scraper byte/time limits; per-job token accounting with USD estimate | `app/core/rate_limit.py`, `app/services/usage.py`, `ScraperClient` |

## OWASP Top 10 (web) mapping

| Risk | Control |
|---|---|
| Broken Access Control | JWT Bearer auth on all `/api/v1/research*` endpoints (HS256, issuer + expiry checked); uniform 401 responses |
| Cryptographic Failures | `SecretStr` everywhere; secrets only via headers/env — never query strings; no secrets in logs (redaction processor) |
| Injection | No string-interpolated queries (SQLite via parameterized `aiosqlite`); topic guardrail before any LLM call |
| Insecure Design | Guardrail-first pipeline: malicious input is rejected before any LLM spend |
| Security Misconfiguration | Security headers middleware (`CSP`, `X-Frame-Options`, `X-Content-Type-Options`, `Referrer-Policy`); CI `permissions: contents: read`, `pull_request` only |
| Vulnerable Components | `pip-audit` in CI; locked dependency file |
| Auth Failures | Rate-limited login; generic "invalid credentials"; expired/invalid tokens share one 401 |
| SSRF | Scraper fetches only `http(s)` URLs, respects `robots.txt`, caps response size and enforces timeouts |
| Logging Failures | structlog JSON pipeline, `job_id` correlation, sensitive-field redaction |
| Integrity of Data | Paused jobs persist a serialized `PipelineContext` in the `paused_states` table; the HITL gate records the human decision |

## Responsible scraping

`ScraperClient` enforces: `robots.txt` compliance (cached per origin), a
configurable delay between requests, a descriptive User-Agent, response size
and content-type limits, and an opt-in Playwright fallback (lazy import —
never loaded unless configured).

## Secret hygiene

- `.env`, `data/*.sqlite`, and all planning docs are gitignored.
- `.env.example` documents every variable with empty values.
- `gitleaks detect` runs in CI; no secret may enter code, tests, fixtures,
  commit messages or this file.

## Known limitations

- Auth is a single static credential (`ADMIN_*`) — appropriate for a
  portfolio/demo deployment, not multi-tenant production.
- The rate limiter is in-process: it does not coordinate across replicas.
- The SSRF guard allowlists schemes but does not yet block private IP ranges
  (documented for a follow-up hardening pass).
