# AIRA — Product Vision & Rescoped Roadmap

> Written 2026-07-22, superseding the "Long-Term Vision / Roadmap" section of CLAUDE.md
> for product strategy and sequencing. CLAUDE.md remains the technical architecture
> reference; this document is the strategy reference. Decisions here were made with the
> owner in an interactive planning session — see the Decision Log (§9).

## 1. Thesis

AIRA today is a single-user tool that analyzes research papers for integrity and
reproducibility signals: it verifies every citation against Crossref (catching
unverifiable and fabricated references), extracts statistical indicators (p-values,
sample sizes, effect sizes, CIs), scores open-science transparency markers into a
heuristic 0–100 integrity grade, and finds related work via SPECTER embeddings and
OpenAlex. The rescoped ambition: a **hosted research workspace where integrity
intelligence is the wedge** — a genuinely competitive product in the space occupied by
Elicit, Scite, and ResearchRabbit, not just a portfolio piece.

The operating posture, decided explicitly: **invest in the UI now** (it pays off in
every outcome), **run market/legal research in parallel** (owner-run, starting
immediately), and **gate all multi-tenant architecture work behind an owner-reviewed
go/no-go decision** once that research returns. Nothing at risk of being built in the
wrong direction starts before the gate.

## 2. Current state snapshot (2026-07-22)

Honest baseline after Weeks 1–2 of the original roadmap:

- **Works**: full analysis pipeline (extract → verify → score), persistent library
  (SQLite + Alembic), projects, SPECTER similarity, OpenAlex related works, React
  library UI. 88 tests green, CI on GitHub Actions.
- **Single-user by construction**: no auth anywhere; no `user_id` on any of the 6
  tables; all data in one global namespace; CORS defaults to explicit local origins, with
  `allow_credentials` gated so a wildcard origin can never combine with credentials
  (`app/main.py:32-41`); PDFs on local disk at `data/pdfs/{sha256}.pdf` with global
  byte-dedup and no ownership concept (deletes orphan blobs).
- **Local-scale engineering**: SQLite with JSON-as-text columns and LargeBinary numpy
  vectors; SPECTER (~2 GB RAM) loaded in-process, `model.encode()` called synchronously
  inside async handlers (blocks the event loop under concurrency); no rate limiting; no
  Dockerfile.
- **Frontend**: the *foundation* is modern and solid — Vite, React 18, Tailwind,
  ~40 shadcn/ui primitives, React Query with sane invalidation, a clean API client, and
  `reportTransform.js` (the 148-line domain contract feeding both fresh-upload and
  stored-report views). The *composition* is toy-grade: SaaS-template landing page,
  a 461-line `Home.jsx` view-state machine with dead mock code, unreachable
  news-analysis panels, unwired dark mode.
- **Legacy**: `app_v2.py` (Streamlit) is hereby **frozen** — no further investment.
  `app_legacy.py` is archival. The product story is the React frontend only.

## 3. Operating model: two tracks + one gate

```
Track A (focused window, now → ~mid-Aug 2026)   UI redesign on current stack
Track B (parallel, starts immediately)           Owner runs deep-research questions (§7)
                                                       │
                                        GATE (~late Aug 2026, owner-reviewed)
                                        GO / NO-GO / PIVOT on hosted multi-tenant
                                                       │
                              GO → phased multi-tenant foundation (§6, ≈ Q1 2027 beta)
                              NO-GO → portfolio-polish fallback (§6)
```

**Gate inputs** (from the §7 research): (1) is integrity checking a defensible wedge or
a feature incumbents already bundle; (2) does anyone credibly pay, and who; (3) is the
legal exposure (hosting copyrighted PDFs + publishing integrity scores) manageable for a
solo operator; (4) the owner's appetite for ops/support burden. **PIVOT is a named third
outcome** — e.g., research showing the real buyer is publishers doing submission
screening would reprioritize toward an API-first B2B shape rather than a consumer
workspace, changing which forks in §5 matter first.

**Pre-gate-safe backend shortlist** — the *only* backend work permitted before the gate
resolves (all of it valuable in every outcome): wrap `model.encode()` in
`asyncio.to_thread` (existing backlog item), Dockerfile + compose, dependency lockfile,
CORS/env hygiene.

## 4. Track A — UI redesign (the focused window)

Stack decision: redesign **on the current stack**. Vite + React 18 + Tailwind +
shadcn/ui + React Query stay; information architecture, pages, and the visual system are
rebuilt. All rebuilt surfaces land in TypeScript (`.tsx`); carried-over files stay JS
until next rewritten.

### Keep (logic and contracts)

- `frontend/src/api/airaApi.js`, `frontend/src/api/queries.js` — data layer
- `frontend/src/lib/reportTransform.js` — the report→UI domain contract; **do not fork**
- `grades.js`, shadcn primitives (re-tokened, not replaced)
- CitationTable's sort/filter/badge logic; QualityGauge's SVG internals; FileUploader's
  drag-drop state machine — logic kept, presentation rebuilt

### Rebuild (presentation, → .tsx)

Layout, Library, PaperDetail, ProjectDetail, PaperCard, StatisticsPanel,
SummaryAccordion, ContentHeader/PaperHeader, Similar/Related panels, AnalysisProgress
(status-driven), and the entire upload surface.

### Delete (git history is the archive)

HeroSection, FeatureGrid, WaitlistSection, ContentTypeSelector, UrlInput,
NewsAnalysisTabs, BiasPanel, LogicPanel, TrustPanel, the mock/news dead code inside
Home.jsx, App.css remnants. The news-analysis path exits the codebase (it remains listed
in CLAUDE.md's deferred table as an idea, not as code).

### What "professional, not vibe-coded" means here

- **IA**: the app opens into the workspace (library-first, `/` → Library) — never a
  marketing funnel. Navigation names objects (Papers, Projects, Reports), not features.
  Upload is an action (modal/drawer), not a homepage.
- **Density**: researchers scan; they don't scroll heroes. Data-dense tables, 13–14px
  body, tabular numerals for statistics, one spacing scale.
- **Tokens**: every color/space/radius from semantic CSS variables; status colors
  (verified / suspicious / not_found) and grade colors (A–F) defined once and used
  identically in badges, gauges, and cards.
- **States**: designed skeleton, empty-with-action, and error-with-retry for every view.
  No spinner-only screens, no dead ends, no `alert()`.
- **A11y floor**: keyboard-navigable tables, visible focus, `aria-sort`, AA contrast,
  reduced-motion support. Dark mode actually wired (ThemeProvider — scaffolding already
  exists).
- **Framing**: the score is presented as "transparency signals," never a verdict on a
  paper or its authors. This is both honest UX and legal mitigation (§7 Q13, §8).

### Phases and checkpoints

| Phase | Days | Scope | Checkpoint (owner-reviewed) |
|-------|------|-------|------------------------------|
| A0 | 1–2 | Design references (Elicit/Scite/Semantic Scholar/Linear-class density), semantic token system, internal `/styleguide` route | Tokens + one mocked screen before any page work |
| A1 | 3–5 | App shell + IA; dismantle Home.jsx into routes; **upload flow built async-shaped** (`uploading → queued → analyzing → done/failed`) even though the backend is still synchronous | Shell walkthrough |
| A2 | wk 2 | Pages in traffic order: Library → PaperDetail (the money screen) → upload flow | Core-flows demo |
| A3 | wk 3 / overflow | ProjectDetail, similar/related panels, a11y/responsive/dark-mode polish, screenshots + demo GIF | Final review; window closes regardless |

The A1 async-shaped upload contract is the one deliberate coupling to the post-gate
world: when ingest becomes a background job (§5 F5), the UI already speaks
job-status and needs no second rebuild.

**Non-goals in the window**: Discovery UI (original Week 3 — skipped), auth UI, any
backend change beyond the pre-gate-safe shortlist in §3.

## 5. Architecture fork map — fork vs additive

What "startup-worthy" requires beyond the current foundation, classified honestly.
**Forks** change assumptions baked into the current design; they are all post-gate.
**Additive** items bolt on without rework and are safe anytime.

### Genuine forks (post-gate only)

**F1 — Single-tenant → multi-tenant data model + query audit** (~2–3 focused weeks)
`user_id` lands on `Paper` and `Project` (children — AnalysisReport, Reference,
Embedding, ProjectPaper — scope via FKs). Every query in `app/routers/papers.py` and
`app/routers/projects.py` must be tenant-scoped: list pagination, detail, the
`/similar` cosine candidate set, project filters, and critically the **sha256 dedup
lookup** in `app/services/ingest.py`. One missed scope is a cross-tenant data leak, so
the deliverable includes tenant fixtures and named cross-tenant-leak tests, plus a
404-vs-403 policy. **Trap**: `Paper.sha256` is globally unique today. Options: per-user
rows + per-user blobs (copyright-safe default — see §7 Q9 — at the cost of storage
dedup, which is cheap at 1–10 MB/PDF), or shared blobs keyed by sha256 (keeps dedup but
serving user B a file user A uploaded is arguably *distribution*, and the dedup
response leaks "someone already uploaded this").

**F2 — SQLite → Postgres** (~1 focused week; do **before** F1 so the tenancy migration
is written once against the final dialect)
Known frictions: `Embedding.vector` LargeBinary numpy bytes (pgvector is optional and
deferrable — bytea + numpy brute force is fine at launch scale), JSON-as-text →
JSONB, naive-UTC datetimes → timestamptz, `alembic.ini`'s hardcoded sqlite URL →
env-driven, CI needs a Postgres service container.

**F3 — Local disk → object storage with ownership** (~3–5 focused days)
`pdf_path` becomes a storage key; presigned GETs for any future PDF viewer; resolves the
existing orphan-blob lifecycle bug deliberately (per-user prefixes make delete trivial;
shared blobs need refcounting). Layout depends on F1's blob decision, which depends on
the §7 Q9 legal answer.

**F4 — Trust-the-client → authn/z + rate limiting** (~1 focused week + 2–4 days)
Managed auth (decision #4): FastAPI dependency verifying provider JWTs via JWKS, user
row provisioned on first-seen `sub`, frontend auth provider + token injection in the API
client. Rate limiting depends on auth to be meaningful — per-IP is nearly useless; the
quota that matters is **per-user ingest**, the expensive path (OCR + SPECTER + up to
~20 references × several Crossref calls each). Includes fixing the CORS
`*`+credentials combination.

**F5 — Synchronous in-process ML → background jobs** (~1–2 focused weeks)
`POST /api/papers` → 202 + job id, status endpoint, worker process owning the 2 GB
SPECTER model; queue either Postgres-backed (e.g., procrastinate — no new
infrastructure) or arq. The UX contract change is already absorbed by Track A's
async-shaped upload flow. **Hosting economics**: SPECTER's footprint sets the floor at
a ~4 GB instance (≈ $20–40/mo — no free tier). The alternative (hosted embeddings API)
changes economics but **invalidates every stored vector** — different model, different
vector space. Before any model swap: add an embedding-model identifier to `Embedding`
(mirroring `analyzer_version` on reports).

### Additive hardening (safe anytime; several worth doing pre-gate)

| Item | Effort | Note |
|------|--------|------|
| CORS/env hygiene | hours | explicit origins; required-var validation |
| Dockerfile + compose | 1–2 days | **pre-gate** — helps every outcome |
| Lockfile (uv/pip-tools) + dependabot | hours | **pre-gate** |
| `asyncio.to_thread` around encode | ~1 day | **pre-gate**; existing backlog item |
| Security headers | ~1 day | mostly proxy-level |
| Structured logging + request IDs | 1–2 days | |
| Sentry + basic metrics | 1–2 days | |
| Secrets management | hours now; real work at deploy | rotate the local OpenAI key as cheap hygiene (`.env` verified never git-tracked — no incident) |
| Upload hardening (OCR timeouts, pdf-bomb caps) | 1–2 days | size/page caps already exist |

The in-process 24 h TTL caches (references, related works) are fine single-instance;
they move to a shared/persistent cache only if the deployment ever becomes
multi-instance — not a fork today.

## 6. Post-gate roadmap

Honest calendar math: post-window pace is ≈ 5–8 hrs/week, so focused-week estimates
convert at roughly **×3 into calendar weeks**. Stated plainly so future sessions don't
re-inflate expectations.

### GO — hosted multi-tenant beta

| Phase | Scope | Calendar (slow pace) |
|-------|-------|----------------------|
| P1 — Deploy foundation (still single-user) | Docker, F2 Postgres, secrets, host (Railway/Fly/Render-class), `to_thread` band-aid, logging + Sentry | 4–6 wks |
| P2 — Identity + tenancy | F4 managed auth + user table; F1 `user_id` + query audit + cross-tenant tests; blob/dedup decision per legal answer; F3 object storage | 6–10 wks |
| P3 — Scale-safety | F5 background jobs (UI already shaped for it); per-user quotas; global outbound Crossref/OpenAlex budget; persistent reference cache | 5–7 wks |
| P4 — Beta hardening | ToS/privacy/DMCA (from §7 legal research), abuse controls, invite-gated beta | 3–4 wks |

**Total: hosted multi-tenant beta ≈ 4–6 months post-gate → roughly Q1 2027.**

### NO-GO — portfolio-polish fallback

Stay single-user (local or personally hosted). README rewrite + architecture diagram +
demo assets (Track A's A3 already produced screenshots/GIF). Resume the original Week-4
**conflict-detection** track from CLAUDE.md as the next depth feature — it was always
the most defensible technical work. All pre-gate hardening keeps its value.

### Either-way hedge

A **read-only hosted demo** — uploads disabled, 10–20 curated open-access papers
preloaded — sidesteps the entire copyright/abuse surface while still being a live,
shareable product. Cheap to stand up after P1-level deploy work; worth doing in both
branches.

## 7. Deep-research question list (owner-run, starting now)

Run each numbered question as a **separate Perplexity Deep Research prompt**, prepending
this context line to each:

> Context: AIRA is a web tool that analyzes uploaded research-paper PDFs for
> integrity/reproducibility signals — it verifies every citation against Crossref
> (flagging unverifiable or fabricated references), extracts statistical indicators
> (p-values, sample sizes, effect sizes), and produces a heuristic 0–100 integrity
> score. Currently single-user/local; evaluating whether to build it into a hosted
> multi-user product.

### (a) Competitive landscape

1. Produce a feature/pricing/positioning/funding comparison table (as of mid-2026) for:
   Elicit, Scite, Consensus, SciSpace, ResearchRabbit, Undermind, Semantic Scholar,
   Zotero, Paperpal, and any AI research assistants launched 2024–2026 that analyze
   uploaded PDFs. For each: core features, free-tier limits, paid pricing, target
   customer, funding/revenue signals, and whether they do any form of citation
   verification or paper-integrity analysis.
2. Which tools or services specifically verify a paper's reference list for fabricated,
   retracted, or unverifiable citations (against Crossref/OpenAlex/Retraction Watch)?
   Include academic prototypes, publisher-side screening tools (STM Integrity Hub,
   Clear Skies / Papermill Alarm, Proofig, ImageTwin, Signals), and anything targeting
   LLM-hallucinated citations. Is "citation integrity checking" a crowded, empty, or
   emerging niche?
3. Is "research integrity / reproducibility screening" a differentiated product wedge
   for a solo-built tool in 2026, or a feature incumbents (Scite, Elicit, publishers)
   already bundle or could trivially add? What do research-integrity experts say end
   users (individual researchers vs institutions) actually want from such tools?
4. What happened to tools that tried to score papers for quality/credibility (e.g.,
   scite's tallies, Altmetric, SciScore, statcheck)? What adoption, backlash, or
   accuracy criticism did they face, and what lessons apply to publishing a heuristic
   0–100 integrity score?

### (b) Business models

5. Who actually pays for research-paper analysis tools: individual researchers, labs
   (grant budgets), university libraries, publishers, or research-integrity offices?
   For each buyer type: typical willingness to pay, sales cycle, procurement path, and
   examples of tools successfully selling to them (with pricing).
6. Evaluate the B2B angle: publishers and journals screening submissions for integrity
   (papermills, fabricated references, statistical anomalies). Market size, existing
   vendors (Clear Skies, Proofig, ImageTwin, STM Integrity Hub members), what
   integration they require (editorial-manager plugins? API?), and whether a solo
   developer could realistically sell into this market vs the individual-researcher
   freemium market.
7. For a solo-developer academic-tools SaaS in 2026: compare freemium B2C
   (researchers), institutional/library site licenses, and open-core (free self-hosted
   + paid cloud). Realistic conversion rates, customer-acquisition costs for reaching
   researchers, examples of solo/indie academic SaaS that reached sustainability (or
   failed), and estimated total addressable market for "AI paper analysis" tools.

### (c) Legal / compliance

8. What is the legal exposure of operating a multi-tenant web service where users
   upload copyrighted research-paper PDFs for automated analysis? Cover: US fair use
   for text-and-data-mining (Authors Guild v. Google, HathiTrust, Thomson Reuters v.
   Ross Intelligence 2025), EU DSM Directive Articles 3/4 TDM exceptions, and the
   ResearchGate/Elsevier litigation precedent. Distinguish (a) private per-user storage
   + analysis only, from (b) any redistribution or display to other users.
9. If two users upload byte-identical PDFs, is storing one deduplicated copy and
   serving it to both users legally different from storing per-user private copies?
   Does content-addressed deduplication of copyrighted files create "distribution"
   liability? How do Dropbox-style services handle dedup of copyrighted content
   legally?
10. What does DMCA §512 safe harbor require for a small SaaS hosting user-uploaded
    PDFs: registered agent, repeat-infringer policy, takedown process, ToS language?
    What's the minimal compliant setup and cost for a solo operator? Equivalent EU
    obligations (DSA) for a micro-enterprise?
11. Do publisher terms of service (Elsevier, Springer Nature, Wiley) prohibit
    researchers from uploading legitimately-accessed PDFs to third-party analysis
    tools? Have publishers enforced this against tools like Zotero cloud storage,
    Paperpile, ReadCube, or AI assistants — and what happened?
12. GDPR obligations for a solo-run SaaS with EU academic users: uploaded papers
    containing personal data, account data, deletion/export rights, data-processing
    agreements with subprocessors (hosting, auth provider, LLM APIs), and whether a
    US-hosted solo operation can serve EU researchers compliantly without an EU entity.
13. What is the defamation / trade-libel / product-disparagement exposure of publishing
    an algorithmic "integrity score" (0–100, letter grade) on named, real, published
    papers — and implicitly their authors? Cover: US opinion privilege for scores based
    on disclosed methodology, relevant precedent (PubPeer litigation, Francesca Gino v.
    Data Colada, image-integrity tools' disclaimer practices), EU/UK differences (where
    defamation is easier to plead), and what disclaimers, framing ("transparency
    signals" vs "misconduct"), private-by-default results, and ToS language mitigate
    the risk.
14. Draft the checklist of legal documents and policies a solo-operated paper-analysis
    SaaS needs at beta launch: ToS clauses specific to user-uploaded copyrighted
    content and algorithmic scoring, privacy policy, acceptable-use policy, DMCA
    policy, and disclaimer language for heuristic scores. Point to good
    templates/examples from comparable academic tools.

### (d) Optional — architecture-relevant

15. If the tool only ingested open-access papers by URL/DOI (arXiv, Unpaywall, PubMed
    Central) instead of user uploads, how much of the copyright exposure disappears?
    Do OA licenses (CC-BY vs CC-BY-NC-ND) permit automated analysis and caching? *(The
    answer could materially change the storage/dedup design in §5 F1/F3.)*

## 8. Risk register / open questions

- **Score-publishing liability** is the single biggest non-technical risk (Q13). Until
  answered: results private-by-default, "transparency signals" framing everywhere, no
  public score pages.
- **Report reproducibility**: reports are immutable; `analyzer_version` exists and
  should be displayed in the UI with a re-run option. **Embeddings have no model
  identifier** — add one before any model swap, or similarity silently degrades across
  incomparable vector spaces.
- **Upstream API posture at scale**: today's semaphores (3 concurrent) are per-process
  politeness; a hosted server aggregates all users behind one IP. Needs a global
  outbound budget (F5's queue naturally serializes), a persistent reference-lookup
  cache in the DB instead of 24 h in-memory TTL, awareness of OpenAlex's daily cap, and
  optionally a Crossref Plus token.
- **LLM cost controls**: if chat/takeaways ever go public — hard per-user quotas,
  and never on a free tier at launch. The local-first/key-gated posture from the
  original vision survives.
- **Abuse surface of free uploads**: file-locker abuse, malware if PDFs are ever
  re-served (per-user privacy + presigned URLs mitigate), OCR CPU DoS (page caps exist;
  add per-job timeouts + daily ingest quotas). Managed auth's verified email is itself
  an abuse control.
- **Account deletion** must cascade PDFs, embeddings, and reports (GDPR-adjacent and
  basic trust); export is cheap goodwill.
- **Hygiene**: rotate the local OpenAI key (`.env` was never git-tracked — verified —
  but rotation is cheap).

## 9. Decision log

Decisions made 2026-07-22 (owner, interactive planning session):

1. **UI redesign owns the remaining focused window** (~2–3 weeks, through ~mid-Aug
   2026) — it pays off in every outcome and doesn't bet on an unvalidated direction.
2. **Deep research runs in parallel, starting immediately** — owner-run via Perplexity
   with the §7 list; not deferred, not blocking UI work.
3. **Multi-tenant architecture waits behind an explicit owner-reviewed go/no-go/pivot
   gate** keyed to the research returning. Auth wiring, Postgres migration, per-user
   isolation: none of it starts pre-gate.
4. **Managed auth** (Clerk/Auth0/Supabase-class; FastAPI verifies JWTs) when tenancy
   lands — speed and secure defaults over self-hosted identity.
5. **UI redesign on the current stack** — Vite/React/Tailwind/shadcn/React Query stay;
   IA, pages, and visual system are rebuilt; no Next.js migration.
6. **Original Week 3 Discovery UI is skipped** — no new components on the old design
   system immediately before a redesign.
7. **Roadmap is phased with owner checkpoints, not open-ended** — A0–A3 checkpoints,
   the research gate, and P1–P4 phase boundaries.
8. **TypeScript for all rebuilt frontend surfaces (`.tsx`)** — new and rebuilt
   components land in TS; untouched carried-over files (`airaApi.js`, `queries.js`,
   `reportTransform.js`) stay JS until they're next rewritten.

Surviving decisions from CLAUDE.md's long-term vision: local-first AI for core
intelligence (SPECTER/NLI keyless; LLM features key-gated), brute-force numpy cosine
until ~10k papers. Superseded: single-user-indefinitely; the original Week 3–5 plan
(Week 4 conflict detection returns as the NO-GO depth feature, with its original
owner-gated day-3 checkpoint).
