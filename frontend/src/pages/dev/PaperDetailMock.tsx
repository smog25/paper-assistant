import { Link } from 'react-router-dom'
import {
  ArrowLeft,
  CheckCircle2,
  AlertTriangle,
  XCircle,
  FolderPlus,
  Download,
  RefreshCw,
  ExternalLink,
  ChevronRight,
} from 'lucide-react'
import ThemeToggle from './ThemeToggle'

// A0 artifact: ONE static mocked screen — the PaperDetail redesign concept
// with hardcoded data. Standalone on purpose: the surrounding app shell is
// A1 scope; the slim bar below is a static placeholder, not the real shell.
// Nothing here touches production components. See frontend/DESIGN.md.

type Status = 'verified' | 'suspicious' | 'notfound'

interface MockCitation {
  ref: string
  matched: string | null
  doi: string | null
  confidence: number
  status: Status
}

const CITATIONS: MockCitation[] = [
  { ref: 'Nosek, B. A., et al. (2015). Promoting an open research culture.', matched: 'Promoting an open research culture', doi: '10.1126/science.aab2374', confidence: 0.97, status: 'verified' },
  { ref: 'Open Science Collaboration (2015). Estimating the reproducibility of psychological science.', matched: 'Estimating the reproducibility of psychological science', doi: '10.1126/science.aac4716', confidence: 0.95, status: 'verified' },
  { ref: 'Simmons, J. P., Nelson, L. D., & Simonsohn, U. (2011). False-positive psychology.', matched: 'False-Positive Psychology', doi: '10.1177/0956797611417632', confidence: 0.93, status: 'verified' },
  { ref: 'Button, K. S., et al. (2013). Power failure: why small sample size undermines reliability.', matched: 'Power failure: why small sample size undermines the reliability of neuroscience', doi: '10.1038/nrn3475', confidence: 0.91, status: 'verified' },
  { ref: 'Chen, L., & Alvarez, R. (2019). Replication attitudes among early-career researchers. J. Meta-Sci., 4(2).', matched: 'Replication attitudes in early career researchers: a survey', doi: '10.1027/2151-2604/a000389', confidence: 0.62, status: 'suspicious' },
  { ref: 'Whitfield, M. (2021). Preregistration in practice: a field review. Res. Integr. Q., 8, 112–130.', matched: 'Preregistration in practice', doi: null, confidence: 0.51, status: 'suspicious' },
  { ref: 'Harlan, D. T., & Voss, K. (2018). Statistical rigor and the replication economy. Ann. Behav. Data Sci., 12, 44–61.', matched: null, doi: null, confidence: 0.0, status: 'notfound' },
  { ref: 'Mercer, S. (2020). Meta-analytic drift in psychology: a ten-year audit. Rev. Quant. Psych., 3, 201–219.', matched: null, doi: null, confidence: 0.0, status: 'notfound' },
]

const SIGNALS = [
  { label: 'Citations verified', detail: '31 of 38', pct: 82, tone: 'ok' },
  { label: 'Statistics reported', detail: 'p, n, effect sizes, CIs', pct: 100, tone: 'ok' },
  { label: 'Limitations discussed', detail: 'section detected', pct: 100, tone: 'ok' },
  { label: 'Conflict of interest', detail: 'statement found', pct: 100, tone: 'ok' },
  { label: 'Open data', detail: 'no statement found', pct: 0, tone: 'warn' },
  { label: 'Pre-registration', detail: 'no statement found', pct: 0, tone: 'warn' },
] as const

const SIMILAR = [
  { title: 'Questionable research practices revisited: prevalence estimates 2011–2023', score: 0.91 },
  { title: 'The reproducibility of social-priming effects: a registered multi-lab report', score: 0.87 },
  { title: 'Effect-size inflation in underpowered designs: simulation evidence', score: 0.84 },
]

const RELATED = [
  'Many Labs 2: Investigating variation in replicability across samples',
  'Publication bias and the canonization of false facts',
  'Redefine statistical significance',
]

const STATUS_META: Record<Status, { label: string; Icon: typeof CheckCircle2; text: string; surface: string }> = {
  verified: { label: 'Verified', Icon: CheckCircle2, text: 'text-status-verified', surface: 'bg-status-verified-surface' },
  suspicious: { label: 'Suspicious', Icon: AlertTriangle, text: 'text-status-suspicious', surface: 'bg-status-suspicious-surface' },
  notfound: { label: 'Not found', Icon: XCircle, text: 'text-status-notfound', surface: 'bg-status-notfound-surface' },
}

function StatusBadge({ status }: { status: Status }) {
  const { label, Icon, text, surface } = STATUS_META[status]
  return (
    <span className={`inline-flex items-center gap-1 whitespace-nowrap rounded px-1.5 py-0.5 text-xs font-medium ${text} ${surface}`}>
      <Icon className="h-3 w-3" aria-hidden="true" />
      {label}
    </span>
  )
}

function ActionButton({ Icon, children }: { Icon: typeof Download; children: React.ReactNode }) {
  return (
    <button
      type="button"
      className="inline-flex h-8 items-center gap-1.5 rounded-md border border-border bg-card px-3 text-[13px] font-medium text-foreground transition-colors duration-150 hover:bg-muted focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
    >
      <Icon className="h-3.5 w-3.5" aria-hidden="true" />
      {children}
    </button>
  )
}

export default function PaperDetailMock() {
  return (
    <div className="min-h-screen bg-background text-foreground">
      {/* Static shell placeholder — the real app shell is A1 scope. */}
      <div className="border-b border-border bg-card">
        <div className="mx-auto flex h-12 max-w-6xl items-center justify-between px-6">
          <div className="flex items-center gap-6">
            <span className="text-sm font-semibold tracking-tight">AIRA</span>
            <nav className="flex items-center gap-4 text-[13px] text-muted-foreground" aria-label="placeholder navigation">
              <span className="text-foreground">Library</span>
              <span>Projects</span>
            </nav>
          </div>
          <div className="flex items-center gap-3">
            <ThemeToggle />
            <Link to="/styleguide" className="text-[13px] text-muted-foreground underline-offset-2 hover:underline">
              ← styleguide
            </Link>
          </div>
        </div>
      </div>

      <main className="mx-auto max-w-6xl px-6 py-6">
        {/* Breadcrumb */}
        <nav aria-label="Breadcrumb" className="flex items-center gap-1 text-[13px] text-muted-foreground">
          <span className="inline-flex items-center gap-1">
            <ArrowLeft className="h-3.5 w-3.5" /> Library
          </span>
          <ChevronRight className="h-3 w-3" />
          <span className="truncate text-foreground">Replication outcomes and reporting practices…</span>
        </nav>

        {/* Paper header — anatomy per Semantic Scholar: title → authors → metadata → actions */}
        <header className="mt-4 border-b border-border pb-5">
          <h1 className="max-w-3xl text-lg font-semibold leading-snug">
            Replication outcomes and reporting practices in social psychology: a five-year
            cross-journal audit
          </h1>
          <p className="mt-1.5 text-sm text-muted-foreground">
            R. Okafor, J. Lindqvist, M. Tanaka, S. Beaumont
          </p>
          <div className="mt-2 flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-muted-foreground">
            <span className="tabular-nums">2024</span>
            <a href="#" className="inline-flex items-center gap-1 underline-offset-2 hover:underline">
              10.1000/mock.2024.10482 <ExternalLink className="h-3 w-3" />
            </a>
            <span className="tabular-nums">8,431 words</span>
            <span className="tabular-nums">22 pages</span>
            <span>
              analyzed 2026-07-22 · <span className="tabular-nums">analyzer v1.0.0</span>
            </span>
          </div>
          <div className="mt-4 flex items-center gap-2">
            <ActionButton Icon={FolderPlus}>Add to project</ActionButton>
            <ActionButton Icon={Download}>Export report</ActionButton>
            <ActionButton Icon={RefreshCw}>Re-run analysis</ActionButton>
          </div>
        </header>

        <div className="mt-6 grid grid-cols-1 gap-6 lg:grid-cols-[1fr_320px]">
          {/* Main column */}
          <div className="min-w-0 space-y-6">
            {/* Citations */}
            <section aria-labelledby="citations-h">
              <div className="flex items-baseline justify-between">
                <h2 id="citations-h" className="text-base font-medium">Citations</h2>
                <p className="text-xs text-muted-foreground">
                  <span className="text-status-verified tabular-nums">31 verified</span>
                  {' · '}
                  <span className="text-status-suspicious tabular-nums">4 suspicious</span>
                  {' · '}
                  <span className="text-status-notfound tabular-nums">3 not found</span>
                  {' · showing 8 of 38'}
                </p>
              </div>
              <div className="mt-2 overflow-hidden rounded-md border border-border">
                <table className="w-full border-collapse text-[13px]">
                  <thead>
                    <tr className="border-b border-border bg-muted/50 text-left text-xs text-muted-foreground">
                      <th scope="col" className="px-3 py-2 font-medium">Status</th>
                      <th scope="col" className="px-3 py-2 font-medium">Reference</th>
                      <th scope="col" className="hidden px-3 py-2 font-medium md:table-cell">Crossref match</th>
                      <th scope="col" className="px-3 py-2 text-right font-medium" aria-sort="descending">Conf.</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-border">
                    {CITATIONS.map((c, i) => (
                      <tr key={i} className="align-top hover:bg-muted/40">
                        <td className="px-3 py-2"><StatusBadge status={c.status} /></td>
                        <td className="max-w-[320px] px-3 py-2 leading-snug text-foreground">{c.ref}</td>
                        <td className="hidden max-w-[220px] px-3 py-2 leading-snug text-muted-foreground md:table-cell">
                          {c.matched ? (
                            <span className="inline-flex items-start gap-1">
                              {c.matched}
                              {c.doi && <ExternalLink className="mt-0.5 h-3 w-3 shrink-0" />}
                            </span>
                          ) : (
                            <span aria-hidden="true">—</span>
                          )}
                        </td>
                        <td className="px-3 py-2 text-right tabular-nums">
                          {c.confidence > 0 ? c.confidence.toFixed(2) : '—'}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </section>

            {/* Statistics */}
            <section aria-labelledby="stats-h">
              <h2 id="stats-h" className="text-base font-medium">Statistics</h2>
              <div className="mt-2 grid grid-cols-2 gap-3 sm:grid-cols-4">
                {[
                  { label: 'p-values', value: '14', note: '3 near .05' },
                  { label: 'Sample sizes', value: '6', note: 'n = 42 – 1,283' },
                  { label: 'Effect sizes', value: '9', note: "Cohen's d, η²" },
                  { label: 'Conf. intervals', value: '7', note: '95% CI' },
                ].map((s) => (
                  <div key={s.label} className="rounded-md border border-border bg-card px-3 py-2.5">
                    <p className="text-xs text-muted-foreground">{s.label}</p>
                    <p className="mt-0.5 text-2xl font-semibold tabular-nums leading-tight">{s.value}</p>
                    <p className="text-xs text-muted-foreground tabular-nums">{s.note}</p>
                  </div>
                ))}
              </div>
              <div className="mt-3 space-y-2">
                <div className="flex items-start gap-2 rounded-md bg-status-suspicious-surface px-3 py-2 text-[13px]">
                  <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0 text-status-suspicious" aria-hidden="true" />
                  <p>
                    <span className="font-medium text-status-suspicious">Flagged for review:</span>{' '}
                    3 of 14 p-values cluster between 0.04 and 0.05 (<span className="tabular-nums">p = .048, .043, .041</span>).
                    A pattern worth checking, not a verdict.
                  </p>
                </div>
                <div className="flex items-start gap-2 rounded-md bg-status-verified-surface px-3 py-2 text-[13px]">
                  <CheckCircle2 className="mt-0.5 h-3.5 w-3.5 shrink-0 text-status-verified" aria-hidden="true" />
                  <p>
                    <span className="font-medium text-status-verified">Good practice:</span>{' '}
                    effect sizes accompany every reported test; limitations and COI sections present.
                  </p>
                </div>
              </div>
            </section>
          </div>

          {/* Right rail */}
          <aside className="space-y-6">
            {/* Transparency signals */}
            <section aria-labelledby="signals-h" className="rounded-md border border-border bg-card p-4">
              <h2 id="signals-h" className="text-base font-medium">Transparency signals</h2>
              <div className="mt-3 flex items-center gap-3">
                <span className="inline-flex h-12 w-12 items-center justify-center rounded-md bg-grade-b text-xl font-semibold text-white">
                  B
                </span>
                <div>
                  <p className="text-2xl font-semibold tabular-nums leading-none">78<span className="text-sm text-muted-foreground">/100</span></p>
                  <p className="mt-1 text-xs text-muted-foreground">heuristic score · not peer review</p>
                </div>
              </div>
              <ul className="mt-4 space-y-2.5">
                {SIGNALS.map((s) => (
                  <li key={s.label} className="text-[13px]">
                    <div className="flex items-baseline justify-between gap-2">
                      <span>{s.label}</span>
                      <span className="text-xs text-muted-foreground tabular-nums">{s.detail}</span>
                    </div>
                    <div className="mt-1 h-1 overflow-hidden rounded-full bg-muted" role="presentation">
                      <div
                        className={`h-full rounded-full ${s.tone === 'ok' ? 'bg-status-verified' : 'bg-status-suspicious'}`}
                        style={{ width: `${Math.max(s.pct, 4)}%` }}
                      />
                    </div>
                  </li>
                ))}
              </ul>
              <p className="mt-4 border-t border-border pt-3 text-xs leading-relaxed text-muted-foreground">
                Automated heuristics over open-science markers. Signals for your judgment —
                never a verdict on the work or its authors.
              </p>
            </section>

            {/* Similar in library */}
            <section aria-labelledby="similar-h">
              <h2 id="similar-h" className="text-base font-medium">Similar in your library</h2>
              <ul className="mt-2 divide-y divide-border rounded-md border border-border bg-card">
                {SIMILAR.map((p) => (
                  <li key={p.title} className="flex items-baseline justify-between gap-3 px-3 py-2.5">
                    <span className="text-[13px] leading-snug">{p.title}</span>
                    <span className="text-xs text-muted-foreground tabular-nums">{p.score.toFixed(2)}</span>
                  </li>
                ))}
              </ul>
            </section>

            {/* Related via OpenAlex */}
            <section aria-labelledby="related-h">
              <h2 id="related-h" className="text-base font-medium">Related works <span className="text-xs font-normal text-muted-foreground">via OpenAlex</span></h2>
              <ul className="mt-2 divide-y divide-border rounded-md border border-border bg-card">
                {RELATED.map((t) => (
                  <li key={t} className="flex items-center justify-between gap-3 px-3 py-2.5">
                    <span className="text-[13px] leading-snug">{t}</span>
                    <ExternalLink className="h-3 w-3 shrink-0 text-muted-foreground" aria-hidden="true" />
                  </li>
                ))}
              </ul>
            </section>
          </aside>
        </div>

        <footer className="mt-10 border-t border-border pt-4 text-xs text-muted-foreground">
          Mocked screen — hardcoded data, A0 checkpoint artifact. Production PaperDetail is untouched.
        </footer>
      </main>
    </div>
  )
}
