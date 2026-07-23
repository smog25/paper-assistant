import { Link } from 'react-router-dom'
import { CheckCircle2, AlertTriangle, XCircle, ArrowRight } from 'lucide-react'
import { Button as ButtonRaw } from '@/components/ui/button'
import { Input as InputRaw } from '@/components/ui/input'
import ThemeToggle from './ThemeToggle'

// The ui/ primitives are still .jsx (untyped forwardRef), so their props fail
// strict TSX checking. Cast here only — they get real types when the ui layer
// migrates to TS during the page rebuilds (A1+).
const Button = ButtonRaw as React.ComponentType<Record<string, unknown>>
const Input = InputRaw as React.ComponentType<Record<string, unknown>>

// A0 artifact: the token system rendered live. Standalone dev route —
// deliberately outside the old Layout shell. See frontend/DESIGN.md.

const STATUS = [
  { key: 'verified', label: 'Verified', Icon: CheckCircle2, text: 'text-status-verified', surface: 'bg-status-verified-surface' },
  { key: 'suspicious', label: 'Suspicious', Icon: AlertTriangle, text: 'text-status-suspicious', surface: 'bg-status-suspicious-surface' },
  { key: 'notfound', label: 'Not found', Icon: XCircle, text: 'text-status-notfound', surface: 'bg-status-notfound-surface' },
] as const

const GRADES = [
  { g: 'A', cls: 'bg-grade-a' },
  { g: 'B', cls: 'bg-grade-b' },
  { g: 'C', cls: 'bg-grade-c' },
  { g: 'D', cls: 'bg-grade-d' },
  { g: 'F', cls: 'bg-grade-f' },
] as const

const TYPE_SCALE = [
  { px: '12px', cls: 'text-xs', use: 'metadata, captions, table headers' },
  { px: '13px', cls: 'text-[13px]', use: 'table body, dense UI (default for data surfaces)' },
  { px: '14px', cls: 'text-sm', use: 'body text, form controls' },
  { px: '16px', cls: 'text-base font-medium', use: 'section headings' },
  { px: '18px', cls: 'text-lg font-semibold', use: 'page titles' },
  { px: '24px', cls: 'text-2xl font-semibold', use: 'display numbers (scores)' },
] as const

const SURFACES = [
  { name: 'background', cls: 'bg-background' },
  { name: 'card', cls: 'bg-card' },
  { name: 'muted', cls: 'bg-muted' },
  { name: 'border', cls: 'bg-border' },
] as const

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="space-y-3">
      <h2 className="text-base font-medium text-foreground">{title}</h2>
      {children}
    </section>
  )
}

export function StatusBadge({ status }: { status: (typeof STATUS)[number] }) {
  const { label, Icon, text, surface } = status
  return (
    <span className={`inline-flex items-center gap-1.5 rounded-md px-2 py-0.5 text-[13px] font-medium ${text} ${surface}`}>
      <Icon className="h-3.5 w-3.5" aria-hidden="true" />
      {label}
    </span>
  )
}

export default function Styleguide() {
  return (
    <div className="min-h-screen bg-background text-foreground">
      <div className="mx-auto max-w-4xl space-y-10 px-6 py-10">
        <header className="flex items-start justify-between border-b border-border pb-6">
          <div>
            <h1 className="text-lg font-semibold">AIRA styleguide</h1>
            <p className="mt-1 text-sm text-muted-foreground">
              A0 token system, rendered live. Direction and rules: <code className="text-xs">frontend/DESIGN.md</code>
            </p>
          </div>
          <div className="flex items-center gap-3">
            <ThemeToggle />
            <Link
              to="/styleguide/paper"
              className="inline-flex h-8 items-center gap-1.5 rounded-md bg-primary px-3 text-[13px] font-medium text-primary-foreground transition-colors duration-150 hover:opacity-90 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
            >
              Mocked screen <ArrowRight className="h-3.5 w-3.5" />
            </Link>
          </div>
        </header>

        <Section title="Core surfaces">
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
            {SURFACES.map((s) => (
              <div key={s.name} className="space-y-1.5">
                <div className={`h-14 rounded-md border border-border ${s.cls}`} />
                <p className="text-xs text-muted-foreground">{s.name}</p>
              </div>
            ))}
          </div>
          <p className="text-xs text-muted-foreground">
            Elevation is a 1px border, not a shadow. Shadows are reserved for overlays.
          </p>
        </Section>

        <Section title="Status — citation verification state">
          <div className="flex flex-wrap items-center gap-3">
            {STATUS.map((s) => (
              <StatusBadge key={s.key} status={s} />
            ))}
          </div>
          <p className="text-xs text-muted-foreground">
            The only saturated color in data surfaces. Never color alone: icon + label always.
            Copy says “unverified” / “flagged”, never “fake” / “wrong”.
          </p>
        </Section>

        <Section title="Grades — integrity letter grade">
          <div className="flex items-center gap-2">
            {GRADES.map(({ g, cls }) => (
              <span
                key={g}
                className={`inline-flex h-9 w-9 items-center justify-center rounded-md text-sm font-semibold text-white ${cls}`}
              >
                {g}
              </span>
            ))}
          </div>
        </Section>

        <Section title="Type scale">
          <div className="divide-y divide-border rounded-md border border-border">
            {TYPE_SCALE.map((t) => (
              <div key={t.px} className="flex items-baseline gap-6 px-4 py-2.5">
                <span className="w-10 shrink-0 text-xs tabular-nums text-muted-foreground">{t.px}</span>
                <span className={t.cls}>Citation integrity, measured.</span>
                <span className="ml-auto text-right text-xs text-muted-foreground">{t.use}</span>
              </div>
            ))}
          </div>
          <div className="flex items-center gap-6 rounded-md border border-border px-4 py-2.5">
            <span className="text-xs text-muted-foreground">tabular-nums</span>
            <div className="text-[13px] tabular-nums">
              <div className="text-right">1,024 words · p = 0.048 · n = 1,283</div>
              <div className="text-right">941 words · p = 0.007 · n = 86</div>
            </div>
            <span className="ml-auto text-xs text-muted-foreground">all numeric data, right-aligned in columns</span>
          </div>
        </Section>

        <Section title="Controls (shadcn primitives under the tokens)">
          <div className="flex flex-wrap items-center gap-3">
            <Button size="sm">Primary</Button>
            <Button size="sm" variant="secondary">Secondary</Button>
            <Button size="sm" variant="outline">Outline</Button>
            <Button size="sm" variant="ghost">Ghost</Button>
            <Button size="sm" variant="destructive">Destructive</Button>
            <Input placeholder="Search papers…" className="h-8 w-56 text-[13px]" />
          </div>
        </Section>

        <Section title="Motion policy">
          <p className="text-sm text-muted-foreground">
            150ms ease-out on interactive state changes only. No entrance animations, no staggered
            reveals, no decorative motion. (This page contains none.)
          </p>
        </Section>
      </div>
    </div>
  )
}
