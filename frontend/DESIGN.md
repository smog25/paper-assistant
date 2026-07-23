# AIRA Design Direction — Track A / A0 artifact

> Written 2026-07-22 for the A0 checkpoint (see VISION.md §4). Reference notes are
> knowledge-based descriptions of these products' design languages; glance at the live
> products during checkpoint review to confirm nothing has shifted materially.

## References — what we take, what we leave

**Elicit** — the workspace IA benchmark. Opens directly into work (no marketing funnel
once signed in), tables are the primary surface, restrained neutral palette where color
only appears when it means something. *Take:* library-first entry, table-centric
density, chrome stays out of the data's way. *Leave:* its occasionally sparse feel —
AIRA's report views need more visual hierarchy than a flat grid.

**Scite** — the closest domain analog. Its supporting/mentioning/contrasting badge
system proves a three-state evidence taxonomy can carry an entire product's visual
identity. *Take:* status-as-color-system discipline — our verified/suspicious/not_found
maps 1:1; badges always pair icon + label, never color alone. *Leave:* report pages that
lean cluttered; we keep a stricter grid.

**Semantic Scholar** — the paper-detail hierarchy benchmark: title → authors → venue/
year → action row → tabbed content, with metadata in quiet muted type. *Take:* that
exact header anatomy for PaperDetail; quiet metadata typography. *Leave:* the dated
visual skin.

**Linear** — the density/token discipline benchmark: 13–14px UI type, semantic color
variables everywhere, borders over shadows, first-class dark mode, motion that's nearly
invisible. *Take:* type scale, token rigor, border-based elevation, dark-mode parity as
a requirement not an afterthought. *Leave:* app-wide keyboard-command surface (later,
not Track A).

## Direction in one line

**A calm instrument, not a dashboard demo**: neutral surfaces, near-zero decoration,
where the only saturated color on screen is a verification status or a grade — so color
always *means* something.

## Token system (implemented in `src/index.css` + `tailwind.config.js`)

- **Status colors** (`--status-verified|suspicious|notfound`, each with `-bg` surface
  variant, light + dark values): the single source for citation-verification state.
  Tailwind: `text-status-verified`, `bg-status-verified-surface`, etc.
- **Grade colors** (`--grade-a` … `--grade-f`, light + dark): the single source for
  integrity grades. Tailwind: `text-grade-a`, `bg-grade-a`, etc. `src/lib/grades.js`
  keeps serving old components until each is rebuilt, then dies.
- **Type scale** (use these, nothing else): 12px metadata/captions · 13px table body &
  dense UI (`text-[13px]` pending a named token) · 14px default body · 16px section
  heads · 18px page titles · 24px display numbers (scores). All numeric data uses
  `tabular-nums`.
- **Spacing**: Tailwind's 4px grid only; component-internal padding 8/12/16, section
  gaps 16/24.
- **Radius**: existing `--radius` (0.5rem) unchanged; no per-component overrides.
- **Elevation**: 1px `border-border` is the default separator; shadows only for
  overlays (popover/dialog). No decorative shadows on cards.
- **Motion**: 150ms ease-out on interactive state changes only. No entrance
  animations, no staggered reveals, no framer-motion in rebuilt surfaces.
- **Dark mode**: every token defined in both `:root` and `.dark` from day one;
  ThemeProvider (next-themes) is active with default `light`; toggle lives in the
  styleguide until the A1 shell lands.
- **A11y floor**: AA contrast for all token pairs, visible focus rings, status never
  conveyed by color alone (icon + text), `tabular-nums` + right-alignment for numeric
  columns, `reduced-motion` respected by keeping motion trivial.
- **Language**: scores are "transparency signals", never verdicts. UI copy says
  "unverified", not "fake"; "flagged", not "wrong".

## What exists after A0

- `/styleguide` — tokens rendered live (colors, type, status/grade badges, controls),
  standalone (outside the old Layout), with theme toggle.
- `/styleguide/paper` — ONE static mocked screen: the PaperDetail ("money screen")
  redesign concept with hardcoded data. Header anatomy per Semantic Scholar; right rail
  with transparency-signals card; dense citations table per Elicit/Linear.
- No production page has been touched. The mock intentionally renders standalone —
  the surrounding app shell is A1 scope.

## Open questions for the A0 checkpoint

1. **Accent color**: current proposal keeps `--primary` neutral (near-black / white)
   so status+grade colors are the only chroma. Alternative: one restrained accent
   (e.g., a deep indigo) for primary actions. Decide before A1.
2. **Title typography**: all-sans (current mock) vs a serif for paper titles only
   (more "scholarly", riskier to keep consistent).
3. **A1 shell shape**: slim top bar (current mock's assumption) vs left sidebar
   (scales better if Discover/Chat tabs return post-gate).
