# ChunkHound Design System

Agent-executable brand specification. All values are final computed tokens — apply directly, no interpretation needed.

## Brand Identity

- **Wordmark:** "ChunkHound" in Inter 700, trailing cyan accent dot (circle, `#22d3ee`)
- **Headline:** "Enterprise-scale engineering research. On your laptop." (source: `site/src/lib/positioning.json`)
- **Personality:** Exciting + likeable. Cutting-edge but approachable.
- **PAD Profile:** Pleasure High, Arousal High, Dominance Medium
- **Logo:** SVG service dog illustration. Min clearspace = logo width around all sides. Use on `--brand-surface` or the light page surface (`--bg-page` in light mode) only.
- **Brand motif:** the hound mark may also be used as a decorative watermark on `--brand-surface` — cropped, masked, and at or below 12% opacity. It is then a texture, never a lockup: the clearspace rule and the full-opacity usage do not apply, but it must stay `aria-hidden`, non-interactive, and clear of body copy.

## Color Tokens

### Primary — Cyan / Teal

| Shade | Hex |
|-------|---------|
| 50 | `#ecfeff` |
| 100 | `#cffafe` |
| 200 | `#a5f3fc` |
| 300 | `#67e8f9` |
| 400 | `#22d3ee` |
| 500 | `#0891b2` |
| 600 | `#0e7490` |
| 700 | `#155e75` |
| 800 | `#164e63` |
| 900 | `#083344` |
| 950 | `#06262f` |

### Neutral — achromatic gray

| Shade | Hex |
|-------|---------|
| 50 | `#f4f4f4` |
| 100 | `#e8e8e8` |
| 200 | `#d3d3d3` |
| 300 | `#b7b7b7` |
| 400 | `#9b9b9b` |
| 500 | `#808080` |
| 600 | `#666666` |
| 700 | `#4f4f4f` |
| 800 | `#3c3c3c` |
| 900 | `#2b2b2b` |
| 950 | `#212121` |

### Semantic

| Role | Light | Dark |
|---------|---------|---------|
| Error | `#a93c39` | `#e7736d` |
| Warning | `#8d5906` | `#c98e43` |
| Success | `#097a45` | `#51b278` |

Semantic backgrounds: color at `12%` opacity (light) / `18%` opacity (dark). Semantic borders: color at `20%` opacity.

### Code Surface Accent

`--code-accent: #22d3ee` — Cyan accent for dark code panels in both themes. Use instead of `--primary` when content appears on `--code-bg`.

`--code-muted: #9b9b9b` — Muted text for dark code panels. Use instead of page text tokens on `--code-bg`.

### Brand Surface

`--brand-surface: #21221e` — the sanctioned backdrop for the hound mark and the page's closing section. It is **theme-invariant**: the page's dark bookend must not move with the OS theme. Its companions (`--on-brand-surface`, `--on-brand-muted`, `--brand-cta`, `--on-brand-cta`, `--brand-border`) are invariant for the same reason, and are only legal on a brand surface. The brand CTA pair (`--brand-cta` fill, `--on-brand-cta` text) is deliberately not `--primary`: `--primary` inverts between themes, which would flip the same button from dull teal to bright cyan on the same surface. So the brand surface takes the fixed brand pair (`.btn--brand`) and a page surface takes the theme-adaptive `--primary`/`--on-primary` pair (`.btn--page`).

### Light Mode

| Token | Value |
|-------|-------|
| `--bg-page` | `#f4f4f4` (neutral-50) |
| `--bg-surface` | `#ffffff` |
| `--bg-muted` | `#e8e8e8` (neutral-100) |
| `--text-primary` | `#212121` (neutral-950) |
| `--text-secondary` | `#5f5f5f` |
| `--text-tertiary` | `#676767` |
| `--text-muted` | `#707070` |
| `--border-color` | `#d3d3d3` (neutral-200) |
| `--border-subtle` | `#e8e8e8` (neutral-100) |
| `--primary` | `#0e7490` (primary-600) |
| `--primary-bright` | `#22d3ee` (primary-400) |
| `--primary-bg` | `#cffafe` (primary-100) |
| `--on-primary` | `#ffffff` |
| `--link` | `#0e7490` (primary-600) |
| `--link-hover` | `#155e75` (primary-700) |
| `--link-visited` | `#164e63` (primary-800) |
| `--link-on-primary-bg` | `#155e75` (primary-700) |
| `--code-bg` | `#2b2b2b` (neutral-900) |
| `--code-text` | `#d3d3d3` (neutral-200) |
| `--code-muted` | `#9b9b9b` (neutral-400) |
| `--code-accent` | `#22d3ee` (primary-400) |

### Dark Mode

| Token | Value |
|-------|-------|
| `--bg-page` | `#2b2b2b` (neutral-900) |
| `--bg-surface` | `#3c3c3c` (neutral-800) |
| `--bg-muted` | `#4f4f4f` (neutral-700) |
| `--text-primary` | `#f4f4f4` (neutral-50) |
| `--text-secondary` | `#c2c2c2` |
| `--text-tertiary` | `#aaaaaa` |
| `--text-muted` | `#949494` |
| `--border-color` | `#4f4f4f` (neutral-700) |
| `--border-subtle` | `#3c3c3c` (neutral-800) |
| `--primary` | `#22d3ee` (primary-400) |
| `--primary-bright` | `#0891b2` (primary-500) |
| `--primary-bg` | `#164e63` (primary-800) |
| `--on-primary` | `#083344` (primary-900) |
| `--link` | `#22d3ee` (primary-400) |
| `--link-hover` | `#a5f3fc` (primary-200) |
| `--link-visited` | `#cffafe` (primary-100) |
| `--link-on-primary-bg` | `#a5f3fc` (primary-200) |
| `--code-bg` | `#212121` (neutral-950) |
| `--code-text` | `#d3d3d3` (neutral-200) |
| `--code-muted` | `#9b9b9b` (neutral-400) |
| `--code-accent` | `#22d3ee` (primary-400) |

### Code Syntax Highlighting

| Role | Light | Dark |
|------|-------|------|
| Keyword / accent | `#0e7490` | `#22d3ee` |
| String | `#155e75` | `#ecfeff` |
| Comment | `#9b9b9b` | `#666666` |
| Inline code bg | `#cffafe` | `#164e63` |
| Inline code text | `--on-primary-bg` (`--text-primary`) | `--on-primary-bg` (`--text-primary`) |
| Search match bg | `rgba(14,116,144,0.15)` (light) / `rgba(34,211,238,0.2)` (dark) |
| Search match text | `#0e7490` (light) / `#22d3ee` (dark) |

## Typography

### Font Stack

| Role | Family | Fallback |
|------|--------|----------|
| UI / Body | Inter | system-ui, -apple-system, sans-serif |
| Code | JetBrains Mono | monospace |

### Type Scale — Major Third (1.250), 16px base, 4px grid snap

| Token | Size | Line Height | Weight | Use |
|-------|------|-------------|--------|-----|
| `--text-4xl` | 31–49px fluid | 1.22 | 700 | Page title |
| `--text-3xl` | 28–39px fluid | 1.23 | 700 | Major heading |
| `--text-2xl` | 25–31px fluid | 1.16 | 600 | Section heading |
| `--text-xl` | 25px | 32px | 600 | Subheading |
| `--text-lg` | 20px | 32px | 500 | Lead paragraph |
| `--text-base` | 16px | 24px | 400 | Body text |
| `--text-sm` | 13px | 20px | 400 | Caption, metadata |
| `--text-xs` | 10px | 16px | 600 | Label, overline |

Display tiers (`4xl`–`2xl`) are **fluid**: `clamp(min, rem + vw, max)` ramps them
across a 320→1440px viewport, then the `>=1600px` block steps the tokens up. The
`rem` term keeps browser zoom working (WCAG 1.4.4 — pure `vw` fails it); the unitless
line heights track the fluid size. Body tiers stay fixed on the 4px grid.
Every font size comes from these tokens: no raw `px`/`rem` values outside the
code and terminal surfaces (see Code Blocks). If a size is missing, add a tier —
never hardcode one.

### Weight Usage

| Weight | Role |
|--------|------|
| 400 | Body, caption |
| 500 | Lead text, body emphasis |
| 600 | Headings (2xl, xl), labels, section titles, controls / actions |
| 700 | Display headings (3xl, 4xl), wordmark |

### Heading Weight Mapping

Element mapping of the roles above, declared once globally — a heading must
never fall through to the browser's `bold` default:

| Element | Weight |
|---------|--------|
| `h1` | 700 |
| `h2`, `h3`, `h4` | 600 |

### Section Title Pattern

Uppercase, `--text-xs`, weight 600, letter-spacing `0.12em`, color `--text-tertiary`.

## Spatial System

### Spacing — 4px base, hybrid geometric

| Token | Value |
|-------|-------|
| `--space-0` | 0 |
| `--space-1` | 4px |
| `--space-2` | 8px |
| `--space-3` | 12px |
| `--space-4` | 16px |
| `--space-5` | 24px |
| `--space-6` | 32px |
| `--space-7` | 40px |
| `--space-8` | 48px |
| `--space-9` | 64px |
| `--space-10` | 80px |

### Gestalt Proximity

- Within component: `space-2` (8px)
- Within group: `space-4` (16px)
- Between groups: `space-5`–`space-6` (24–32px)
- Between sections: `space-8`–`space-10` (48–80px)

Ratio between adjacent tiers must be >= 2x for grouping to be perceptible.

### Border Radius — High pleasure, 6px base

| Token | Value |
|-------|-------|
| `--radius-none` | 0 |
| `--radius-sm` | 3px |
| `--radius-md` | 6px |
| `--radius-control` | `var(--radius-md)` (6px) |
| `--radius-lg` | 9px |
| `--radius-xl` | 12px |
| `--radius-2xl` | 18px |
| `--radius-full` | 9999px (pill) |

**Semantic radius overrides:**
- Error/warning alerts: `--radius-sm` (sharper = threat congruent)
- Success alerts: `--radius-lg` (rounder = positive valence)
- Controls: `--radius-control` — the ONE source for inputs, selects, toggles, and option rows
- Buttons: `--radius-full` (pill) — see Component Patterns → Buttons
- Cards/panels: `--radius-xl`
- Code blocks: `--radius-md`
- Inputs/search bars: `--radius-control`
- Pills/tags: `--radius-full`

Focus is a state, not a shape: a focus state never sets geometry (no radius, width,
or height). A focus rule that writes geometry silently reshapes whichever control it
lands on, which is how the nav toggles got squared and the CTA needed a radius pin.

### Border Weight — Medium dominance

| Token | Width | Use |
|-------|-------|-----|
| `--border-w-subtle` | 1px (low opacity) | Separators, inner lines |
| `--border-w-default` | 1px | Cards, inputs, table rules |
| `--border-w-emphasis` | 2px | Active/focus states, hero callout |
| `--border-w-strong` | 2px (high contrast) | Section dividers |
| `--border-w-accent` | 3px | Accent indicators, progress bars |

Max ~4 visible borders per viewport section. Prefer spacing and surface tone over borders.

### Target Sizes — WCAG compliant (Fitts' law)

| Token | Height | Horizontal Padding | Level |
|-------|--------|--------------------|-------|
| `--target-sm` | 32px | 16px | AA |
| `--target-md` | 40px | 24px | AA |
| `--target-lg` | 48px | 32px | AAA |
| `--target-xl` | 56px | 48px | AAA |

### Layout

- Max content width: `1440px`, centered — one shell width for nav, docs, and
  landing sections, all via `--container-max`; display type is fluid below the
  ramp's 1440px cap and the scale grows >=1600px
- Section padding: `space-10` vertical, `space-6` horizontal
- Grid gap: `space-5` (24px) default
- Container: `max-width: var(--container-max); margin: 0 auto`
- Large desktops (>=1600px): type scale tokens increase ~118% for viewing distance
- Hero: one full-bleed terminal surface (`--code-bg`) carrying a single **centered column** — the `h1` leads that column (typography only: no eyebrow, no subheadline, no logo; the running demo is the explanation), then the install action, the run's **window** — a hairline frame (a theme-stable code token: the surface is theme-stable, so the frame must be; flat, never a shadow) around the titlebar (the decorative session-dot seam plus the session control, chrome bracketed to the run's own measure) with the live run and the fixed harness composer, so the frame ends where the copy ends and the terminal reads as a window on the surface. The column IS the terminal's measure plus its inline gutter, centered in the section: the frame's edges are the content edges, so the terminal never leaves its slack on one side. The hero band is the surface the window sits on: no separate intro band, and the window's separation is its frame plus the surface left around it — never a shadow or a second tone (an annotation already steps up off `--code-bg`, and that step is a contrast ceiling). Each engine beat's copy is an **annotation over the run** (`.anchor` holds its beat's `.line.call` + `.line.evidence` and then the `aside.note` that annotates them, so DOM order IS the visual order: call → receipt → note), so reading order and screen-reader order match and the no-JS render is the same DOM. Placement is CSS's, keyed to the card's inline size, never a script's: the note hangs inline below its call + receipt at every card width, railed, connected up to the receipt by the rail continued through the block-start gap, and indented to the copy column (past the role + glyph columns, both derived from the run's own `--hero-role-col` token) — capped below the run's own measure so it never stretches into a band. One width tier is keyed to the card, not the viewport: **narrow** (<=479px card) collapses the transcript's visible 10ch role column (role text stays sr-only; the seam's glyph key carries the mapping) and the composer's. An annotation is marked by grammar the run never uses: a **3px rail** that brackets the note (leading edge plus bottom edge) and **draws itself in as the beat is raised** — the leading edge strokes down from the receipt (the note's copy revealing with it), then the bottom edge fills left-to-right like a progress bar for the beat's own lifetime — then rests invisible, taking the accent only while its beat is live, an **indent** out of the run's measure column, a small mono **ordinal chip**, and **Inter prose** against the terminal's monospace — on `--code-note-bg`, so the annotation reads as a layer above the run, not a matching block inside it. The conditions and the interfaces moved to the scale section as a **terms rail** (see Layout): a note's `data-condition` still maps each beat to the value prop it demonstrates, but nothing in the hero tracks the run. (This reverses four earlier hero decisions — the removed leading rail, the shared band/note surface, the flush no-offset note, and the removed pointing leaders — deliberately; do not re-add the old grammar silently. The margin placement, the in-row band and the band ordinal chip were then dropped too: the note is inline at every width, so the terminal rows carry no highlight band and the rail is the only live mark.) Reduced motion and JS-off render the whole run and all three notes at once, unemphasised. The brand lockup is not repeated in the hero — the nav wordmark carries it — but the CTA does carry the hound as a decorative **micro-mark**: an `aria-hidden` mask of `logo-light.svg` filled with `--code-text` at 1.25em, leading `Build your hound`. The mark sits **outside the link** in its own `nowrap` cluster and holds a fixed tone — no hover state, no hit area — so it stays decoration beside the control, never part of it. It is a mark, not the lockup, so the inline clearspace rule does not apply (as with the watermark and the diagram's `layer-mark`); the words carry the accessible name, and the "no logo" clause above is superseded for this one element.
- Scale section (`#scale`): the origin story and the proof receipt (`.proof-grid`), then the engine lede ("…a single engine, running on your machine…"). Below that lede sits the **terms rail** — the conditions (free & open-source · local-first · nothing leaves your machine · your models, your bill) then the interfaces (CLI · MCP · GitHub + the live star count). Icon-led and chip-free, so a promise reads as a term, never as a control; the 4+3 seam is the section's one hairline grammar — a 1px `|` at the text's height between the two groups, 16px either side against 8px between items — dropped in the tier where the section's rows stack, exactly as the receipt drops its column dividers; two wrap units (the conditions, then the interfaces + repo) peel apart rather than splitting the interface pair; always lit (nothing tracks the run); marketing-surface tokens only (`--primary` accents at 16px, labels at the body tier — `--text-base`/500/`--text-secondary`, below the section's lead prose and above its labels — with a `--text-tertiary` star count). Relocated from the hero, where the window alone now closes the surface.
- Alignment: one left-anchored system after the hero. Section intros (eyebrow + `h2` + body), stats receipts, quotes, and the footer's claims share the hero copy column's left edge — centered section headers are retired. Center only short, isolated accent bars (<=2 lines); never alternate alignment section-to-section.
- Alignment width: section content spans the container so it meets both page margins (no per-section measure caps); `text-wrap: balance` on headings, `pretty` on paragraphs.
- Shared section recipes: intros compose the `.intro` shell (eyebrow + `h2` + lead paragraph), labelled fine-print notices (legal or privacy) use `.disclaimer`, unlabelled caveat footnotes use `.boundary`, peer rows use `.card-grid` + `.card-tile`. A component owns only its deviations — never re-declare header, footnote, or heading typography.
- Primary actions use the shared `.btn` recipe (global.css) — see Component Patterns → Buttons.
- Closing section: the last landing section sits on the brand surface (`--brand-surface`) — the page's dark bookend with the hero — and is the one section exempt from positional banding. It is the page's single activation moment: exactly one `.btn--brand` action and no second link; the install command is the quiet button variant, not a second action. It carries the hound watermark (see Brand Identity). Its copy states the *act* (what happens next), never a restatement of the hero claim; the ownership claim stays in the footer sign-off so the close and the footer cannot duplicate each other.
- Closing row: a `--target-lg` quiet copy button + the `--target-xl` CTA, `--space-4` apart; the size inset is deliberate — it renders a field, not a second action.

## Surface Hierarchy

Three tiers via luminance stepping (flat design, no shadows). Homepage sections additionally band with `--bg-band` (50% page + 50% surface) — a transitional tone for full-bleed section rhythm, never used for cards. The closing section opts out of the band: it sits on the brand surface.

| Surface | Light | Dark |
|---------|-------|------|
| Page | `#f4f4f4` | `#2b2b2b` |
| Raised (cards) | `#ffffff` | `#3c3c3c` |
| Muted | `#e8e8e8` | `#4f4f4f` |
| Code/hero (`--code-bg`) | `#2b2b2b` | `#212121` |
| Brand (`--brand-surface`) | `#21221e` | `#21221e` |

## Component Patterns

### Selected / Active States

```
Selected chips / active nav items:
  text:       --text-primary
  background: --primary-bg
  border / indicator: --primary (1px or accent weight)
  icon:       currentColor (inherits --text-primary)
```

Cyan (--primary) is a structural accent only — never a text color on interactive selected states.

Text links use dedicated `--link*` tokens. On `--primary-bg` or primary-tinted surfaces, use
`--link-on-primary-bg*` tokens rather than `--primary`.

### Buttons

```
base:    radius:   --radius-full (pill) — declared once here, never per control
         font:     inherit, gap space-2, transition 0.2s
         focus:    outline border-w-emphasis solid --focus-ring, offset 3px
size:    .btn--xl  height --target-xl (56px), padding 0 space-8, weight 600 —
                   the page's ONE navigation action. The box IS --target-xl; the
                   line box centres inside it. Never derive the height from the
                   type because --lh-base grows at >=1600px. At <=360px the
                   inline padding drops to space-4 (`global.css`).
         .btn--lg  min-height --target-lg (48px), padding space-2 space-5 — the
                   command/copy field
variant: .btn--page   bg --primary, text --on-primary — the theme-adaptive action
                      on a page surface: dark teal/white in light, cyan/dark in
                      dark. hover → --bg-page; ring --link-focus
         .btn--brand  bg --brand-cta, text --on-brand-cta — the theme-invariant
                      action on the brand surface (closing CTA). hover →
                      --brand-surface; ring --on-brand-surface
         .btn--quiet  transparent, border --brand-border, text --on-brand-surface,
                      ring --on-brand-surface   (copy field beside an action)
```

`.btn` is the ONE control recipe: it declares the shape (pill radius, border, gap,
transition) and the focus state, and never colour. A component composes `.btn` +
one size + one variant keyed to the surface behind it, and declares neither shape
nor palette — so the command field and the closing CTA cannot drift on radius or
colour.

Buttons are pills (`--radius-full`); `--radius-control` is for the other controls
(inputs, selects, toggles, option rows). One primary button per section — a second
equal-weight action splits the decision, which is why a copy control beside the
CTA takes the quiet variant.

The variants own their palette, so a filled control cannot drift from the one
already shipping. Each filled variant must restate its foreground for `:hover`
and `:visited`: the prose-link recipe (`a`, `a:visited`, `a:hover` — element +
pseudo, 0,1,1) outranks a lone class (0,1,0), so a filled link that omits the
states silently takes `--link-hover` as its text colour and gains a hover
underline — an unreadable label on its own fill. The variant's own
`:hover`/`:visited` (class + pseudo, 0,2,0) outrank the prose-link recipe.

Each hover fill is derived with `color-mix()` toward the surface behind the
control and never a bare token: the contrast matrix treats every `var(--token)`
used as a `background` as a surface, and the control sits on one.

The focus ring is drawn **outside** the fill, so it must contrast the surface
**behind** the control. Each variant sets `--focus-ring` and the page fallback is
`--link-focus`, which is dark in light mode and would disappear on
`--brand-surface`; the recipe writes no geometry.

### Alerts

```
padding: space-3 space-4
font: text-sm, weight 500
layout: flex row, gap space-3, align center
status dot: space-2 circle
radius: error/warning -> radius-sm, success -> radius-lg
bg: semantic color at 12%/18% opacity
border: 1px semantic color at 20% opacity
```

### Cards

```
radius: radius-xl
padding: space-5
border: border-w-default solid --border-color (light) or --border-subtle (dark)
background: --bg-surface
```

### Code Blocks

```
font: JetBrains Mono 400, 14px, line-height 1.4
bg: --code-bg
padding: space-5
radius: radius-md
overflow-x: auto
```

### Search Input

```
height: target-md (40px)
padding: space-2 space-4
radius: --radius-control
bg: neutral-800 (dark) / neutral-100 (light)
font: text-sm
icon color: --primary
```

### Inline Code

```
bg: --primary-bg
color: --on-primary-bg
padding: 1px 4px
radius: radius-sm
font: JetBrains Mono, 0.8em relative to parent
```

### Stats Receipt

A short accent row of 2–4 figures. Lay it on the `.card-grid` columns so its
edges land on the same grid as the tile rows; each figure centers inside its
column and the row shares one baseline.

```
layout:     .card-grid columns + gutters (full container width)
figure:     --text-2xl / --lh-2xl, JetBrains Mono 600, tabular-nums, --primary
label:      --text-sm / --lh-sm, --text-secondary
divider:    border-w-default solid --border-color between columns
stack:      single column before the row can no longer share one baseline
```

### Quote / Testimonial

One voice, distinct from the docs **admonition** (`.docs-content blockquote`
renders `> **Note:**` callouts — notes, not quotations).

```
text:   --text-xl / --lh-xl (one rung above the lead tier), weight 400, roman, --text-primary
marks:  an accent glyph pair --text-2xl, --primary, weight 700, line-height 1,
        inline; ::before opens and ::after closes the prose
cite:   --text-sm, name 600 + role 400, --text-tertiary, letter-spacing 0.04em,
        em-dash lead, margin-top space-3
width:  container measure (no per-section cap), left-anchored
```

The quote stands one rung above the lead tier (`--text-xl`, never lower) so it
outranks the prose beside it, while the marks stay subordinate — at or below
~1.25x the copy. A mark near 2x the copy reads as decoration, not punctuation;
a lone opening glyph reads as an artifact, so ship the pair. Never italic.

### Disclaimer / Labelled Note

A **labelled** fine-print notice for legal disclosures and privacy/trust
boundaries: the label reuses the eyebrow pattern (the page's one label voice),
so the notice is unmistakable at a glance. Separate it from the prose above with
**spacing, not a rule** — see Border Weight: prefer spacing and surface tone
over borders. Distinct from a plain caveat footnote (`.boundary`).

```
label: <p class="eyebrow">Disclaimer</p>  (uppercase, --text-xs, 600, .12em, --text-tertiary)
text:  --text-sm / --lh-sm, --text-tertiary, margin-top space-2
block: margin-top space-6, left-anchored
```

Never shrink legal text below `--text-sm`: fine print still has to stay readable
and pass contrast. A labelled note is not a bare caption and never a standalone
horizontal rule.

**Rails.** Consecutive labelled notes form one rail, separated by spacing alone
(`--space-5` between items) — never boxes, borders, or surface fills, so a
section's real cards keep the only borders in view. Reuse one label per concept
across a page (the VoyageAI data-flow note and the web-research note are both
`Privacy`). A named disclosure takes the eyebrow label; an unnamed caveat stays a
`.boundary` footnote.

## Competitive Positioning

ChunkHound occupies a cyan/teal position: technical, fast, and readable across dark code surfaces. Preserve contrast over hue purity; use `--code-accent` for dark panels and `--primary` for normal page surfaces.

## Transitions

All theme-switching elements: `transition: background 0.3s, color 0.3s, border-color 0.3s`. Interactive hover states: `0.2s`.
