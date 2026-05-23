# ChunkHound Design System

Agent-executable brand specification. All values are final computed tokens — apply directly, no interpretation needed.

## Brand Identity

- **Wordmark:** "ChunkHound" in Inter 700, trailing cyan accent dot (circle, `#22d3ee`)
- **Tagline:** "Your entire codebase, deeply understood"
- **Personality:** Exciting + likeable. Cutting-edge but approachable.
- **PAD Profile:** Pleasure High, Arousal High, Dominance Medium
- **Logo:** SVG service dog illustration. Min clearspace = logo width around all sides. Use on dark (`#21221e`) or light (`#f4f6f1`) backgrounds only.

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

### Light Mode

| Token | Value |
|-------|-------|
| `--bg-page` | `#f4f4f4` (neutral-50) |
| `--bg-surface` | `#ffffff` |
| `--bg-muted` | `#e8e8e8` (neutral-100) |
| `--text-primary` | `#212121` (neutral-950) |
| `--text-secondary` | `#5f5f5f` |
| `--text-tertiary` | `#6e6e6e` |
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
| `--text-secondary` | `#a8a8a8` |
| `--text-tertiary` | `#919191` |
| `--text-muted` | `#777777` |
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
| `--text-4xl` | 49px | 60px | 700 | Page title |
| `--text-3xl` | 39px | 48px | 700 | Major heading |
| `--text-2xl` | 31px | 36px | 600 | Section heading |
| `--text-xl` | 25px | 32px | 600 | Subheading |
| `--text-lg` | 20px | 32px | 500 | Lead paragraph |
| `--text-base` | 16px | 24px | 400 | Body text |
| `--text-sm` | 13px | 20px | 400 | Caption, metadata |
| `--text-xs` | 10px | 16px | 600 | Label, overline |

### Weight Usage

| Weight | Role |
|--------|------|
| 400 | Body, caption |
| 500 | Lead text, body emphasis |
| 600 | Headings (2xl, xl), labels, section titles |
| 700 | Display headings (3xl, 4xl), wordmark |

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
| `--radius-lg` | 9px |
| `--radius-xl` | 12px |
| `--radius-2xl` | 18px |
| `--radius-full` | 9999px (pill) |

**Semantic radius overrides:**
- Error/warning alerts: `--radius-sm` (sharper = threat congruent)
- Success alerts: `--radius-lg` (rounder = positive valence)
- Cards/panels: `--radius-xl`
- Code blocks: `--radius-md`
- Inputs/search bars: `--radius-md`
- Pills/tags: `--radius-full`

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

- Max content width: `1120px`, centered
- Section padding: `space-10` vertical, `space-6` horizontal
- Grid gap: `space-5` (24px) default
- Container: `max-width: 1120px; margin: 0 auto`

## Surface Hierarchy

Three tiers via luminance stepping (flat design, no shadows):

| Surface | Light | Dark |
|---------|-------|------|
| Page | `#f4f6f1` | `#2b2c27` |
| Raised (cards) | `#ffffff` | `#3c3e37` |
| Muted | `#e7e9e3` | `#4f5149` |
| Hero/code bg | `#21221e` | `#21221e` |

## Component Patterns

### Selected / Active States

```
Selected chips / active nav items:
  text:       --text-primary
  background: --primary-bg
  border / indicator: --primary (1px or accent weight)
  icon:       currentColor (inherits --text-primary)
```

Green (--primary) is a structural accent only — never a text color on interactive selected states.

Text links use dedicated `--link*` tokens. On `--primary-bg` or green-tinted surfaces, use
`--link-on-primary-bg*` tokens rather than `--primary`.

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
radius: radius-md
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

## Competitive Positioning

ChunkHound occupies a cyan/teal position: technical, fast, and readable across dark code surfaces. Preserve contrast over hue purity; use `--code-accent` for dark panels and `--primary` for normal page surfaces.

## Transitions

All theme-switching elements: `transition: background 0.3s, color 0.3s, border-color 0.3s`. Interactive hover states: `0.2s`.
