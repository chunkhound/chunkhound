# Color Scheme Generation Guide for AI Agents

Production color palette generation using OKLCH color science, color emotion psychophysics, WCAG/APCA contrast validation, three-tier token architecture, and typography-aware accessibility engineering. Designed as agent-executable specification — every formula is code-ready. Domain-agnostic: works for any brand category.

This guide has three phases: **Brand Strategy** (human-driven research and hue selection), **Palette Engineering** (agent-executable math), and **Typography, Localization & Accessibility** (agent verification when applying the palette to content). The first phase grounds the human operator in marketing science, color emotion research, and competitor analysis before they commit to a hue. The second phase takes that hue and generates a complete, validated token system. The third phase verifies that the palette produces legible, accessible results across font sizes, weights, scripts, and user capabilities.

---

## Phase 1: Brand Strategy (Human Judgment)

Before picking a hue, ground the decision in marketing science, color emotion research, competitor landscape, and audience psychology. The agent assists with research retrieval; the human makes the judgment call.

### Distinctive Brand Assets Framework

**Source:** Jenni Romaniuk, *Building Distinctive Brand Assets* (Oxford University Press, 2018). Based on the Ehrenberg-Bass Institute for Marketing Science (University of South Australia) — decades of empirical data across thousands of brands.

Brand colors are evaluated on two axes:

```
                    HIGH UNIQUENESS
                         │
           Signature     │    Aspirational
          (own it, use   │   (build fame
            it heavily)  │    to match)
                         │
  LOW FAME ──────────────┼──────────────── HIGH FAME
                         │
           Avoid         │    Cemetery
          (no value,     │   (everyone uses it,
           retire it)    │    no one owns it)
                         │
                    LOW UNIQUENESS
```

- **Fame**: % of category buyers who associate the element with *any* brand
- **Uniqueness**: % who associate it with *only your* brand

A color shared by many competitors sits in the **Cemetery quadrant** — high fame, zero uniqueness. It triggers category recall ("a fintech app") but not brand recall ("*your* brand"). The goal is to move toward **Signature**: high fame *and* high uniqueness.

**Key formula:**

```
Brand Salience = Σ (Category Entry Points × DBA Linkage Strength)
```

Where DBA Linkage Strength is the probability that encountering the asset triggers recall of *your* brand specifically. A unique color maximizes this linkage; a shared color dilutes it across competitors.

### Sharp's Laws of Brand Growth

**Source:** Byron Sharp, *How Brands Grow* (Oxford University Press, 2010); *How Brands Grow Part 2* (2016).

Core empirical findings relevant to color selection:

1. **Growth comes from penetration, not loyalty.** Market share differences are explained almost entirely by *how many people know you*, not how often they engage. Implication: maximize recognizability across the broadest audience.

2. **Consumers perceive weak differentiation between rivals.** In practice, 72% of Coke drinkers also buy Pepsi. Meaningful product differences matter less than being *noticed and remembered*. Implication: invest in sensory distinctiveness (color, shape, sound), not messaging differentiation.

3. **Mental availability drives growth.** The easier it is for someone to think of your brand in a buying/learning situation, the more likely they are to choose it. Mental availability is built through consistent, unique sensory cues encountered across many touchpoints.

> *"Rather than striving for meaningful, perceived differentiation, marketers should seek meaningless distinctiveness."*
> — Byron Sharp

**Application:** Copying the dominant color in your category (e.g., blue in finance, purple in AI, green in sustainability) is textbook category conformity. It maximizes category association but destroys brand-level recall. A distinctive color builds a Distinctive Brand Asset that is exclusively yours.

### Color Emotion Science

Color triggers measurable physiological and psychological responses along three orthogonal dimensions: Pleasure, Arousal, and Dominance (PAD). Crucially, **saturation and lightness drive emotional response more than hue does** — a finding that overturns the popular "red = exciting, blue = calm" simplification.

**Sources:** Valdez & Mehrabian (1994), *J. Experimental Psychology: General*; Wilms & Oberfeld (2018), *Psychological Research*; Mehrabian & Russell (1974), *An Approach to Environmental Psychology*, MIT Press.

#### Emotional Response by Color Property

| Color Property | Pleasure (Valence) | Arousal | Dominance | Evidence Strength |
|---|---|---|---|---|
| **High lightness** | Strong ↑ | Moderate ↑ (at high sat only) | ↓ (lighter = approachable) | Very strong, cross-cultural |
| **High saturation** | Curvilinear — medium optimal | Strong ↑ (η² = .693) | ↑ (commanding) | Very strong |
| **Hue (red vs blue)** | Weak / non-significant (p = .051) | Red > Blue (high sat only) | Weak | Moderate, context-dependent |

Key findings:

1. **Saturation is the dominant emotional lever.** Effect size η² = .693 for arousal (Wilms & Oberfeld 2018) — the largest of any color property. Two blues at different saturations produce more divergent emotional responses than red vs. blue at the same saturation.

2. **Medium saturation maximizes pleasantness; high saturation maximizes arousal.** These are different goals. Backgrounds and surfaces want medium chroma (pleasant, non-fatiguing). Alerts and CTAs want high chroma (attention-grabbing). Valence peaks at medium saturation (M = 5.82) and drops slightly at high saturation (M = 5.52).

3. **Lightness → pleasure is the most reliable cross-cultural finding.** Confirmed across 30 nations (Jonauskaite et al. 2020, r = .88 cross-country agreement). Lighter colors consistently evoke positive responses; darker colors evoke authority and dominance.

4. **Hue effects on valence are non-significant** when saturation and brightness are properly controlled (Wilms & Oberfeld 2018, p = .051). The blue > green > red preference ranking only emerges at high saturation levels.

5. **Color emotion is context-dependent.** Elliot & Maier's Color-in-Context Theory (2012) demonstrated that the same color produces opposite behavioral responses depending on context — red increased approach motivation in romantic contexts but avoidance in achievement contexts. Any hue→emotion mapping without context specification is incomplete.

6. **Ecological valence explains 80% of color preference.** Palmer & Schloss (2010) showed that people prefer colors associated with objects they like (blue → clear sky/water) and dislike colors associated with negative objects (brown → waste). These associations are partially domain-specific: terminal green for developers, institutional blue for finance, clinical white for healthcare.

#### Critical Interaction Effects

These interactions are as large as main effects — you cannot reason about dimensions independently (Wilms & Oberfeld 2018):

| Interaction | η² | Implication |
|---|---|---|
| Hue × Saturation on arousal | .637 | Red-is-arousing only shows up at high saturation. A desaturated red feels similar to a desaturated blue. |
| Brightness × Saturation on arousal | .543 | Brightness only increases arousal when saturation is also high. A bright but desaturated color is not arousing. |
| Brightness × Saturation on valence | — | At low saturation, valence strongly depends on brightness. At high saturation, brightness matters less. |

#### Physiological vs. Subjective Responses

Skin conductance (autonomic arousal) correlates only moderately with self-reported arousal (r = 0.42). Saturation significantly affected skin conductance (η² = .184); hue did not (p = .113). People's felt responses and their bodies' responses are related but not identical.

### Color Associations by Domain

Hue associations are **contextual tendencies** modulated by saturation and lightness (per Elliot & Maier 2012), not fixed properties. At low saturation, hue-based associations largely disappear. The table below maps general associations and shows how domain context shifts their signal.

| Color Family | General Associations | Example Domain Signals |
|---|---|---|
| Purple (260–280°) | Authority, innovation, wisdom | Education: "advanced"; AI: saturated/low uniqueness; Luxury: "premium" |
| Blue (220–250°) | Trust, reliability, competence | Finance: "institutional trust"; Tech: generic; Healthcare: "clinical calm" |
| Cyan (185–210°) | Precision, clarity, digital | Dev tools: "terminal culture"; Science: "analytical"; Health: "clinical precision" |
| Green (130–160°) | Growth, freshness, progression | Education: "leveling up"; Finance: "prosperity"; Sustainability: "natural" |
| Red (0–30°) | Energy, urgency, danger | Conflicts with error semantic — avoid as primary in any domain |
| Orange (30–55°) | Energy, warmth, boldness | Media: "creative energy"; Food: "appetite"; conflicts with warning semantic |
| Fuchsia (310–340°) | Boldness, modernity, creativity | Maximum distinctiveness; Fashion: "avant-garde"; can feel less formal |

**Source for personality mappings:** Labrecque & Milne (2012), *J. Academy of Marketing Science* — Red → Excitement, Blue → Competence, White → Sincerity, Black/Purple/Pink → Sophistication, Brown → Ruggedness. Saturation amplifies the existing hue-personality association; it does not create new ones.

Quantitative findings on color and cognition (SHIFT eLearning):
- Intentional color schemes amplify learning by **55–78%** and comprehension by **up to 73%**
- Cool colors (blue, green, purple) set focused moods for concentrated study
- Cognitive overload from too many bright colors is a significant risk — minimalism matters
- Colors used to segment information act as mnemonic aids for complex topics

### Typeface Personality Science

Font classifications trigger measurable personality associations analogous to color
emotion. Like color, typeface personality operates in PAD-adjacent space — but font
weight primarily modulates Dominance while color primarily modulates Pleasure/Arousal,
making them partially orthogonal. They should be selected jointly, not independently.

#### Typeface Personality Dimensions

Three dimensions (Brumberger 2003, factor analysis of 15 typefaces rated on 20
adjective pairs):

| Font Dimension | PAD Mapping | Description |
|---|---|---|
| Elegance | ≈ Arousal | Refined, sophisticated, distinguished |
| Directness | ≈ Dominance | Professional, stable, assertive |
| Friendliness | ≈ Pleasure | Warm, playful, approachable |

#### Font Classification → PAD Profile

(Synthesized from Brumberger 2003, Shaikh et al. 2006, Henderson et al. 2004,
Monotype/Neurons 2023)

| Classification | Pleasure | Arousal | Dominance | Personality Keywords |
|---|---|---|---|---|
| Serif (Old Style) | Med | Low | Med-High | Trustworthy, traditional, scholarly |
| Serif (Transitional) | Med | Low-Med | High | Authoritative, refined, professional |
| Serif (Didone/Modern) | Med-High | Med | High | Elegant, dramatic, fashionable |
| Sans (Geometric) | Low-Med | Med | Med | Precise, cold, modern, innovative |
| Sans (Humanist) | High | Med | Med | Warm, friendly, accessible, readable |
| Sans (Neo-Grotesque) | Med | Low | Med | Neutral, corporate, invisible |
| Slab Serif | Low-Med | High | High | Bold, industrial, confident |
| Monospace | Low | Low | Med | Technical, precise, honest |
| Script (Formal) | High | Low | Low | Elegant, personal, feminine |
| Script (Casual) | High | Med | Low | Playful, creative, youthful |
| Display | Variable | High | High | Unique, bold, attention-grabbing |

#### Font Weight as Independent Emotional Lever

Weight operates on the Dominance axis independently of hue:

| Weight Range | Dominance | Personality |
|---|---|---|
| 100–200 (Thin) | Low | Exclusive, delicate, luxury, minimal |
| 300 (Light) | Low-Med | Elegant, refined, airy |
| 400 (Regular) | Med | Neutral, readable, default |
| 500–600 (Medium) | Med-High | Confident, structured, clear |
| 700 (Bold) | High | Authoritative, urgent, confident |
| 800–900 (Black) | Very High | Maximum impact, commanding |

Color saturation is the dominant arousal lever (η² = .693). Font weight is the
dominant dominance lever. They are partially orthogonal — bold + high chroma =
maximum arousal AND dominance (not merely additive).

#### Font-Color Congruence

Font personality and color emotion should align. When they conflict, the penalty
is real: 22% credibility loss for incongruent typography (Fox et al. 2007), and
incongruent signals create active confusion, not neutral averaging.

Congruence examples:

| Target Profile | Congruent Font | Congruent Color | Why It Works |
|---|---|---|---|
| Calm Trust | Humanist sans, traditional serif | Blue (210–240°), teal | Both signal accessibility + stability |
| Urgent Authority | Slab serif, bold geometric sans | Red, deep orange | Both signal dominance + arousal |
| Friendly Innovation | Humanist sans, rounded geometric | Cyan (185–210°), bright green | Both signal warmth + modernity |
| Luxury Elegance | Didone serif, light weights | Deep purple, gold | Both signal refinement + exclusivity |
| Technical Precision | Monospace, geometric sans | Cool neutrals, cyan | Both signal rationality + honesty |

Incongruence red flags:
- Script font + saturated red → font says "elegant," color says "urgent"
- Heavy slab serif + pastel pink → font says "industrial," color says "soft"
- Comic Sans/casual script in any professional domain → immediate credibility loss

#### Font Classification × Domain Associations

Like color, font classifications carry domain associations. When font-domain and
color-domain signals align, the effect is multiplicative:

| Domain | Expected Font | Expected Color | Mismatched Font Signal |
|---|---|---|---|
| Law, finance | Serif (transitional) | Navy blue, dark green | Sans-serif = "too casual" |
| Tech, SaaS | Sans (geometric/humanist) | Blue, cyan, purple | Serif = "dated" |
| Developer tools | Monospace + sans body | Cyan, green, neutral | Script = "not serious" |
| Healthcare | Humanist sans | Teal, green, blue | Display = "not trustworthy" |
| Luxury, fashion | Didone serif, thin weights | Gold, deep purple, black | Slab serif = "too industrial" |
| Education | Humanist sans, readable serif | Blue, green, warm accents | Monospace = "too technical" |

#### Serif vs Sans-Serif: The Readability Non-Issue

72 studies reviewed (Lund 1999): no meaningful readability difference between serif
and sans-serif on screen. Arditi & Cho (2005): the minor serif advantage sometimes
observed is a spacing artifact. Choose serif vs sans-serif for personality, not
readability. What actually affects legibility: x-height, open counters, character
differentiation, stroke uniformity, spacing (see Phase 3 § X-Height and Font Metrics).

At sizes below ~10px on screen, serifs become noise and reduce reading speed
(Morris et al. 2002). At 18px+, they are decorative signal contributing to personality.

#### Monospace: Developer Context

Monospace carries precision/honesty associations (Shaikh et al. 2006: 40% chose
monospace for programming contexts) but also "dull/plain/conforming" personality.
For developer-facing brands wanting both technical credibility and approachability,
pair monospace (code blocks) with a humanist sans (body text) — the contrast creates
"warm but competent."

Monospace typography is typically exempt from brand color — syntax highlighting
dominates. The font's personality contribution is structural (uniform advance width,
even typographic color) rather than chromatic.

#### Cultural Variation

Monotype/Neurons (2023, N=1,957, 8 countries):
- Humanist sans-serif scored highest for trust in 7 of 8 countries
- Germany uniquely preferred serif (Cotford) for trust
- Romance-language countries preferred classic serif styles
- Japan: low-contrast humanistic typefaces → innovation; brushstroke-feel → trust

Font personality associations have significant cultural modulation. When targeting
non-English markets, validate font choice with locale-specific research.

#### Evidence Quality Caveat

Font classification → personality: strong evidence (multiple replicated studies).
Font-color interaction: weak evidence (inferential only, no controlled experiments
manipulating both variables simultaneously). The congruence framework above is
grounded in theory and converging practitioner evidence, not in direct measurement
of interaction effects.

### Competitor Landscape Analysis

The agent should research the current competitor color map before the human selects a hue. Use this template:

```
Research query: "[your category] brands and platforms primary colors [current year]"
```

**Competitor map template:**

| Brand | Primary Hue | Hex | Sub-category |
|---|---|---|---|
| [Competitor 1] | ???° | #?????? | [sub-category] |
| [Competitor 2] | ???° | #?????? | [sub-category] |
| ... | | | |

The agent populates this table via web research. The human reviews for completeness.

**Key observations to surface:**
- Which hue zones are crowded (Cemetery quadrant — high fame, zero uniqueness)?
- Which category leaders have already differentiated away from the dominant hue?
- Which hue zones are unoccupied in your specific category?
- What is the minimum hue distance achievable from the nearest competitor (target ≥40°)?

### Audience Credibility & Color-Brand Congruence

**Sources:** Bottomley & Doyle (2006), *Marketing Theory*; Labrecque & Milne (2012), *J. Academy of Marketing Science*.

Two findings create productive tension with Sharp's distinctiveness advice:

1. **Color-product congruence increases processing fluency.** When a color "fits" the category, it increases brand recognition and positive evaluation (Bottomley & Doyle 2006). Consistent color usage can increase brand recognition by up to 80%.

2. **Atypical colors grab attention but can decrease purchase intent.** Schema incongruity attracts notice but increases skepticism, especially in categories with strong color conventions.

**Resolution:** Distinctiveness works when the brand has substance to back it up. Audiences in every domain evaluate credibility on content quality, transparency, and peer validation — not on whether your color matches the category leaders. A distinctive color signals confidence; a copied color signals imitation.

**Brand personality hue mapping** (Labrecque & Milne 2012):

| Hue | Brand Personality Dimension |
|---|---|
| Red | Excitement |
| Blue | Competence |
| White | Sincerity |
| Black, Purple, Pink | Sophistication |
| Brown | Ruggedness |

High saturation amplifies the existing hue-personality association (e.g., saturated red = more exciting). It does not create new personality dimensions.

### Hue Distance Formula

When evaluating candidate hues against competitors and semantic roles, calculate angular distance on the 360° wheel:

```
distance(h1, h2) = min(|h1 − h2|, 360 − |h1 − h2|)
```

Minimum recommended distances:
- **≥30°** from any semantic role hue (error, warning, success)
- **≥40°** from the nearest direct competitor
- **≥60°** from the previous brand color (if rebranding) for the shift to be perceptible to casual viewers

### Decision Framework

Score each candidate hue across these dimensions:

| Dimension | Weight | What to Evaluate |
|---|---|---|
| Differentiation | High | Hue distance from competitors; uniqueness in category |
| Semantic Safety | High | Hue distance from error (25°), warning (70°), success (155°) |
| Emotional Profile | Medium | Does the PAD profile at planned peak chroma match brand intent? (See Color Emotion Science) |
| Authority Signal | Medium | Color psychology fit for target audience |
| Dual-Mode WCAG | Medium | Can shade 600 pass AA on white AND shade 400 pass AA on dark? |
| Domain Fit | Medium | Does it signal appropriate values for your category? |
| Typeface Congruence | Medium | Does the chosen font's PAD profile align with the color's PAD profile? (See Typeface Personality Science) |
| Cultural Resonance | Low | Domain-specific subculture associations (varies by audience) |

The human operator evaluates these qualitatively. The agent can compute hue distances and WCAG numbers to support the decision. Once a hue is chosen, everything below is math.

---

## Phase 2: Palette Engineering (Agent Math)

From this point forward, the agent generates the palette autonomously. The human provides perceptual feedback during validation.

**IMPORTANT: The agent MUST write code to run the math then execute it, NEVER attempt to compute values directly. Strict mathematical adherance!**

## Why OKLCH Over HSL

HSL is not perceptually uniform. `hsl(60,100%,50%)` (yellow) appears far brighter than `hsl(240,100%,50%)` (blue) at identical lightness values. OKLCH fixes this by design — equal numerical changes produce equal visual changes across all hues.

Three channels:
- **L** (Lightness): 0.0–1.0 — perceptual brightness
- **C** (Chroma): 0.0–~0.37 — saturation intensity
- **H** (Hue): 0°–360° — position on the color wheel

All palette math operates in OKLCH. sRGB hex is output-only.

## Input: Source Hue Table

The only human judgment calls. Everything downstream is computable math.

| Role | Hue | Peak Chroma | Rationale |
|------|-----|-------------|-----------|
| Primary | _chosen_ | 0.13–0.15 | Brand identity, CTAs, active states |
| Neutral | _same as primary_ | 0.015 | Brand-tinted gray (same hue, ~10% chroma) |
| Error | 25° | 0.15 | Convention: red-family for danger |
| Warning | 70° | 0.12 | Convention: amber for caution |
| Success | 155° | 0.13 | Teal-green — colorblind-safe vs error red |

Error is red because users expect it. Success uses teal-green (not pure green) because teal remains distinguishable from error-red under protanopia and deuteranopia. These are convention-fixed, not derived from the brand hue.

### Primary Hue Selection Constraints

The primary hue must avoid collision with semantic roles:

| Zone | Hue Range | Conflict |
|------|-----------|----------|
| Avoid | 0°–50° | Error red (25°) |
| Avoid | 55°–85° | Warning amber (70°) |
| Avoid | 140°–170° | Success teal (155°) |
| Caution | _category-specific_ | Your category's dominant competitor hue zone — low uniqueness |

Good candidate zones: 90°–135°, 175°–210°, 215°–255°, 285°–350°.

## Shade Scale Generation

### Lightness Curve (Non-Linear)

11 stops mapping to shade names. Not evenly spaced — compressed at extremes for perceptual evenness:

```
Shade:     50    100   200   300   400   500   600   700   800   900   950
Lightness: 0.97  0.93  0.87  0.78  0.69  0.60  0.51  0.43  0.36  0.29  0.25
```

### Parabolic Chroma Curve

Peak saturation at midtones, taper at extremes. Prevents oversaturated near-white or muddy near-black shades:

```
chroma(L) = peakChroma × max(0, 1 − ((L − 0.6) / 0.5)²)
```

- Vertex at L=0.6 (shade 500) — maximum chroma
- Denominator 0.5 controls parabola width
- Chroma reaches zero when L deviates ±0.5 from center
- Clamp to `[0, peakChroma]`

> *Emotional rationale: The parabolic curve peaks at medium chroma (shade 500, L=0.60), which research shows maximizes pleasantness (Valdez & Mehrabian 1994). Extreme shades taper toward zero chroma, reducing arousal — appropriate for backgrounds and subtle UI surfaces. High-chroma midtones are reserved for interactive elements where attention (arousal) is needed.*

### Dark Mode Inversion

Same shade names, reversed lightness indices:

```
L_dark(i) = LIGHTNESS_STOPS[10 − i]
```

Shade 50 in dark mode gets the lightness of shade 950 (0.25), and vice versa. Chroma curve is recomputed from the new lightness — not copied.

### Bezold-Brücke Hue Compensation

Colors perceptually drift toward yellow (~65°) or blue (~250°) as lightness changes. For production accuracy, apply per-shade hue correction. The basic implementation uses fixed hue; production systems shift hue ±2–5° per shade to counteract the drift.

## OKLCH → sRGB Conversion

Four-step pipeline. All intermediate values are floating-point.

### Step 1: OKLCH → OKLab

```
h_rad = H × π / 180
a = C × cos(h_rad)
b = C × sin(h_rad)
```

### Step 2: OKLab → LMS (cube-root space)

```
l' = L + 0.3963377774·a + 0.2158037573·b
m' = L − 0.1055613458·a − 0.0638541728·b
s' = L − 0.0894841775·a − 1.2914855480·b

l = l'³
m = m'³
s = s'³
```

### Step 3: LMS → Linear sRGB

```
R_lin = +4.0767416621·l − 3.3077115913·m + 0.2309699292·s
G_lin = −1.2684380046·l + 2.6097574011·m − 0.3413193965·s
B_lin = −0.0041960863·l − 0.7034186147·m + 1.7076147010·s
```

### Step 4: Linear → Gamma sRGB

```
f(x) = 12.92·x                       if x ≤ 0.0031308
f(x) = 1.055·x^(1/2.4) − 0.055      if x > 0.0031308
```

Clamp each channel to `[0, 1]` before gamma. Values outside sRGB gamut are clamped by reducing chroma while preserving hue (gamut mapping).

### Hex Output

```
hex = '#' + toHex(f(clamp(R_lin))) + toHex(f(clamp(G_lin))) + toHex(f(clamp(B_lin)))
toHex(x) = round(x × 255).toString(16).padStart(2, '0')
```

## WCAG Contrast Validation

### Relative Luminance (ITU-R BT.709)

```
Y = 0.2126·R_lin + 0.7152·G_lin + 0.0722·B_lin
```

Where `R_lin`, `G_lin`, `B_lin` are the linear (pre-gamma) sRGB values, clamped to `[0, 1]`.

### Contrast Ratio (WCAG 2.1 §1.4.3)

```
CR = (Y_lighter + 0.05) / (Y_darker + 0.05)
```

### Thresholds

| Level | Normal text | Large text (≥18pt or ≥14pt bold) |
|-------|-------------|----------------------------------|
| AA | ≥ 4.5:1 | ≥ 3.0:1 |
| AAA | ≥ 7.0:1 | ≥ 4.5:1 |

### Best Text Color Selection

```
white_CR = (1.0 + 0.05) / (Y_bg + 0.05)
black_CR = (Y_bg + 0.05) / (0.0 + 0.05)
text_color = white_CR ≥ black_CR ? white : black
```

The crossover occurs around shade 500–600. This is computed, not chosen.

> **Note:** WCAG 2.x contrast ratios are the legal compliance baseline but have known perceptual inaccuracies, especially for dark color pairs (overestimation up to 250%). Phase 3 introduces APCA (Accessible Perceptual Contrast Algorithm) as a perceptual-truth layer. Run both; flag discrepancies.

## Semantic Token Mapping

### Three-Tier Architecture

| Tier | Name | Example | Purpose |
|------|------|---------|---------|
| Primitive | Raw values | `--color-primary-500: oklch(0.60 0.13 195)` | Color space, no semantics |
| Semantic | Intent | `--color-primary: var(--color-primary-600)` | Meaning, theme-switchable |
| Component | Scoped | `--button-bg: var(--color-primary)` | Component-specific binding |

Components reference semantic tokens, never primitives. A theme swap changes only the primitive→semantic mapping.

### Light/Dark Semantic Mapping

Use shade **600** for light-mode semantic colors (passes AA on white). Use shade **400** for dark-mode semantic colors (passes AA on dark backgrounds).

| Semantic Token | Light Theme | Dark Theme |
|----------------|-------------|------------|
| `--bg-page` | neutral-50 | neutral-900 |
| `--bg-surface` | white | neutral-800 |
| `--bg-muted` | neutral-100 | neutral-700 |
| `--text-primary` | neutral-900 | neutral-50 |
| `--text-secondary` | neutral-500 | neutral-400 |
| `--border` | neutral-200 | neutral-700 |
| `--primary` | primary-600 | primary-400 |
| `--on-primary` | white | primary-950 |
| `--error` | error-600 | error-400 |
| `--on-error` | white | error-950 |
| `--error-subtle` | error-50 | error-950 |
| `--warning` | warning-600 | warning-400 |
| `--warning-subtle` | warning-50 | warning-950 |
| `--success` | success-600 | success-400 |
| `--success-subtle` | success-50 | success-950 |

### Paired Foreground Tokens

Every semantic background needs a paired foreground with guaranteed contrast ≥ 4.5:1. Material Design 3 calls these `on-primary`, `on-error`. Without explicit pairs, agents pick arbitrary text colors and contrast breaks silently.

## Color Harmony (Decorative Only)

Harmony rotation produces hue sets for data visualization, illustration, and accents. **Not** for semantic roles — those are convention-fixed.

| Harmony | Rotation from base | Use case |
|---------|-------------------|----------|
| Monochromatic | 0° (shade scale only) | Single-brand UIs |
| Analogous | ±30° | Harmonious palettes, gradients |
| Complementary | 180° | CTAs, high-contrast accents |
| Split-complementary | 150° + 210° | Balanced accent pairs |
| Triadic | ±120° | Data visualization |
| Tetradic | 90° intervals | Complex UIs needing 4+ hues |

Apply harmony at shade 500 lightness (L=0.60) using the parabolic chroma at that lightness.

## Dark Mode Best Practices

- Never use pure black (`#000000`) — causes halation (white text bleeds). Use dark grays: `#0d1117`, `#1a1a2e`, `#161b22`.
- Never use pure white (`#ffffff`) for body text on dark — use off-white: `#e6edf3`, `#d4d4d4`.
- Pure white is acceptable for headings and high-emphasis text.
- The 60-30-10 rule: 60% dark background, 30% mid-tone surfaces, 10% accent color.
- Neutral scale uses brand hue at 0.015 chroma — produces brand-tinted gray, not pure gray.

> **Note:** Dark mode introduces additional perceptual effects (astigmatic halation, APCA polarity asymmetry, chroma overstatement) covered in Phase 3 § Accessibility Matrix > Dark Mode Accessibility.

## Validation Checklist

For every generated palette, verify:

- [ ] 5 roles × 11 shades = 55 primitive tokens generated
- [ ] Every shade has both OKLCH and hex values
- [ ] Shade 600 of each role passes WCAG AA (≥4.5:1) on white
- [ ] Shade 400 of each role passes WCAG AA (≥4.5:1) on dark background
- [ ] Shade 700 provides AA fallback if 600 is marginal in light mode
- [ ] Neutral scale chroma capped at 0.015
- [ ] Every semantic background has a paired `on-*` foreground token
- [ ] No out-of-gamut values (all hex channels in 00–ff)
- [ ] Primary hue ≥30° from any semantic role hue
- [ ] Light and dark theme mappings are complete and cross-validated

Phase 3 extends this checklist with typography, localization, and accessibility verification items. If the palette will be applied to text content, complete both checklists.

---

## Phase 3: Typography, Localization & Accessibility (Agent Verification)

Phase 2 produces a validated color palette. Phase 3 verifies that the palette produces legible, accessible results when applied to actual text content. Contrast is not a property of two colors — it is a function of two colors + font size + font weight + x-height ratio + script complexity + rendering context + user capabilities.

This phase is required when the palette will be used with text. It can be skipped if the palette is decorative-only (e.g., data visualization with labeled values).

**IMPORTANT: The agent MUST write code to run the math then execute it, NEVER attempt to compute values directly. Strict mathematical adherence!**

### APCA: Next-Generation Contrast Model

WCAG 2.x contrast ratios (Phase 2) remain the legal compliance baseline. APCA (Accessible Perceptual Contrast Algorithm) is the perceptual-truth layer used in WCAG 3.0 drafts. Use both; flag discrepancies.

**Why APCA over WCAG 2.x:**

WCAG 2.x uses a simple luminance ratio derived from a 1988 CRT standard. It has three documented failure modes:

1. **False passes:** Dark-on-dark combinations that compute CR ≥ 4.5:1 but are actually unreadable (measure only Lc 33 in APCA).
2. **False fails:** Readable light-gray-on-white combinations that fail CR 3:1 but measure Lc 61 in APCA.
3. **Dark pair overestimation:** WCAG 2.x overestimates contrast for very dark color pairs by 200–250%.

Additionally, WCAG 2.x is **polarity-blind** — it returns the same ratio for white-on-black and black-on-white. Human perception is asymmetric: light-on-dark produces halation effects that reduce readability compared to dark-on-light at equivalent luminance differences.

**APCA Lc (Lightness Contrast):**

APCA produces a signed value: positive for dark text on light background, negative for light text on dark background. Use the absolute value |Lc| for threshold comparison. The Lc scale is perceptually uniform — equal increments represent equal perceived changes across the entire luminance range.

Reference implementation: `apca-w3` npm package (Myndex). Do not implement the algorithm manually — it involves piecewise power curves with multiple exponents.

**Lc Thresholds by Use Case:**

| Use Case | Minimum |Lc| | Preferred |Lc| | Notes |
|---|---|---|---|
| Body text (14–16px, weight 400) | 75 | 90 | Primary reading content |
| Subheadings (18–24px, weight 600–700) | 60 | 75 | Section navigation |
| Large headlines (32px+, weight 700+) | 45 | 60 | Display text |
| Secondary text (captions, bylines) | 45 | 60 | Metadata, timestamps |
| Placeholder text, disabled | 30 | 40 | Non-essential, ghosted |
| Decorative text, watermarks | 15 | 25 | Not intended to be read fluently |
| Non-text UI (icons, borders, focus rings) | 45 | 60 | WCAG 1.4.11 equivalent |

**Dual-Model Rule:**

```
Run both WCAG 2.x CR and APCA Lc for every text-color pair.
Flag if:
  - WCAG passes (CR ≥ 4.5) but APCA fails (|Lc| < 75 for body text)
  - WCAG fails (CR < 4.5) but APCA passes (|Lc| ≥ 75 for body text)
Report both values. Use WCAG for legal compliance; use APCA for perceptual truth.
```

**Sources:** Andrew Somers, *APCA Readability Criterion* (W3C Silver/WCAG 3.0 draft); Somers, "The Realities And Myths Of Contrast And Color" (*Smashing Magazine*, 2022).

### Font Size × Weight × Contrast Matrix

Human contrast sensitivity is a function of **spatial frequency**. Thinner strokes at smaller sizes produce higher spatial frequency, which the visual system resolves with significantly lower contrast sensitivity. The same gray text on white appears less readable in weight 300 than weight 600 — identical luminance difference, but the perceived contrast drops because the spatial frequency is higher.

APCA replaces WCAG's binary "normal/large text" with a continuous lookup table. Weight and size trade off against each other:

**Minimum |Lc| by Font Size and Weight (APCA Silver Level):**

| Size (px) | W100 | W200 | W300 | W400 | W500 | W600 | W700 | W800 | W900 |
|---|---|---|---|---|---|---|---|---|---|
| 12 | — | — | — | — | — | — | — | — | — |
| 14 | — | — | — | 100 | 100 | 90 | 75 | — | — |
| 15 | — | — | — | 100 | 90 | 75 | 70 | — | — |
| 16 | — | — | — | 90 | 75 | 70 | 60 | 60 | — |
| 18 | — | — | 100 | 75 | 70 | 60 | 55 | 55 | 55 |
| 21 | — | — | 90 | 70 | 60 | 55 | 50 | 50 | 50 |
| 24 | — | — | 75 | 60 | 55 | 50 | 45 | 45 | 45 |
| 28 | — | 100 | 70 | 55 | 50 | 45 | 43 | 43 | 43 |
| 32 | — | 90 | 65 | 50 | 45 | 43 | 40 | 40 | 40 |
| 36 | — | 75 | 60 | 45 | 43 | 40 | 38 | 38 | 38 |
| 42 | 100 | 70 | 55 | 43 | 40 | 38 | 35 | 35 | 35 |
| 48 | 90 | 60 | 50 | 40 | 38 | 35 | 33 | 33 | 33 |
| 60 | 75 | 55 | 45 | 38 | 35 | 33 | 30 | 30 | 30 |
| 72 | 60 | 50 | 40 | 35 | 33 | 30 | 30 | 30 | 30 |
| 96 | 50 | 45 | 35 | 33 | 30 | 30 | 30 | 30 | 30 |

`—` = prohibited: insufficient stroke density for legibility at any contrast level.

**Practical shade selection rule:**

```
Given: font_size, font_weight, background_shade
1. Look up required_Lc from the table above
2. Iterate candidate text shades from darkest (or lightest on dark bg):
     text_shade = first shade where |Lc(text, bg)| >= required_Lc
3. If no shade passes: increase font_size or font_weight until a shade qualifies
```

> *Note: These are 2023 draft values from Myndex APCA Silver. The canonical source is the `apca-w3` package. Values may shift as the WCAG 3.0 specification finalizes.*

**Source:** Myndex, *APCA Silver Level Font Lookup Tables* (2023 draft).

### Text Chroma Limits (Helmholtz-Kohlrausch Effect)

Text readability depends almost entirely on **luminance contrast**. The visual cortex resolves letter shapes through the achromatic luminance channel. Chromatic channels (OKLab a, b axes) process hue and saturation at much lower spatial resolution — they contribute to object categorization but cannot resolve fine text strokes.

The **Helmholtz-Kohlrausch (H-K) effect** causes highly saturated colors to appear perceptually brighter than their measured luminance. Two color pairs with identical Lc values can produce vastly different reading comfort — the high-chroma pair causes halation and eye strain even though the computed contrast is equivalent.

**Chroma Budget by Text Role (OKLCH C value):**

| Text Role | Max Chroma | Rationale |
|---|---|---|
| Body text (paragraphs, lists) | C < 0.04 | Minimize H-K distortion, maximize stroke clarity |
| Labels, captions | C < 0.04 | Same spatial frequency as body text |
| Interactive text (links, buttons) | C < 0.08 | Brand expression allowed; must verify Lc independently of chroma |
| Display headings (28px+) | C < 0.12 | Larger size reduces spatial frequency, tolerates more chroma |
| UI surfaces (button fills, badges) | Full palette | No fine detail to resolve at surface level |
| Backgrounds | Full palette | Large area, low spatial frequency |
| Decorative / illustrative | Full palette | Not read as text |

**Rule:**

```
For any token assigned as a text foreground color:
  assert oklch_chroma(token) <= chroma_limit[role]
If brand color must appear as text, reduce chroma to the role limit and re-verify Lc.
```

**Sources:** Helmholtz (1867); von Kohlrausch (1935); Fairchild & Pirrotta (1991), "Predicting the brightness of different hues."

### X-Height and Font Metrics

The APCA lookup table assumes an **x-height ratio ≥ 0.5** (approximately Inter, Roboto, system sans-serif). CSS `font-size` specifies the em square — not the visible height of lowercase letters. Two fonts at `font-size: 16px` can differ by 38% in actual rendered letter height:

| Font | x-height Ratio | Effective Size at 16px CSS | Category |
|---|---|---|---|
| Verdana | 0.55 | 17.6px | System sans |
| Inter | 0.54 | 17.3px | Modern sans |
| Roboto | 0.53 | 17.0px | Modern sans |
| Noto Sans | 0.52 | 16.6px | Universal sans |
| Noto Sans CJK | 0.52 | 16.6px (but see Localization) | CJK sans |
| Georgia | 0.48 | 15.4px | System serif |
| Times New Roman | 0.45 | 14.4px | System serif |
| Garamond | 0.41 | 13.1px | Classic serif |

**Compensation formula:**

```
effective_size = css_font_size × (actual_x_height_ratio / 0.5)
```

Use `effective_size` when consulting the APCA font-size × weight lookup table. A font with x-height ratio 0.41 (Garamond) at 16px CSS is effectively 13.1px for contrast purposes — requiring significantly higher |Lc| than 16px Inter.

**Font rendering caveat:**

`-webkit-font-smoothing: antialiased` forces grayscale antialiasing, which reduces perceived weight and contrast by ~10–15% on standard-DPI displays (~96 DPI). On HiDPI (≥192 DPI), the effect is imperceptible. Gate it behind a resolution media query:

```css
@media (min-resolution: 192dpi) {
  body { -webkit-font-smoothing: antialiased; }
}
/* Standard DPI: leave as system default (subpixel rendering) */
```

Palettes validated on Retina displays must be re-verified on standard-resolution monitors.

### Localization Constraints

The APCA lookup table and all preceding contrast calculations assume **Latin script with x-height ratio ~0.52**. Scripts with higher stroke complexity pack more detail into each glyph, creating higher spatial frequency at the same nominal size. This requires larger minimum sizes, heavier weights, and higher contrast targets.

**Script Complexity and Minimum Thresholds:**

| Script | Avg Strokes/Glyph | Min Body Size | Min Weight | Min Line-Height | Max Chars/Line | APCA Offset |
|---|---|---|---|---|---|---|
| Latin | 2–3 | 14px | 300 | 1.4× | 75 | 0 (baseline) |
| Cyrillic | 2–3 | 14px | 300 | 1.4× | 75 | 0 |
| Greek | 2–3 | 14px | 300 | 1.4× | 75 | 0 |
| CJK (Han/Kanji) | 8–12 | 16px (18px pref.) | 400 | 1.6× | 40 | −3px |
| Korean (Hangul) | 4–6 | 16px | 400 | 1.5× | 45 | −2px |
| Arabic | 3–5 (connected) | 16px | 400 | 1.7× | 50 | −2px |
| Devanagari | 4–6 | 16px | 400 | 1.8× | 55 | −2px |
| Thai | 3–5 (stacking) | 16px | 400 | 1.8× | 50 | −2px |
| Vietnamese (Latin+) | 2–3 | 14px (16px pref.) | 300 | 1.7× | 70 | 0 |

**APCA Offset** means: when consulting the font-size × weight lookup table, subtract this value from the CSS font size to get the effective size for that script. CJK at 18px behaves like Latin at 15px for legibility purposes.

**CJK-specific rules:**

```
CJK thin weights (100–200) are prohibited — stroke merging makes text illegible.
CJK body text under 20px: target |Lc| 90 minimum (not the standard 75).
CJK line width: max 40 characters (WCAG SC 1.4.8).
```

**Arabic-specific rules:**

```
Arabic harakat (vowel marks) at 14px render at 2–3px height — illegible on standard screens.
Minimum 16px for Arabic with diacritics; 18px preferred.
Connected cursive creates variable stroke widths within words — verify Lc at thinnest stroke points.
```

**Diacritical mark thresholds:**

| Script | Feature | Minimum Size | Preferred Line-Height |
|---|---|---|---|
| Vietnamese | Stacked diacritics (e.g., ệ, ở) | 14px (16px pref.) | 1.7× |
| Thai | Up to 4 vertical stacking marks | 16px | 1.8× |
| Arabic | Harakat (shaddah + fathah stacking) | 16px (18px pref.) | 1.7× |
| Hebrew | Nikkud (vowel points) | 14px (16px pref.) | 1.5× |

Universal safe line-height for any script with diacritical marks: **1.7×**.

**Text expansion and line length:**

Translation from English expands text, which pushes line length toward the upper bound where color fatigue increases:

| Target Language | Expansion (medium strings) | Line Length Adjustment |
|---|---|---|
| German | +20–35% | Reduce max-width to ~58ch |
| Finnish | +30–40% | Reduce max-width to ~55ch |
| French, Spanish | +15–20% | Reduce max-width to ~62ch |
| Russian | +20% | Reduce max-width to ~60ch |
| CJK | −20% char count (full em-width) | Use max-width: 40ch (WCAG) |

**Font fallback and metric consistency:**

System CJK fonts across platforms (Hiragino Sans, Yu Gothic, PingFang, Malgun Gothic) have different ascent/descent ratios. A fallback font with a smaller x-height drops the effective visual size by ~13%, potentially failing contrast checks that passed for the primary font.

Mitigation with CSS `@font-face` descriptors:

```css
@font-face {
  font-family: 'CJK Fallback';
  src: local('Noto Sans CJK SC'), local('Hiragino Sans');
  size-adjust: 113%;       /* Compensate for x-height mismatch */
  ascent-override: 88%;    /* Normalize vertical metrics */
}
```

**Font loading performance budgets:**

| Scope | WOFF2 Budget | Strategy |
|---|---|---|
| Latin-only | < 100 KB | Variable font, subset Latin + Latin Extended |
| Latin + 1 CJK region | < 500 KB | unicode-range subsetting via Google Fonts |
| Full multilingual (Noto) | Subset per locale | On-demand loading by detected script |

> *Note: WCAG and APCA do not define script-specific contrast ratios. The offsets and thresholds above are derived from stroke-density analysis and spatial frequency research. They represent engineering best practice, not specification requirements.*

### Accessibility Matrix

#### Color Vision Deficiency (CVD)

APCA solves **luminance contrast**. It does NOT solve **hue confusion**. Common CVD types (protan, deutan, tritan) have normal luminance perception — their problem is that certain hue pairs collapse to the same perceived color. Hue confusion must be solved through redundant encoding, not contrast algorithms.

**Redundant encoding rule:**

```
Every piece of information conveyed by color MUST also be conveyed by
at least one non-color channel: icon shape, text label, pattern, position, or underline.
```

**Specific CVD hazards:**

| CVD Type | Prevalence | Confused Pairs | Critical Hazard |
|---|---|---|---|
| Protanopia (no L-cone) | ~1% male | Red↔green, red↔black, red↔brown | Red on black: appears near-zero contrast |
| Deuteranopia (no M-cone) | ~1% male | Red↔green, green↔brown | Same red-green confusion as protan |
| Tritanopia (no S-cone) | ~0.01% | Blue↔yellow, blue↔green | Rare; covered by redundant encoding |
| Achromatopsia | ~0.003% | All hue pairs | Relies entirely on luminance contrast |

**Safe categorical palette (Okabe-Ito):**

For data visualization or any context requiring ≥6 distinguishable categories under all common CVD types:

| Name | Hex | Approximate OKLCH |
|---|---|---|
| Black | #000000 | L=0 C=0 H=0 |
| Orange | #E69F00 | L=0.74 C=0.15 H=75 |
| Sky Blue | #56B4E9 | L=0.72 C=0.10 H=230 |
| Bluish Green | #009E73 | L=0.60 C=0.13 H=170 |
| Yellow | #F0E442 | L=0.92 C=0.17 H=100 |
| Blue | #0072B2 | L=0.50 C=0.12 H=245 |
| Vermillion | #D55E00 | L=0.56 C=0.18 H=40 |
| Reddish Purple | #CC79A7 | L=0.62 C=0.10 H=350 |

**Source:** Okabe & Ito (2002), "Color Universal Design."

#### Low Vision

APCA introduces **contrast reserve** — the gap between bare legibility (just-noticeable difference) and fluent reading speed. Body text needs Lc 75 minimum for legibility, Lc 90 for comfortable sustained reading. Low vision users (20/70 to 20/200 acuity) operate with reduced contrast sensitivity and benefit disproportionately from the higher end of this range.

| Text Role | Min |Lc| | Preferred |Lc| | Min x-height | Min Weight |
|---|---|---|---|---|
| Body text | 75 | 90 | 9px | 400 |
| Captions, secondary | 60 | 75 | 7px | 400 |
| Headlines | 45 | 60 | — | 600+ |
| Interactive (buttons, links) | 60 | 75 | 9px | 500 |

#### Dyslexia (5–10% of Population)

OpenDyslexic and similar "dyslexia fonts" have **no robust evidence of benefit** (Rello & Baeza-Yates 2013; Kuster et al. 2018). What does measurably improve dyslexic reading speed:

```
letter-spacing: >= 0.12em
word-spacing:   >= 0.16em
line-height:    >= 1.5×
font-family:    sans-serif (no italic for body text)
background:     cream/pastel preferred over pure white (reduces glare)
contrast:       CR 15–18:1 preferred over maximum 21:1 (avoids "halation" on white)
text-align:     left (never justified — uneven word spacing disrupts reading)
```

These thresholds align with WCAG SC 1.4.12 (Text Spacing) user override requirements — a palette designed for dyslexic readability automatically passes SC 1.4.12.

**Source:** Rello & Baeza-Yates (2013), "Good fonts for dyslexia," *ASSETS '13*.

#### Aging (60+ Population)

By age 80, less than one-third of blue light (480 nm) passes through the yellowed crystalline lens. Pupil diameter decreases from ~6.6 mm (age 20–30) to ~5.3 mm (age 50+), reducing retinal illumination. Combined with presbyopia pushing screens farther away (reducing effective font size):

```
Rules for aging-accessible palettes:
  - Body text: >= 18px, weight >= 400
  - Contrast: increase Lc targets by 15 over baseline (body text |Lc| 90+)
  - Blue dependence: never use blue as the sole differentiator between states
  - Blue text on dark backgrounds: particularly poor for aging eyes;
    shift toward cyan (hue ~195°) or green to maintain luminance
```

#### Cognitive Load Limits

Evidence converges on maximum complexity budgets beyond which comprehension degrades:

```
Font families:       1–2 maximum (3 with monospace for code)
Emphasis colors:     2–3 maximum (beyond neutral scale)
Categorical colors:  6–8 maximum (charts, tags, status indicators)
Line length:         50–75 characters (66 ideal)
Motion:              respect prefers-reduced-motion — no parallax, auto-play, or rapid transitions
```

Font pairing: use contrasting strategy (different classifications with complementary
personalities, e.g., serif heading + sans body). Concordant pairing (two similar fonts)
risks monotony. Conflicting pairing (two display fonts) creates visual noise. A typical
three-font system: display/personality heading + neutral body + monospace code.

~35% of users report motion sensitivity. The `prefers-reduced-motion` media query is essential for any animation or color transition.

#### User Style Overrides

Users may activate Windows High Contrast Mode (now "Contrast Themes"), browser zoom, text-only zoom, or custom stylesheets. A well-designed token system must survive these overrides.

**Detection:**

```css
@media (forced-colors: active) {
  /* System colors replace all custom colors.
     Borders, outlines, and text decorations remain visible.
     Background-color differentiation is lost. */
}
```

**Design rules for override resilience:**

- Use `outline: 3px solid transparent` for focus states — invisible normally, becomes visible in forced-colors mode:

```css
:focus-visible {
  outline: 3px solid transparent;
  outline-offset: 2px;
  box-shadow: 0 0 0 3px var(--primary);
}

@media (forced-colors: active) {
  :focus-visible {
    outline-color: Highlight;
    box-shadow: none;
  }
}
```

- WCAG SC 1.4.12 requires that user-overridden text spacing (line-height 1.5×, letter-spacing 0.12em, word-spacing 0.16em, paragraph-spacing 2×) does not break layout. Never use fixed `height` on text containers; use `min-height` or no height constraint.
- Token naming must be purpose-based (`--text-primary`, `--bg-surface`), not appearance-based (`--dark-gray`, `--light-bg`). Purpose-based names remain meaningful after user overrides remap the actual colors.

#### Color-Only Information (WCAG 1.4.1)

```
Triple encoding rule:
  Every status indicator = color + icon + text

  Error:   red border + ⚠ icon + "Error: ..." text
  Success: green border + ✓ icon + "Success: ..." text
  Warning: amber border + △ icon + "Warning: ..." text

  Links: must have underline (or equivalent non-color indicator), not just color
  Charts: must have shapes/patterns/labels, not just color-coded bars/lines
```

#### Photosensitivity

Saturated red flashes pose the highest seizure risk:

```
Red flash threshold:
  R / (R + G + B) >= 0.8 AND chromaticity shift > 0.2 → seizure risk

General flash threshold:
  Maximum 3 flashes per second
  Flashing area must be < 25% of a 10° visual field (~341 × 256 CSS pixels)

Mitigation:
  prefers-reduced-motion disables all flashing and animation
  No large-area saturated red transitions at any speed
```

**Source:** WCAG SC 2.3.1; Harding & Harding (2010).

#### Dark Mode Accessibility

Astigmatism affects 30–60% of the population. On dark backgrounds, pupil dilation increases optical aberrations, causing bright text to visually "bleed" into surrounding dark pixels (**halation**). This effect compounds with thin font weights and high contrast.

```
Dark mode rules:
  Background:   never pure black (#000). Use #121212 to #1E1E1E (OKLCH L 0.15–0.18).
  Body text:    never pure white. Cap at neutral-200 range (OKLCH L ~0.87).
  Chroma:       reduce text and surface chroma by 20–30% (multiply C by 0.7–0.8).
  Body text Lc: cap at |Lc| 85–90 (pure white on pure black = Lc ~106, which is excessive).
  Weight:       use weight >= 400 for body text; increase by one stop vs. light mode.
  Size:         increase font-size by 1–2px vs. light mode for equivalent readability.
  Light toggle: always offer a light mode alternative — dark mode is not universally better.
  Polarity:     APCA Lc values differ for light-on-dark vs. dark-on-light.
                Always compute for actual polarity, never assume symmetry.
```

### Variable Fonts as Accessibility Tools

Three variable font axes directly interact with contrast requirements:

**Optical size (`opsz`):** Automatically adjusts glyph design for the intended display size. At small sizes (8–14), the font thickens stems, opens counters, and increases x-height — effectively moving leftward (heavier) in the APCA weight table without changing `font-weight`. Enable with:

```css
body { font-optical-sizing: auto; }
```

Fonts with `opsz` axis: Inter, Roboto Flex, Source Sans 3, Segoe UI Variable, SF Pro.

**Grade (`GRAD`):** Changes perceived stroke density without altering character advance widths (no layout reflow). Use cases:
- Dark mode: increase grade by +50 to counteract halation-induced perceived weight loss
- Hover/active states: change perceived emphasis without layout shift

```css
[data-theme='dark'] body {
  font-variation-settings: 'GRAD' 50;
}
```

**Continuous weight (`wght`):** Variable fonts support any weight value (not just 100-step increments). This allows precise targeting of the exact stroke thickness needed to meet a specific Lc threshold at a given size, rather than rounding to 400 or 700.

### Typography Layout and Color Fatigue

Typographic layout properties interact with color to affect readability and eye strain during sustained reading:

**Line length:**

```
Optimal: 45–75 characters (66 ideal)
Implementation: max-width: 66ch on text containers
CJK: max-width: 40ch (WCAG SC 1.4.8)
```

**Line height by context:**

| Context | Line Height | Rationale |
|---|---|---|
| Body text (Latin) | 1.4–1.6× | Saccade return accuracy over multiple lines |
| Body text (CJK, Thai, Devanagari) | 1.6–1.8× | Stroke density + diacritical clearance |
| Body text (Arabic with harakat) | 1.7× | Vowel mark clearance |
| Headings | 1.1–1.2× | Tight for visual cohesion at large sizes |
| Code blocks | 1.3–1.5× | Monospace character alignment |

**Color fatigue interactions:**

- High contrast + long lines (>75ch) = increased saccade fatigue. If line length exceeds 75ch and cannot be reduced, target Lc 80 rather than 90 for body text.
- Saturated backgrounds + sustained reading = chromatic adaptation stress. Use backgrounds at OKLCH C < 0.02 for content surfaces.
- **Paper reading experience** (Somers): Surround the text block with a neutral surface (`--bg-surface`) at ~85% of page background luminance, creating a comfortable luminance transition. This reduces peak luminance while maintaining high Lc within the reading area.

### Extended Validation Checklist (Phase 3)

For every palette applied to text content, verify in addition to the Phase 2 checklist:

**Contrast (dual-model):**
- [ ] Body text (14–16px, W400): |Lc| ≥ 75 AND CR ≥ 4.5:1
- [ ] Subheadings: |Lc| ≥ 60 AND CR ≥ 3.0:1 (large text)
- [ ] Headlines (32px+): |Lc| ≥ 45 AND CR ≥ 3.0:1
- [ ] Non-text UI (icons, borders): |Lc| ≥ 45 AND CR ≥ 3.0:1
- [ ] All WCAG/APCA discrepancies flagged and documented

**Typography:**
- [ ] Text chroma verified: body text C < 0.04, interactive C < 0.08
- [ ] X-height ratio of chosen font ≥ 0.5 (or effective_size compensation applied)
- [ ] Font-smoothing antialiased gated behind ≥192dpi media query
- [ ] Line length: max-width set to 66ch (or script-appropriate value)
- [ ] Line height: ≥ 1.4× for Latin body, ≥ 1.6× for CJK/Arabic/Devanagari

**Localization (if multilingual):**
- [ ] CJK body text: ≥ 16px, weight ≥ 400, APCA offset applied
- [ ] Arabic body text: ≥ 16px, harakat legible
- [ ] Diacritical scripts: line-height ≥ 1.7×
- [ ] Font fallback metrics adjusted (size-adjust, ascent-override)
- [ ] Text expansion tested for longest target languages

**Accessibility:**
- [ ] All color-only signals have redundant encoding (icon + text)
- [ ] Error/success/warning states use triple encoding
- [ ] Links have non-color indicators (underline or equivalent)
- [ ] Dark mode: no pure black background, body text |Lc| capped at 85–90
- [ ] Dark mode: chroma reduced 20–30% vs. light mode
- [ ] `@media (forced-colors: active)` tested — focus states and borders visible
- [ ] Text spacing override (SC 1.4.12) does not break layout
- [ ] No saturated-red flashing > 3 Hz
- [ ] `prefers-reduced-motion` respected

## References

### Color Science
- **Björn Ottosson** — OKLCH/OKLab color space specification (2020)
- **W3C** — WCAG 2.1 Success Criterion 1.4.3 (Contrast Minimum), 1.4.11 (Non-text Contrast)
- **ITU-R BT.709** — Relative luminance coefficients (R=0.2126, G=0.7152, B=0.0722)
- **CSS Color Level 4** — `oklch()` function specification
- **Bezold-Brücke effect** — Hue shift under varying luminance (psychophysics)
- **Material Design 3** — Paired foreground token pattern (`on-primary`, `on-error`)

### Branding & Marketing Science
- **Sharp, B.** — *How Brands Grow* (Oxford University Press, 2010). Empirical evidence that brand growth is driven by penetration and mental availability, not perceived differentiation. Based on Ehrenberg-Bass Institute data across thousands of brands and multiple decades.
- **Sharp, B.** — *How Brands Grow Part 2* (Oxford University Press, 2016). Extended findings on emerging markets and services. Reinforces that distinctiveness drives recall while differentiation claims are poorly perceived by consumers.
- **Romaniuk, J.** — *Building Distinctive Brand Assets* (Oxford University Press, 2018). Operational framework for creating and measuring Distinctive Brand Assets (DBAs). Introduces the Fame × Uniqueness grid and DBA linkage measurement methodology.
- **Ehrenberg-Bass Institute for Marketing Science** — University of South Australia. The empirical research institution behind Sharp and Romaniuk's work. Studies spanning FMCG, technology, services, and B2B markets.

### Color Emotion & Psychophysiology
- **Wilms, L., & Oberfeld, D.** — "Color and emotion: Effects of hue, saturation, and brightness" (*Psychological Research*, 2018). Most methodologically rigorous factorial study: 62 participants, 27 chromatic colors, CIE LCh colorimetric control, SAM + physiological measures. Saturation is the dominant driver of arousal (η² = .693). Hue did not significantly affect valence (p = .051).
- **Valdez, P., & Mehrabian, A.** — "Effects of color on emotions" (*Journal of Experimental Psychology: General*, 1994). Established that brightness is the strongest predictor of pleasure, saturation the strongest predictor of arousal. Set the methodological standard for independent manipulation of color dimensions.
- **Elliot, A. J., & Maier, M. A.** — "Color-in-context theory" (*Advances in Experimental Social Psychology*, 2012). Color's psychological effect is moderated by context — same color produces opposite responses (approach vs. avoidance) depending on the situation.
- **Jonauskaite, D., et al.** — "Universal patterns in color-emotion associations are further shaped by linguistic and geographic proximity" (*Psychological Science*, 2020). 4,598 participants across 30 nations. Cross-country agreement r = .88. Lightness-valence association confirmed cross-culturally.
- **Palmer, S. E., & Schloss, K. B.** — "An ecological valence theory of human color preferences" (*PNAS*, 2010). 80% of color preference variance explained by the affective valence of associated objects (WAVE model).
- **Mehrabian, A., & Russell, J. A.** — *An Approach to Environmental Psychology* (MIT Press, 1974). The Pleasure-Arousal-Dominance (PAD) framework for emotional response to environments.

### Color Psychology, Education & Brand Personality
- **SHIFT eLearning** — Research on color's impact on learning outcomes. Findings: intentional color schemes amplify learning by 55–78% and comprehension by up to 73%. Cool colors (blue, green, purple) promote focused learning moods.
- **Labrecque, L. I., & Milne, G. R.** — "Exciting red and competent blue: The importance of color in marketing" (*Journal of the Academy of Marketing Science*, 2012). Mapped hues to Aaker's brand personality dimensions. Saturation amplifies existing hue-personality associations.
- **Bottomley, P. A., & Doyle, J. R.** — "The interactive effects of colors and products on perceptions of brand logo appropriateness" (*Marketing Theory*, 2006). Color-product congruence matters more than color alone. Atypical colors grab attention but decrease purchase intent.
- **Adams, F. M., & Osgood, C. E.** — "A cross-cultural study of the affective meanings of color" (*Journal of Cross-Cultural Psychology*, 1973). Semantic differential scales across 23 cultures. Brightness-valence association stable cross-culturally.

### Typeface Personality & Classification
- **Brumberger, E.** — "The rhetoric of typography: The awareness and impact of typeface appropriateness" (*Technical Communication*, 2003). Factor analysis of 15 typefaces on 20 adjective pairs. Three personality dimensions: Elegance, Directness, Friendliness.
- **Shaikh, A.D., Chaparro, B.S., & Fox, D.** — "Perception of fonts: Perceived personality traits and uses" (*Usability News*, 2006). 78% consistency in font-domain matching. Monospace associated with programming contexts by 40% of participants.
- **Henderson, P.W., Giese, J.L., & Cote, J.A.** — "Impression management using typeface design" (*Journal of Marketing*, 2004). Typeface design characteristics systematically predict brand personality impressions.
- **Monotype & Neurons** — Cross-cultural typeface perception study (2023). N=1,957, 8 countries. Humanist sans scored highest for trust in 7/8 countries. Significant cultural modulation in serif/sans preference.
- **Arditi, A., & Cho, J.** — "Serifs and font legibility" (*Vision Research*, 2005). Minor serif advantage is a spacing artifact, not a serif benefit.
- **Lund, O.** — *Knowledge construction in typography: The case of legibility research and the legibility of sans serif typefaces* (PhD thesis, University of Reading, 1999). Review of 72 legibility studies: no meaningful serif vs sans-serif difference on screen.
- **Morris, R.A., et al.** — "Serifs slow RSVP reading at very small sizes but don't matter at larger sizes" (2002). Serifs become noise below ~10px; decorative signal at 18px+.
- **Fox, D., Shaikh, A.D., & Chaparro, B.S.** — "Effect of typeface appropriateness on the perception of documents" (2007). 22% credibility loss for incongruent typography.
- **Koch, B.E.** — *Emotional response to typographic design* (Doctoral dissertation, 2011). Emotional responses to type parallel color emotion on PAD-adjacent dimensions.

### Contrast & Typography Science
- **Somers, A.** — *APCA Readability Criterion* (W3C Silver/WCAG 3.0 draft, 2022–present). Accessible Perceptual Contrast Algorithm. Polarity-sensitive, font-size-aware contrast model for WCAG 3.0. Reference implementation: `apca-w3` npm package (Myndex).
- **Somers, A.** — "The Realities And Myths Of Contrast And Color" (*Smashing Magazine*, 2022). Comprehensive explanation of spatial frequency, APCA vs. WCAG 2.x failures, and font-size × weight interaction with contrast.
- **Myndex** — *APCA Silver Level Font Lookup Tables* (2023 draft). Minimum Lc values by font size and weight. Derived from visual acuity and spatial frequency research.
- **Fairchild, M. D., & Pirrotta, E.** — "Predicting the brightness of different hues" (1991). Quantification of the Helmholtz-Kohlrausch effect: chromatic colors appear brighter than achromatic colors of equal luminance.
- **Whittaker, S. G., & Lovie-Kitchin, J.** — Research on critical size, critical contrast, and contrast reserve for reading. Referenced throughout APCA documentation as the empirical foundation for Lc threshold calibration.

### Localization & Script Research
- **W3C Internationalization (i18n)** — Best practices for text sizing, line breaking, and vertical text across scripts. Script-specific typography considerations.
- **Noto Fonts** — Google's open-source font family covering 800+ languages. Variable font support for weight 100–900. CJK WOFF2 subset: ~200–500 KB with unicode-range.
- **WCAG SC 1.4.8** — Visual Presentation (AAA): specifies ≤40 characters per line for CJK scripts, ≤80 for Latin.

### Accessibility Research
- **Rello, L., & Baeza-Yates, R.** — "Good fonts for dyslexia" (*ASSETS '13*, 2013). No evidence OpenDyslexic improves readability; letter spacing and sans-serif fonts show measurable improvement.
- **Kuster, S. M., et al.** — "Dyslexie font does not benefit reading in children with or without dyslexia" (*Annals of Dyslexia*, 2018). Replication confirming no benefit from specialized dyslexia fonts.
- **Okabe, M., & Ito, K.** — "Color Universal Design (CUD): How to make figures and presentations that are friendly to colorblind people" (2002). The Okabe-Ito palette: 8 colors distinguishable under all common CVD types.
- **Birch, J.** — "Worldwide prevalence of red-green color deficiency" (*J. Optical Society of America A*, 2012). ~8% of males, ~0.5% of females.
- **Harding, G., & Harding, P.** — "Photosensitive epilepsy and image safety" (*Applied Ergonomics*, 2010). Flash frequency and saturated-red area thresholds for seizure risk.
- **WCAG SC 1.4.1** — Use of Color: color must not be the only visual means of conveying information.
- **WCAG SC 1.4.11** — Non-text Contrast: UI components and graphical objects require 3:1 minimum contrast ratio.
- **WCAG SC 1.4.12** — Text Spacing: users must be able to override line-height (1.5×), letter-spacing (0.12em), word-spacing (0.16em), paragraph-spacing (2×) without loss of content or functionality.
- **WCAG SC 2.3.1** — Three Flashes or Below Threshold: no content flashes more than 3 times per second.
