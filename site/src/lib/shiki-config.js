import githubDarkDimmed from "@shikijs/themes/github-dark-dimmed";
import githubLight from "@shikijs/themes/github-light";

const COMMENT_TOKEN_COLOR = "#8B949E";
const COMMENT_SCOPES = new Set([
  "comment",
  "punctuation.definition.comment",
  "string.comment",
]);

function withSharedCommentToken(theme) {
  return {
    ...theme,
    tokenColors: theme.tokenColors.map((rule) => {
      const scopes = Array.isArray(rule.scope) ? rule.scope : [rule.scope];
      if (!scopes.some((scope) => COMMENT_SCOPES.has(scope))) {
        return rule;
      }
      return {
        ...rule,
        settings: {
          ...rule.settings,
          foreground: COMMENT_TOKEN_COLOR,
        },
      };
    }),
  };
}

// The site always renders code blocks on a dark code surface with the dark
// token palette for readability and visual consistency, even when the
// surrounding page is light. We keep the dual-theme Shiki shape because Astro
// emits the expected CSS variables from it, but the runtime stylesheet
// deliberately resolves rendered code to the dark token set.
export const SHIKI_THEMES = {
  light: withSharedCommentToken(githubLight),
  dark: withSharedCommentToken(githubDarkDimmed),
};

// Demo snippets should stay aligned with the dark Shiki palette and dark code
// surface used by the rendered docs and marketing code blocks.
export const DEMO_CODE_PALETTE = {
  cmd: "#F69D50",
  string: "#96D0FF",
  op: "#6CB6FF",
  jsonKey: "#F69D50",
  comment: COMMENT_TOKEN_COLOR,
  text: "#ADBAC7",
};

export const COPY_SVG = `
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
    <rect x="9" y="9" width="10" height="10" rx="2"></rect>
    <path d="M5 15V7a2 2 0 0 1 2-2h8"></path>
  </svg>
`;
