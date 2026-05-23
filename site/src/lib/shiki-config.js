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
  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 256" fill="currentColor" aria-hidden="true">
    <path d="M216 40v128H88V40Z" opacity="0.2"></path>
    <path d="M216 32H88a8 8 0 0 0-8 8v40H40a8 8 0 0 0-8 8v128a8 8 0 0 0 8 8h128a8 8 0 0 0 8-8v-40h40a8 8 0 0 0 8-8V40a8 8 0 0 0-8-8Zm-56 176H48V96h112Zm48-48h-32V88a8 8 0 0 0-8-8H96V48h112Z"></path>
  </svg>
`;
