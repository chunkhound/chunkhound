import { defineConfig } from 'astro/config';
import { SHIKI_THEMES, COPY_SVG } from './src/lib/shiki-config.js';

export default defineConfig({
  site: 'https://chunkhound.ai',
  // The site is hand-written HTML that relies on HTML's own inline-flow
  // whitespace (the hero's headline + trailing CTA, prose + <code>/<strong>/<a>).
  // Astro 7's default `compressHTML: "jsx"` applies JSX whitespace rules and
  // strips newline-adjacent spaces, gluing inline siblings ("laptop.Build",
  // "custombase_url"). `true` compresses losslessly and keeps render-affecting
  // spaces. Never set this back to "jsx". Guarded by
  // tests/site/test_inline_spacing_contract.py.
  compressHTML: true,
  // The enterprise roadmap page predated the product architecture docs and is
  // superseded by it; keep old links working.
  redirects: {
    '/enterprise/architecture': '/docs/architecture/',
  },
  markdown: {
    // Astro 7 handles GitHub-Flavored Markdown natively, so remark-gfm is no longer needed here.
    shikiConfig: {
      // Code blocks intentionally keep a dark code surface and the dark Shiki
      // token palette in both site themes. We still emit Shiki's dual-theme
      // variables because Astro's renderer expects them, but the site
      // stylesheet always resolves rendered code to the dark token set.
      themes: SHIKI_THEMES,
      defaultColor: false,
      transformers: [{
        pre(node) {
          const rawCode = this.source;
          return {
            type: 'element',
            tagName: 'div',
            properties: { class: 'code-block-md' },
            children: [
              {
                type: 'element',
                tagName: 'button',
                properties: {
                  class: 'copy-btn',
                  type: 'button',
                  'aria-label': 'Copy code',
                  'data-copy': rawCode,
                },
                children: [{ type: 'raw', value: COPY_SVG }],
              },
              node,
            ],
          };
        },
      }],
    },
  },
});
