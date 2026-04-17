import { defineConfig } from 'astro/config';
import remarkGfm from 'remark-gfm';
import { SHIKI_THEMES, COPY_SVG } from './src/lib/shiki-config.js';

export default defineConfig({
  site: 'https://chunkhound.ai',
  markdown: {
    remarkPlugins: [remarkGfm],
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
