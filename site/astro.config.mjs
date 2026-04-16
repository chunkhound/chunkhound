import { defineConfig } from 'astro/config';
import remarkGfm from 'remark-gfm';
import { SHIKI_THEME, COPY_SVG } from './src/lib/shiki-config.js';

export default defineConfig({
  site: 'https://chunkhound.ai',
  markdown: {
    remarkPlugins: [remarkGfm],
    shikiConfig: {
      theme: SHIKI_THEME,
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
