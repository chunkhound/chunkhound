import { defineConfig } from "astro/config";
import remarkGfm from "remark-gfm";

export default defineConfig({
  site: 'http://localhost:4321',
  devToolbar: { enabled: false },
  markdown: {
    remarkPlugins: [remarkGfm],
    shikiConfig: {
      theme: 'github-light'
    }
  }
});
