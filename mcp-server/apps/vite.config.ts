import { defineConfig } from "vite";
import { viteSingleFile } from "vite-plugin-singlefile";

export default defineConfig({
  plugins: [viteSingleFile()],
  build: {
    emptyOutDir: true,
    rollupOptions: {
      input: process.env.INPUT,
      output: {
        // Force the output filename to be index.html
        entryFileNames: '[name].js',
        assetFileNames: '[name].[ext]',
      },
    },
  },
});
