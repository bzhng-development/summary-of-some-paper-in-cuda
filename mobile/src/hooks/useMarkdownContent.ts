import { useState, useEffect } from 'react';
import * as FileSystem from 'expo-file-system/next';

// Papers are stored in the sibling paper-graph-ui repo's content directory.
// During dev/build we load via Metro's asset system or we include a pre-bundled
// text asset. For the mobile build we bundle the markdown files as assets.
// The paper bodies live under assets/papers/<category>/<slug>.md

export function useMarkdownContent(category: string, slug: string) {
  const [content, setContent] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);

    // Papers are bundled as static assets under assets/papers/
    // Metro resolves them via require() at build time — see _layout.tsx's
    // asset registration. Here we use the bundled require map.
    const key = `${category}/${slug}`;
    loadPaperContent(key)
      .then((text) => {
        if (!cancelled) {
          setContent(text);
          setLoading(false);
        }
      })
      .catch((err) => {
        if (!cancelled) {
          setError(String(err));
          setLoading(false);
        }
      });

    return () => { cancelled = true; };
  }, [category, slug]);

  return { content, loading, error };
}

// Asset map: built at prebuild time by scripts/build-graph.mjs which also
// copies paper .md files into assets/papers/. Fallback: empty body.
async function loadPaperContent(key: string): Promise<string> {
  try {
    // Try loading from bundled assets directory
    const assetUri = `${FileSystem.documentDirectory}../assets/papers/${key}.md`;
    const file = new FileSystem.File(assetUri);
    if (await file.exists()) {
      return await file.text();
    }
  } catch {
    // pass
  }
  // Return a placeholder if file not found
  return `# Paper not found\n\nThe content for **${key}** could not be loaded.\n\nThis may happen if the asset bundle was not rebuilt. Run \`npm run build:graph\` and rebuild the app.`;
}
