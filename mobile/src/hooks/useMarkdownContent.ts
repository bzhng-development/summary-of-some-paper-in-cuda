import { useState, useEffect } from 'react';
import { Asset } from 'expo-asset';
import { File } from 'expo-file-system';
import { PAPER_ASSETS } from '../lib/paper-asset-map';

/**
 * Load the full paper body for `<category>/<slug>` from Metro-bundled assets.
 *
 * Build-time pipeline:
 *  - scripts/build-graph.mjs copies every paper-graph-ui .md into assets/papers/
 *    and emits src/lib/paper-asset-map.ts with one `require()` per paper.
 *  - Metro statically resolves the requires and bundles the .md files as assets
 *    (metro.config.js adds 'md' to assetExts).
 *
 * Runtime: Asset.fromModule(...) gives us the bundle's local URI; we read it
 * with expo-file-system/next. Cached by Asset across the app lifecycle.
 */
export function useMarkdownContent(category: string, slug: string) {
  const [content, setContent] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);

    loadPaperContent(`${category}/${slug}`)
      .then((text) => {
        if (cancelled) return;
        setContent(text);
        setLoading(false);
      })
      .catch((err: unknown) => {
        if (cancelled) return;
        setError(String(err));
        setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [category, slug]);

  return { content, loading, error };
}

async function loadPaperContent(key: string): Promise<string> {
  const moduleId = PAPER_ASSETS[key];
  if (moduleId === undefined) {
    return `# Paper not found\n\nKey: **${key}** is not in the bundled asset map.\n\nThis usually means the app was built before this paper was added. Run \`npm run build:graph\` and rebuild.`;
  }
  const asset = Asset.fromModule(moduleId);
  if (!asset.localUri) {
    await asset.downloadAsync();
  }
  const uri = asset.localUri ?? asset.uri;
  if (!uri) {
    throw new Error(`Asset for ${key} has neither localUri nor uri`);
  }
  return new File(uri).text();
}
