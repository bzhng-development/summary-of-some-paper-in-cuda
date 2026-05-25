#!/usr/bin/env node
// Walk imports starting from the project's real entry points and print
// every project-local file that is transitively reachable.
//
// Output (stdout): one absolute path per line, sorted, unique.
//
// Heuristic (string-based, not a real parser): good enough for this codebase.

import fs from 'node:fs';
import path from 'node:path';

const ROOT = process.cwd();
const SRC = path.join(ROOT, 'src');

const ENTRIES = [
  'src/app/layout.jsx',
  'src/app/page.jsx',
  'src/app/not-found.jsx',
  'src/app/timeline/page.jsx',
  'src/app/graph/page.jsx',
  'src/app/c/[category]/page.jsx',
  'src/app/p/[category]/[slug]/page.jsx',
  'src/app/topic/[topic]/page.jsx',
  'src/app/fonts.js',
  'src/styles/globals.css',
  'src/styles/app.css',
  'src/lib/papers.js',
  'src/lib/headings.js',
  'src/lib/shiki.js',
  'src/lib/rehype-code-props.js',
  'next.config.ts',
  'tailwind.config.js',
  'postcss.config.js',
  'empty.js',
];

const JS_EXTS = ['.jsx', '.tsx', '.js', '.ts', '.mjs', '.cjs'];
const EXTS_TO_TRY = ['', ...JS_EXTS, '.css', '.svg', '.png', '.jpg', '.jpeg', '.webp', '.gif', '.mp4', '.webm', '.woff', '.woff2', '.ttf', '.json'];
const INDEX_CANDIDATES = ['index.js', 'index.jsx', 'index.ts', 'index.tsx'];

const seen = new Set();

const existsFile = (p) => {
  try { return fs.statSync(p).isFile(); } catch { return false; }
};
const existsDir = (p) => {
  try { return fs.statSync(p).isDirectory(); } catch { return false; }
};

// Resolve an import specifier against the containing file.
// Returns an absolute path (or null for bare packages / unresolvable).
const resolveSpec = (spec, fromFile) => {
  if (!spec) return null;
  // bare packages: start with letter or @ and don't look like path/alias
  const isRelative = spec.startsWith('./') || spec.startsWith('../') || spec.startsWith('/');
  const isAlias =
    spec.startsWith('components/') ||
    spec.startsWith('hooks/') ||
    spec.startsWith('utils/') ||
    spec.startsWith('lib/') ||
    spec.startsWith('constants/') ||
    spec.startsWith('contexts/') ||
    spec.startsWith('icons/') ||
    spec.startsWith('images/') ||
    spec.startsWith('styles/') ||
    spec.startsWith('app/') ||
    spec.startsWith('fonts/') ||
    spec.startsWith('config/') ||
    spec.startsWith('generated/');

  let base;
  if (isRelative) {
    base = path.resolve(path.dirname(fromFile), spec);
  } else if (isAlias) {
    base = path.join(SRC, spec);
  } else if (spec.startsWith('../../../../content/')) {
    // handled by isRelative branch above
    base = path.resolve(path.dirname(fromFile), spec);
  } else {
    return null;
  }

  // Already has a known extension? (eg .svg, .css)
  if (path.extname(base)) {
    return existsFile(base) ? base : null;
  }

  // Try file + extensions
  for (const ext of JS_EXTS.concat(['.css'])) {
    if (existsFile(base + ext)) return base + ext;
  }

  // Directory with index
  if (existsDir(base)) {
    for (const idx of INDEX_CANDIDATES) {
      const p = path.join(base, idx);
      if (existsFile(p)) return p;
    }
  }

  return null;
};

const IMPORT_RE = /(?:import|export)[^'"`]+from\s+['"]([^'"]+)['"]/g;
const REQUIRE_RE = /require\(['"]([^'"]+)['"]\)/g;
const SIDE_IMPORT_RE = /^\s*import\s+['"]([^'"]+)['"]/gm;
const DYNAMIC_IMPORT_RE = /import\(\s*['"]([^'"]+)['"]\s*\)/g;
const CSS_IMPORT_RE = /@import\s+['"]([^'"]+)['"]/g;
const CSS_CONFIG_RE = /@config\s+['"]([^'"]+)['"]/g;

const extractSpecs = (code, isCss) => {
  const out = new Set();
  const patterns = isCss
    ? [CSS_IMPORT_RE, CSS_CONFIG_RE]
    : [IMPORT_RE, REQUIRE_RE, SIDE_IMPORT_RE, DYNAMIC_IMPORT_RE];
  for (const re of patterns) {
    re.lastIndex = 0;
    let m;
    while ((m = re.exec(code)) !== null) out.add(m[1]);
  }
  return [...out];
};

const visit = (absPath) => {
  if (!absPath || seen.has(absPath)) return;
  if (!existsFile(absPath)) return;
  seen.add(absPath);
  const ext = path.extname(absPath);
  if (!JS_EXTS.includes(ext) && ext !== '.css') return;
  const code = fs.readFileSync(absPath, 'utf8');
  const specs = extractSpecs(code, ext === '.css');
  for (const spec of specs) {
    const resolved = resolveSpec(spec, absPath);
    if (resolved) visit(resolved);
  }
};

for (const entry of ENTRIES) {
  const abs = path.join(ROOT, entry);
  if (existsFile(abs)) visit(abs);
}

const out = [...seen].sort();
for (const p of out) console.log(p);
