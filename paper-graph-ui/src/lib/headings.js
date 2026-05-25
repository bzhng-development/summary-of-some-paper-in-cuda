import { cache } from 'react';
import remarkParse from 'remark-parse';
import slugify from 'slugify';
import { unified } from 'unified';
import { visit } from 'unist-util-visit';

// Matches AnchorHeading's algorithm so TOC IDs line up with rendered anchors.
const toSlug = (text) =>
  slugify(text.replace(/\(#[^)]+\)$/, ''), {
    lower: true,
    strict: true,
    remove: /[*+~.()'"!:@]/g,
  }).replace(/_/g, '');

const plainText = (node) => {
  if (typeof node.value === 'string') return node.value;
  if (!node.children) return '';
  return node.children.map(plainText).join('');
};

// Quick reject: no line starts with `## ` or `### ` => nothing to extract.
// Saves a full mdast parse for psets without section structure.
const HAS_HEADING_RE = /^#{2,3} /m;

// Extract h2/h3 headings from markdown into the shape TableOfContents expects.
// Returns: [{ title, id, level: 2, items: [{ title, id, level: 3 }, ...] }]
// h2 items omit `items` when no h3 children follow (smaller RSC payload).
export const extractToc = cache((markdown) => {
  if (!markdown || !HAS_HEADING_RE.test(markdown)) return [];
  const tree = unified().use(remarkParse).parse(markdown);
  const flat = [];
  visit(tree, 'heading', (node) => {
    if (node.depth !== 2 && node.depth !== 3) return;
    const title = plainText(node).trim();
    if (!title) return;
    flat.push({ title, id: toSlug(title), level: node.depth });
  });

  const out = [];
  for (const h of flat) {
    if (h.level === 2) {
      out.push(h);
    } else if (out.length > 0) {
      const parent = out[out.length - 1];
      if (!parent.items) parent.items = [];
      parent.items.push(h);
    }
  }
  return out;
});
