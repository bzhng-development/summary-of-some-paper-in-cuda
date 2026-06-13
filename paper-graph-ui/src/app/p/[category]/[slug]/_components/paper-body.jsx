import Markdown from 'react-markdown';
import rehypeKatex from 'rehype-katex';
import rehypeSlug from 'rehype-slug';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';

import OverflowFade from './overflow-fade';

// react-markdown renders raw markdown with no JSX-tag inference. That's the
// point — the source files contain "<1%" and similar fragments that MDX
// rejects as malformed JSX. Plain markdown reading is what we want for
// paper summaries: code blocks, tables, math, no embedded React components.
//
// We wrap horizontally-overflowing children (code blocks, tables, KaTeX
// display equations) in a small client island that paints a fade gradient
// on the side that has more content offscreen.
const PaperBody = ({ markdown }) => (
  <Markdown
    remarkPlugins={[remarkGfm, remarkMath]}
    rehypePlugins={[rehypeSlug, rehypeKatex]}
    components={{
      // The H1 in the body is the paper title; we render it in the page
      // header, so collapse it here to avoid duplication.
      h1: () => null,
      table: ({ node: _node, ...props }) => (
        <OverflowFade className="table-wrapper">
          <table {...props} />
        </OverflowFade>
      ),
      pre: ({ node: _node, ...props }) => (
        <OverflowFade>
          <pre
            className="my-4 overflow-x-auto rounded-md border border-gray-new-15 bg-black-fog p-3 font-mono text-[12px] leading-snug text-gray-new-80"
            {...props}
          />
        </OverflowFade>
      ),
      code: ({ inline: _inline, className, children, ...props }) => {
        if (className?.startsWith('language-')) {
          return (
            <code className={`${className} block`} {...props}>
              {children}
            </code>
          );
        }
        return (
          <code
            className="rounded bg-gray-new-15 px-1.5 py-0.5 font-mono text-[0.92em] text-primary-1/90"
            {...props}
          >
            {children}
          </code>
        );
      },
      a: ({ node: _node, href, children, ...props }) => {
        const isExternal = typeof href === 'string' && /^https?:\/\//.test(href);
        return (
          <a
            href={href}
            target={isExternal ? '_blank' : undefined}
            rel={isExternal ? 'noopener noreferrer' : undefined}
            className="text-primary-1 underline decoration-primary-1/40 underline-offset-2 transition-colors hover:decoration-primary-1"
            {...props}
          >
            {children}
          </a>
        );
      },
    }}
  >
    {markdown}
  </Markdown>
);

export default PaperBody;
