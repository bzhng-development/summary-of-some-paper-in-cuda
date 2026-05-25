'use client';

import Link from 'next/link';
import { useState } from 'react';

import GradientCard from 'components/shared/gradient-card/gradient-card';

const EDGE_LABEL = {
  'llm-similar': 'LLM-judged similar',
  'category-chronology': 'in-domain predecessor',
  topic: 'topic thread',
  similarity: 'similar topic',
};

const Node = ({ node, depth, direction }) => {
  const [open, setOpen] = useState(depth < 1);
  const hasChildren = node.children && node.children.length > 0;

  return (
    <li className="relative">
      <div className="flex items-baseline gap-2">
        {hasChildren ? (
          <button
            type="button"
            onClick={() => setOpen((v) => !v)}
            aria-expanded={open}
            className="mt-1 inline-flex h-5 w-5 shrink-0 items-center justify-center rounded border border-gray-new-20 bg-gray-new-10 font-mono text-[10px] leading-none text-gray-new-60 transition-colors hover:border-primary-1/50 hover:text-primary-1"
          >
            {open ? '−' : '+'}
          </button>
        ) : (
          <span className="mt-1 inline-block h-5 w-5 shrink-0" aria-hidden />
        )}

        <span className="min-w-0 flex-1">
          <Link
            href={`/p/${encodeURIComponent(node.paper.category)}/${encodeURIComponent(node.paper.slug)}`}
            className="group flex flex-wrap items-baseline gap-x-2 gap-y-0.5"
          >
            <span className="t-sm font-medium text-white group-hover:text-primary-1">
              {node.paper.title}
            </span>
            <span className="t-sm font-mono text-gray-new-50">{node.paper.year}</span>
            <span className="t-sm font-mono text-xs text-gray-new-50">· {node.paper.category}</span>
          </Link>
          {node.edge ? (
            <span className="t-sm mt-0.5 block font-mono text-[10px] tracking-wider text-gray-new-50 uppercase">
              {direction === 'forward' ? '→' : '←'} {EDGE_LABEL[node.edge.type] ?? node.edge.type}
            </span>
          ) : null}
        </span>
      </div>

      {hasChildren && open ? (
        <ul className="mt-1.5 ml-2.5 flex flex-col gap-2 border-l border-gray-new-15 pl-3">
          {node.children.map((child, i) => (
            <Node
              key={`${child.paper.id}-${i}`}
              node={child}
              depth={depth + 1}
              direction={direction}
            />
          ))}
        </ul>
      ) : null}
    </li>
  );
};

const DependencyTree = ({ title, description, tree, direction = 'backward' }) => {
  if (!tree) return null;
  const empty = !tree.children || tree.children.length === 0;
  return (
    <section>
      <header className="mb-3">
        <h2 className="t-sm font-mono tracking-[0.2em] text-gray-new-50 uppercase">{title}</h2>
        {description ? (
          <p className="t-sm mt-1 max-w-2xl text-gray-new-70">{description}</p>
        ) : null}
      </header>
      <GradientCard className="p-4 sm:p-3">
        {empty ? (
          <p className="t-sm font-mono text-gray-new-50">No connections found.</p>
        ) : (
          <ul className="flex flex-col gap-2">
            {tree.children.map((child, i) => (
              <Node
                key={`${child.paper.id}-${i}`}
                node={child}
                depth={0}
                direction={direction}
              />
            ))}
          </ul>
        )}
      </GradientCard>
    </section>
  );
};

export default DependencyTree;
