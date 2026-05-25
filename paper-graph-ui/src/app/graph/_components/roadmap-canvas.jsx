'use client';

import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { useEffect, useMemo, useRef, useState } from 'react';

// Roadmap layout: time on the X axis, category lanes on the Y axis. Papers
// are dots; lineage edges are quadratic-bezier curves between them.
//
// All sizing is in SVG user units; the outer wrapper handles overflow-scroll
// on mobile and pan/zoom on desktop. We deliberately avoid a force layout —
// the strong temporal+categorical structure of the dataset gives a clean
// deterministic layout that beats spring relaxation on a small screen.

const LANE_HEIGHT = 28;
const YEAR_WIDTH_MIN = 60;   // narrow column for sparse years
const YEAR_WIDTH_MAX = 220;  // wider for dense ones
const PAPER_RADIUS = 5;
const LANE_PADDING_X = 12;
const HEADER_HEIGHT = 28;
const LABEL_WIDTH = 140;

// Pick a per-year column width that scales with paper density that year.
function yearWidths(papers, years) {
  const countByYear = new Map();
  for (const p of papers) {
    countByYear.set(p.year, (countByYear.get(p.year) ?? 0) + 1);
  }
  const maxCount = Math.max(1, ...countByYear.values());
  const widths = new Map();
  let cursor = LABEL_WIDTH;
  const starts = new Map();
  for (const y of years) {
    const c = countByYear.get(y) ?? 0;
    const w = Math.round(
      YEAR_WIDTH_MIN + ((YEAR_WIDTH_MAX - YEAR_WIDTH_MIN) * Math.sqrt(c)) / Math.sqrt(maxCount)
    );
    widths.set(y, w);
    starts.set(y, cursor);
    cursor += w;
  }
  return { widths, starts, total: cursor };
}

// Inside a (year, lane) cell, jitter the X by month and a tiny Y wobble to
// avoid stacking dots on top of each other when several papers landed in
// the same month within the same domain.
function layoutPapers(papers, years, lanes, widths, starts) {
  const laneOfCategory = new Map(lanes.map((c, i) => [c.slug, i]));
  const out = new Map();
  // Group by cell to know how many wobble slots each needs.
  const cellCount = new Map();
  for (const p of papers) {
    const k = `${p.year}|${p.category}`;
    cellCount.set(k, (cellCount.get(k) ?? 0) + 1);
  }
  const cellIdx = new Map();
  for (const p of papers) {
    const lane = laneOfCategory.get(p.category);
    if (lane == null) continue;
    const startX = starts.get(p.year);
    const width = widths.get(p.year);
    if (startX == null) continue;
    const monthFrac = ((p.month ?? 6) - 1) / 12; // 0..1
    const x = startX + LANE_PADDING_X + monthFrac * (width - 2 * LANE_PADDING_X);
    const k = `${p.year}|${p.category}`;
    const total = cellCount.get(k) ?? 1;
    const idx = cellIdx.get(k) ?? 0;
    cellIdx.set(k, idx + 1);
    // small vertical wobble: spread up to ±9px within the lane
    const wobble = total > 1 ? ((idx + 0.5) / total - 0.5) * (LANE_HEIGHT - 12) : 0;
    const y = HEADER_HEIGHT + lane * LANE_HEIGHT + LANE_HEIGHT / 2 + wobble;
    out.set(p.id, { x, y, paper: p });
  }
  return out;
}

const RoadmapCanvas = ({ papers, edges, categories }) => {
  const router = useRouter();
  const [hoverId, setHoverId] = useState(null);
  const [filterCats, setFilterCats] = useState(null); // null = all
  const [minScore, setMinScore] = useState(0);
  const [showEdges, setShowEdges] = useState(true);
  const scrollRef = useRef(null);

  const visiblePapers = useMemo(() => {
    return papers.filter(
      (p) =>
        (filterCats == null || filterCats.has(p.category)) &&
        (minScore === 0 || (p.score ?? 0) >= minScore)
    );
  }, [papers, filterCats, minScore]);

  const years = useMemo(() => {
    const set = new Set(visiblePapers.map((p) => p.year));
    return [...set].sort((a, b) => a - b);
  }, [visiblePapers]);

  const lanes = useMemo(() => {
    // only render lanes that have visible papers
    const used = new Set(visiblePapers.map((p) => p.category));
    return categories.filter((c) => used.has(c.slug));
  }, [categories, visiblePapers]);

  const { widths, starts, total } = useMemo(
    () => yearWidths(visiblePapers, years),
    [visiblePapers, years]
  );

  const positions = useMemo(
    () => layoutPapers(visiblePapers, years, lanes, widths, starts),
    [visiblePapers, years, lanes, widths, starts]
  );

  // Index edges by endpoint for hover-highlighting connected papers.
  const edgesById = useMemo(() => {
    const map = new Map();
    for (const e of edges) {
      if (!positions.has(e.source) || !positions.has(e.target)) continue;
      if (!map.has(e.source)) map.set(e.source, []);
      if (!map.has(e.target)) map.set(e.target, []);
      map.get(e.source).push(e);
      map.get(e.target).push(e);
    }
    return map;
  }, [edges, positions]);

  const visibleEdges = useMemo(
    () =>
      edges.filter(
        (e) => positions.has(e.source) && positions.has(e.target)
      ),
    [edges, positions]
  );

  const width = total + 20;
  const height = HEADER_HEIGHT + lanes.length * LANE_HEIGHT + 20;

  const catBySlug = useMemo(
    () => new Map(categories.map((c) => [c.slug, c])),
    [categories]
  );

  // After mount, scroll the lineage right so newest papers are visible by
  // default. The chart is wide and starts at 2014 on the left.
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollLeft = scrollRef.current.scrollWidth;
    }
  }, [width]);

  const connectedToHover = useMemo(() => {
    if (!hoverId) return null;
    const out = new Set([hoverId]);
    for (const e of edgesById.get(hoverId) ?? []) {
      out.add(e.source);
      out.add(e.target);
    }
    return out;
  }, [hoverId, edgesById]);

  // Curve drawing — quadratic bezier with control point biased toward the
  // midpoint, slightly arched. Direction-stable so backward edges arc up.
  function curve(a, b) {
    const mx = (a.x + b.x) / 2;
    const my = (a.y + b.y) / 2;
    const dx = Math.abs(b.x - a.x);
    const lift = Math.min(40, 8 + dx * 0.06);
    const direction = a.y <= b.y ? -1 : 1;
    return `M ${a.x} ${a.y} Q ${mx} ${my + direction * lift} ${b.x} ${b.y}`;
  }

  return (
    <div className="flex flex-col gap-3">
      <Filters
        categories={categories}
        filterCats={filterCats}
        setFilterCats={setFilterCats}
        minScore={minScore}
        setMinScore={setMinScore}
        showEdges={showEdges}
        setShowEdges={setShowEdges}
        totalVisible={visiblePapers.length}
        totalEdges={visibleEdges.length}
      />

      <div
        ref={scrollRef}
        className="relative w-full overflow-x-auto overflow-y-hidden rounded-xl border border-gray-new-15 bg-black-fog"
        style={{ maxHeight: '70vh' }}
        role="region"
        aria-label="Paper lineage roadmap"
      >
        <svg
          width={width}
          height={height}
          viewBox={`0 0 ${width} ${height}`}
          className="block"
          // Click is delegated per-node via the SVG <a> element below; the
          // svg root only needs to declare its semantic role.
        >
          {/* Year header bar — sticky-ish (we re-render it as a foreignObject
              with sticky CSS for the X axis only) */}
          {years.map((y) => {
            const x = starts.get(y);
            const w = widths.get(y);
            return (
              <g key={`yr-${y}`}>
                <rect
                  x={x}
                  y={0}
                  width={w}
                  height={HEADER_HEIGHT}
                  fill="#0c0d0d"
                />
                <text
                  x={x + w / 2}
                  y={HEADER_HEIGHT - 9}
                  textAnchor="middle"
                  className="font-mono"
                  fontSize="11"
                  fill="#71717A"
                >
                  {y}
                </text>
                {/* faint year gridline */}
                <line
                  x1={x}
                  x2={x}
                  y1={0}
                  y2={height}
                  stroke="rgba(255,255,255,0.04)"
                />
              </g>
            );
          })}

          {/* Lane labels (sticky-left via the parent's overflow + a colored
              left edge per lane) */}
          {lanes.map((c, i) => {
            const ly = HEADER_HEIGHT + i * LANE_HEIGHT;
            return (
              <g key={`lane-${c.slug}`}>
                <rect
                  x={0}
                  y={ly}
                  width={LABEL_WIDTH}
                  height={LANE_HEIGHT}
                  fill="#0c0d0d"
                />
                <rect
                  x={LABEL_WIDTH - 3}
                  y={ly + 6}
                  width={3}
                  height={LANE_HEIGHT - 12}
                  fill={c.color}
                  rx={1.5}
                />
                <Link href={`/c/${c.slug}`} className="hover:underline">
                  <text
                    x={12}
                    y={ly + LANE_HEIGHT / 2 + 4}
                    className="font-sans"
                    fontSize="12"
                    fill="#C9CBCF"
                    style={{ cursor: 'pointer' }}
                  >
                    {c.title}
                  </text>
                </Link>
                {/* lane separator */}
                <line
                  x1={LABEL_WIDTH}
                  x2={width}
                  y1={ly + LANE_HEIGHT}
                  y2={ly + LANE_HEIGHT}
                  stroke="rgba(255,255,255,0.04)"
                />
              </g>
            );
          })}

          {/* Lineage edges — render under the dots */}
          {showEdges
            ? visibleEdges.map((e, i) => {
                const a = positions.get(e.source);
                const b = positions.get(e.target);
                if (!a || !b) return null;
                const isHover =
                  connectedToHover &&
                  connectedToHover.has(e.source) &&
                  connectedToHover.has(e.target);
                const color =
                  e.type === 'llm-similar'
                    ? isHover
                      ? 'rgba(0,229,153,0.85)'
                      : 'rgba(0,229,153,0.30)'
                    : isHover
                    ? 'rgba(170,153,255,0.65)'
                    : 'rgba(140,140,160,0.10)';
                return (
                  <path
                    key={i}
                    d={curve(a, b)}
                    stroke={color}
                    strokeWidth={isHover ? 1.4 : e.type === 'llm-similar' ? 1 : 0.6}
                    fill="none"
                  />
                );
              })
            : null}

          {/* Papers — SVG <a> wrapping a <g>. Next.js Link inside SVG doesn't
              navigate reliably in all browsers; useRouter().push on click is
              the safe path. We still expose href on the anchor so right-click
              "open in new tab" works. */}
          {[...positions.values()].map(({ x, y, paper }) => {
            const cat = catBySlug.get(paper.category);
            const isHover = hoverId === paper.id;
            const dim =
              connectedToHover != null && !connectedToHover.has(paper.id);
            const href = `/p/${encodeURIComponent(paper.category)}/${encodeURIComponent(paper.slug)}`;
            return (
              <a
                key={paper.id}
                href={href}
                onClick={(ev) => {
                  // Let cmd/ctrl-click open in a new tab natively.
                  if (ev.metaKey || ev.ctrlKey || ev.shiftKey) return;
                  ev.preventDefault();
                  router.push(href);
                }}
                onPointerEnter={() => setHoverId(paper.id)}
                onPointerLeave={() => setHoverId(null)}
                onFocus={() => setHoverId(paper.id)}
                onBlur={() => setHoverId(null)}
                style={{ cursor: 'pointer' }}
              >
                <circle
                  cx={x}
                  cy={y}
                  r={isHover ? PAPER_RADIUS + 2 : PAPER_RADIUS}
                  fill={cat?.color ?? '#94979E'}
                  fillOpacity={dim ? 0.18 : 0.85}
                  stroke={isHover ? '#FFFFFF' : 'rgba(0,0,0,0.4)'}
                  strokeWidth={isHover ? 1.5 : 0.5}
                />
                {paper.score != null && paper.score >= 9 ? (
                  <circle
                    cx={x}
                    cy={y}
                    r={PAPER_RADIUS + 3}
                    fill="none"
                    stroke="#00E599"
                    strokeOpacity={dim ? 0.15 : 0.55}
                    strokeWidth={0.8}
                  />
                ) : null}
                <title>
                  {paper.title} ({paper.year}) — {cat?.title ?? paper.category}
                  {paper.score != null ? ` · score ${paper.score}` : ''}
                  {paper.organization ? ` · ${paper.organization}` : ''}
                </title>
              </a>
            );
          })}
        </svg>
      </div>

      {/* Hover detail readout — gives keyboard users a stable spot to see the
          paper title without relying on the native <title> tooltip alone. */}
      <HoverDetail
        positions={positions}
        hoverId={hoverId}
        catBySlug={catBySlug}
      />

      <Legend />
    </div>
  );
};

const Filters = ({
  categories,
  filterCats,
  setFilterCats,
  minScore,
  setMinScore,
  showEdges,
  setShowEdges,
  totalVisible,
  totalEdges,
}) => (
  <div className="flex flex-wrap items-center gap-x-4 gap-y-2 rounded-lg border border-gray-new-15 bg-gray-new-10 px-3 py-2 text-xs">
    <div className="flex items-center gap-2">
      <span className="font-mono text-[10px] tracking-wider text-gray-new-50 uppercase">
        Min score
      </span>
      <div className="flex gap-1">
        {[0, 7, 8, 9].map((s) => (
          <button
            key={s}
            type="button"
            onClick={() => setMinScore(s)}
            className={`rounded border px-1.5 py-0.5 font-mono text-[11px] transition-colors ${
              minScore === s
                ? 'border-primary-1/60 bg-primary-1/15 text-primary-1'
                : 'border-gray-new-20 text-gray-new-70 hover:text-white'
            }`}
          >
            {s === 0 ? 'any' : `≥${s}`}
          </button>
        ))}
      </div>
    </div>

    <label className="flex items-center gap-2 font-mono text-[11px] text-gray-new-70">
      <input
        type="checkbox"
        checked={showEdges}
        onChange={(ev) => setShowEdges(ev.target.checked)}
        className="accent-primary-1"
      />
      Show lineage
    </label>

    <button
      type="button"
      onClick={() => setFilterCats(null)}
      className={`rounded border px-2 py-0.5 font-mono text-[11px] tracking-wider uppercase transition-colors ${
        filterCats == null
          ? 'border-primary-1/60 bg-primary-1/15 text-primary-1'
          : 'border-gray-new-20 text-gray-new-70 hover:text-white'
      }`}
    >
      All domains
    </button>

    <details className="group inline-block">
      <summary className="cursor-pointer rounded border border-gray-new-20 px-2 py-0.5 font-mono text-[11px] tracking-wider text-gray-new-70 uppercase transition-colors group-open:border-primary-1/40 group-open:text-primary-1">
        Pick domains {filterCats ? `(${filterCats.size})` : ''}
      </summary>
      <div className="absolute z-10 mt-1 flex max-h-[40vh] w-[260px] flex-wrap gap-1 overflow-y-auto rounded-md border border-gray-new-20 bg-black-new p-2 shadow-lg">
        {categories.map((c) => {
          const active = filterCats?.has(c.slug);
          return (
            <button
              key={c.slug}
              type="button"
              onClick={() => {
                setFilterCats((prev) => {
                  const next = new Set(prev ?? []);
                  if (next.has(c.slug)) next.delete(c.slug);
                  else next.add(c.slug);
                  return next.size === 0 ? null : next;
                });
              }}
              className={`flex items-center gap-1 rounded border px-1.5 py-0.5 font-mono text-[10px] transition-colors ${
                active
                  ? 'border-primary-1/60 bg-primary-1/15 text-primary-1'
                  : 'border-gray-new-20 text-gray-new-70 hover:text-white'
              }`}
            >
              <span
                className="inline-block h-1.5 w-1.5 rounded-full"
                style={{ backgroundColor: c.color }}
                aria-hidden
              />
              {c.title}
            </button>
          );
        })}
      </div>
    </details>

    <span className="ml-auto font-mono text-[10px] tracking-wider text-gray-new-50 uppercase">
      {totalVisible} papers · {totalEdges} lineage edges
    </span>
  </div>
);

const HoverDetail = ({ hoverId, positions, catBySlug }) => {
  if (!hoverId) {
    return (
      <p className="font-mono text-[10px] tracking-wider text-gray-new-50 uppercase">
        Hover or tap a dot to inspect · pinch to zoom · scroll the canvas
      </p>
    );
  }
  const node = positions.get(hoverId);
  if (!node) return null;
  const cat = catBySlug.get(node.paper.category);
  return (
    <Link
      href={`/p/${encodeURIComponent(node.paper.category)}/${encodeURIComponent(node.paper.slug)}`}
      className="flex flex-wrap items-baseline gap-x-3 gap-y-0.5 rounded-md border border-primary-1/30 bg-primary-1/5 px-3 py-2 transition-colors hover:bg-primary-1/10"
    >
      <span className="t-sm font-medium text-white">{node.paper.title}</span>
      <span className="font-mono text-[11px] text-gray-new-60">
        {cat?.title ?? node.paper.category}
        {' · '}
        {node.paper.year}
        {node.paper.month ? `/${String(node.paper.month).padStart(2, '0')}` : ''}
        {node.paper.score != null ? ` · score ${node.paper.score}` : ''}
        {node.paper.organization ? ` · ${node.paper.organization}` : ''}
        {node.paper.readTimeMin ? ` · ~${node.paper.readTimeMin} min` : ''}
      </span>
    </Link>
  );
};

const Legend = () => (
  <div className="flex flex-wrap items-center gap-4 font-mono text-[10px] tracking-wider text-gray-new-50 uppercase">
    <span className="flex items-center gap-1.5">
      <svg width="20" height="6">
        <line x1="0" y1="3" x2="20" y2="3" stroke="rgba(0,229,153,0.75)" strokeWidth="1.4" />
      </svg>
      LLM-similar lineage
    </span>
    <span className="flex items-center gap-1.5">
      <svg width="20" height="6">
        <line x1="0" y1="3" x2="20" y2="3" stroke="rgba(170,153,255,0.65)" strokeWidth="0.8" />
      </svg>
      Same-domain successor
    </span>
    <span className="flex items-center gap-1.5">
      <svg width="14" height="14">
        <circle cx="7" cy="7" r="3.5" fill="#00E599" />
        <circle cx="7" cy="7" r="6" fill="none" stroke="#00E599" strokeOpacity="0.55" strokeWidth="0.8" />
      </svg>
      Score ≥ 9
    </span>
  </div>
);

export default RoadmapCanvas;
