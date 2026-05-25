'use client';

import { useEffect, useRef } from 'react';

import Highlights from './highlights';
import MarginNotes from './margin-notes';

// Wraps children (the rendered markdown body) and shares a ref into the
// Highlights + MarginNotes client components. Keeps the markdown
// server-rendered while giving us a stable DOM handle.
const PaperInteractions = ({ paperId, children }) => {
  const containerRef = useRef(null);

  // Apply a tiny opt-in: mark the container as our "annotated" surface so
  // browser styles for selection look intentional.
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return undefined;
    el.classList.add('pg-annotatable');
    return () => el.classList.remove('pg-annotatable');
  }, []);

  return (
    <>
      <div ref={containerRef}>{children}</div>
      <Highlights paperId={paperId} containerRef={containerRef} />
      <MarginNotes paperId={paperId} containerRef={containerRef} />
    </>
  );
};

export default PaperInteractions;
