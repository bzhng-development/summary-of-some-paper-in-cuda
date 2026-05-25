// Centralized localStorage key registry. Bump the version suffix when the
// stored shape changes incompatibly so old data is simply ignored.

export const STORAGE_KEYS = {
  recent: 'pg.recent.v1',
  read: 'pg.read.v1',
  streak: 'pg.streak.v1',
  queue: 'pg.queue.v1',
  resume: 'pg.resume.v1',
  highlights: 'pg.highlights.v1',
  notes: 'pg.notes.v1',
};

// Custom event names used to fan out cross-component state changes inside
// the same tab without forcing a navigation. Storage events handle cross-tab.
export const STORAGE_EVENTS = {
  read: 'pg:read-changed',
  queue: 'pg:queue-changed',
  streak: 'pg:streak-changed',
  highlights: 'pg:highlights-changed',
  notes: 'pg:notes-changed',
};
