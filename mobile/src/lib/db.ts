import { openDatabaseSync } from 'expo-sqlite';

// Use expo-sqlite/kv-store pattern for user state
// (read: mark-read set, recently-opened, streak counter)

const db = openDatabaseSync('papergraph.db');

// Initialize schema once
db.execSync(`
  PRAGMA journal_mode = WAL;
  CREATE TABLE IF NOT EXISTS kv (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
  );
  CREATE TABLE IF NOT EXISTS read_papers (
    id TEXT PRIMARY KEY,
    read_at INTEGER NOT NULL DEFAULT (strftime('%s', 'now'))
  );
  CREATE TABLE IF NOT EXISTS recent_papers (
    id TEXT PRIMARY KEY,
    opened_at INTEGER NOT NULL DEFAULT (strftime('%s', 'now'))
  );
`);

// --- KV helpers ---
function kvGet(key: string): string | null {
  const row = db.getFirstSync<{ value: string }>('SELECT value FROM kv WHERE key = ?', [key]);
  return row?.value ?? null;
}

function kvSet(key: string, value: string): void {
  db.runSync('INSERT OR REPLACE INTO kv (key, value) VALUES (?, ?)', [key, value]);
}

// --- Read-papers ---
export function isRead(paperId: string): boolean {
  return db.getFirstSync<{ id: string }>('SELECT id FROM read_papers WHERE id = ?', [paperId]) != null;
}

export function markRead(paperId: string): void {
  db.runSync(
    'INSERT OR REPLACE INTO read_papers (id, read_at) VALUES (?, strftime(\'%s\', \'now\'))',
    [paperId]
  );
  updateStreak();
}

export function unmarkRead(paperId: string): void {
  db.runSync('DELETE FROM read_papers WHERE id = ?', [paperId]);
}

export function getAllReadIds(): string[] {
  const rows = db.getAllSync<{ id: string }>('SELECT id FROM read_papers');
  return rows.map((r) => r.id);
}

// --- Recently-opened ---
const MAX_RECENT = 16;

export function recordOpened(paperId: string): void {
  db.runSync(
    'INSERT OR REPLACE INTO recent_papers (id, opened_at) VALUES (?, strftime(\'%s\', \'now\'))',
    [paperId]
  );
  // Trim to max
  db.runSync(
    `DELETE FROM recent_papers WHERE id NOT IN (
       SELECT id FROM recent_papers ORDER BY opened_at DESC LIMIT ?
     )`,
    [MAX_RECENT]
  );
}

export function getRecentIds(): string[] {
  const rows = db.getAllSync<{ id: string }>(
    'SELECT id FROM recent_papers ORDER BY opened_at DESC LIMIT ?',
    [MAX_RECENT]
  );
  return rows.map((r) => r.id);
}

// --- Streak ---
function updateStreak(): void {
  const today = new Date().toISOString().slice(0, 10);
  const last = kvGet('streak.lastDay');
  const count = parseInt(kvGet('streak.count') ?? '0', 10);

  if (last === today) return; // already counted today

  const yesterday = new Date(Date.now() - 86400000).toISOString().slice(0, 10);
  const newCount = last === yesterday ? count + 1 : 1;

  kvSet('streak.lastDay', today);
  kvSet('streak.count', String(newCount));
}

export function getStreak(): number {
  return parseInt(kvGet('streak.count') ?? '0', 10);
}

// --- Theme / prefs ---
export function getTheme(): 'dark' | 'light' {
  return (kvGet('prefs.theme') as 'dark' | 'light') ?? 'dark';
}

export function setTheme(theme: 'dark' | 'light'): void {
  kvSet('prefs.theme', theme);
}

export function getFontSize(): number {
  return parseInt(kvGet('prefs.fontSize') ?? '16', 10);
}

export function setFontSize(size: number): void {
  kvSet('prefs.fontSize', String(size));
}
