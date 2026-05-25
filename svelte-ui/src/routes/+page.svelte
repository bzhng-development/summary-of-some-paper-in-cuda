<script lang="ts">
	import { onMount } from 'svelte';
	import type { Paper } from '$lib/types';

	let papers = $state<Paper[]>([]);
	let interestedIds = $state<Set<string>>(new Set());
	let loading = $state(true);
	let loadError = $state('');

	// Filters
	let titleQ = $state('');
	let authorQ = $state('');
	let descQ = $state('');
	let minScore = $state(0);
	let tagFilter = $state('');
	// Min paper date in "YYYY-MM-DD" (matches <input type="date">). On load we
	// pin it to the 1st of the latest month in the dataset. If the paper has
	// a `published` field we compare full dates; otherwise the arxiv id only
	// encodes YYMM and we fall back to a month-level comparison.
	let minDate = $state('');

	// Selection + add-paper form
	let selected = $state<Set<string>>(new Set());
	let addUrl = $state('');
	let toastMsg = $state('');
	let toastShown = $state(false);
	let toastTimer: ReturnType<typeof setTimeout> | undefined;

	onMount(async () => {
		try {
			const [papersResp, interestedResp] = await Promise.all([
				fetch('/api/papers'),
				fetch('/interested-ids').catch(() => null)
			]);
			if (!papersResp.ok) throw new Error(`HTTP ${papersResp.status}`);
			papers = await papersResp.json();
			// Default the date filter to the 1st of the latest month present.
			let latestMonth = '';
			for (const p of papers) {
				const d = arxivDate(p.arxiv_id);
				if (d !== 'unknown' && d > latestMonth) latestMonth = d;
			}
			if (latestMonth) minDate = `${latestMonth}-01`;
			if (interestedResp?.ok) {
				const body = await interestedResp.json();
				// Server returns {"ids": [...]}; some older builds return a bare array.
				const ids: string[] = Array.isArray(body) ? body : Array.isArray(body?.ids) ? body.ids : [];
				interestedIds = new Set(ids);
			}
		} catch (err: unknown) {
			loadError = err instanceof Error ? err.message : String(err);
		} finally {
			loading = false;
		}
	});

	function arxivDate(id: string): string {
		const m = id.match(/^(\d{2})(\d{2})\./);
		return m ? `20${m[1]}-${m[2]}` : 'unknown';
	}

	function scoreBucket(s: number): 'high' | 'mid' | 'low' {
		if (s >= 8) return 'high';
		if (s >= 5) return 'mid';
		return 'low';
	}

	function truncateAuthors(p: Paper): string {
		const names = (p.authors ?? []).slice(0, 3);
		const parts = names.map((n) => {
			const affs = p.affiliations?.[n];
			return affs?.length ? `${n} (${affs.join(', ')})` : n;
		});
		const total = p.authors?.length ?? 0;
		if (total > 3) parts.push(`+${total - 3}`);
		return parts.join(', ');
	}

	// Derived: pre-normalized searchable strings per paper so filtering is cheap.
	// `pubDay` is the YYYY-MM-DD from arxiv-enriched `published`, when present.
	// `ymMonth` is always populated from the arxiv id (YYYY-MM).
	const index = $derived(
		papers.map((p) => ({
			p,
			titleLc: p.title.toLowerCase(),
			authorsLc: (p.authors ?? []).join(' ').toLowerCase(),
			reasonLc: (p.reason ?? '').toLowerCase(),
			tag: p.tag_category ?? '',
			ymMonth: arxivDate(p.arxiv_id),
			pubDay: p.published ? p.published.slice(0, 10) : ''
		}))
	);

	const allTags = $derived(
		Array.from(new Set(papers.map((p) => p.tag_category).filter(Boolean) as string[])).sort()
	);

	const filtered = $derived(() => {
		const q = titleQ.toLowerCase();
		const qa = authorQ.toLowerCase();
		const qd = descQ.toLowerCase();
		// minDate is YYYY-MM-DD. When the paper carries a real `published` field
		// we compare full dates; otherwise fall back to month-level (YYYY-MM)
		// comparison against the arxiv id, since that's all the data we have.
		const minMonth = minDate ? minDate.slice(0, 7) : '';
		return index
			.filter(
				(r) =>
					(!q || r.titleLc.includes(q)) &&
					(!qa || r.authorsLc.includes(qa)) &&
					(!qd || r.reasonLc.includes(qd)) &&
					r.p.score >= minScore &&
					(!tagFilter || r.tag === tagFilter) &&
					(!minDate ||
						(r.pubDay
							? r.pubDay >= minDate
							: r.ymMonth !== 'unknown' && r.ymMonth >= minMonth))
			)
			.map((r) => r.p);
	});

	const grouped = $derived(() => {
		const groups = new Map<string, Paper[]>();
		for (const p of filtered()) {
			const key = arxivDate(p.arxiv_id);
			const arr = groups.get(key);
			if (arr) arr.push(p);
			else groups.set(key, [p]);
		}
		// Within each group, highest score first, then HF upvotes as tiebreaker.
		for (const arr of groups.values())
			arr.sort((a, b) => b.score - a.score || (b.upvotes ?? 0) - (a.upvotes ?? 0));
		return Array.from(groups.entries()).sort((a, b) => b[0].localeCompare(a[0]));
	});

	function toggleSelect(id: string) {
		if (selected.has(id)) selected.delete(id);
		else selected.add(id);
		selected = new Set(selected);
	}

	function clearAll() {
		selected = new Set();
	}

	function showToast(msg: string) {
		toastMsg = msg;
		toastShown = true;
		clearTimeout(toastTimer);
		toastTimer = setTimeout(() => (toastShown = false), 1500);
	}

	async function markInterested() {
		const byId = new Map(papers.map((p) => [p.arxiv_id, p]));
		const picks = Array.from(selected)
			.filter((id) => !interestedIds.has(id))
			.map((id) => byId.get(id))
			.filter(Boolean) as Paper[];
		if (picks.length === 0) {
			showToast('Nothing to mark (all selected already interested)');
			return;
		}
		try {
			const resp = await fetch('/interested', {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({ papers: picks })
			});
			const data = await resp.json();
			const n = Array.isArray(data?.marked) ? data.marked.length : picks.length;
			for (const p of picks) interestedIds.add(p.arxiv_id);
			interestedIds = new Set(interestedIds);
			selected = new Set();
			showToast(`Marked ${n} as interested`);
		} catch (err: unknown) {
			showToast(`Error: ${err instanceof Error ? err.message : String(err)}`);
		}
	}

	function copySelected() {
		if (selected.size === 0) {
			showToast('Nothing selected');
			return;
		}
		const byId = new Map(papers.map((p) => [p.arxiv_id, p]));
		const lines = Array.from(selected)
			.map((id) => byId.get(id))
			.filter(Boolean)
			.map((p) => `${p!.title} (https://arxiv.org/abs/${p!.arxiv_id})`);
		navigator.clipboard.writeText(lines.join('\n')).then(() => {
			showToast(`Copied ${lines.length} papers`);
		});
	}

	async function addPaper() {
		const url = addUrl.trim();
		if (!url) {
			showToast('Paste an arxiv/HF URL or ID');
			return;
		}
		try {
			const resp = await fetch('/add-paper', {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({ url })
			});
			const data = await resp.json();
			if (data.error) {
				showToast(`Error: ${data.error}`);
				return;
			}
			if (data.status === 'already_exists') {
				showToast(`${data.arxiv_id} already in DB`);
				return;
			}
			showToast(`Added ${data.arxiv_id}${data.has_meta ? ' (with metadata)' : ''}`);
			addUrl = '';
		} catch (err: unknown) {
			showToast(`Error: ${err instanceof Error ? err.message : String(err)}`);
		}
	}
</script>

<svelte:head>
	<title>Paper Viewer ({papers.length.toLocaleString()} papers)</title>
</svelte:head>

<header class="page-header">
	<span class="material-symbols-outlined header-icon">menu_book</span>
	<div>
		<h1>Paper Viewer</h1>
		<p class="subtitle">
			{papers.length.toLocaleString()} scored papers · HF daily papers × arXiv × your reading history
		</p>
	</div>
</header>

<div class="toolbar">
	<div class="tb-group">
		<button class="btn btn-primary" onclick={markInterested}>
			<span class="material-symbols-outlined">bookmark_add</span>
			<span>Mark Interested</span>
		</button>
		<button class="btn" onclick={copySelected}>
			<span class="material-symbols-outlined">content_copy</span>
			<span>Copy</span>
		</button>
		<button class="btn btn-ghost" onclick={clearAll}>
			<span class="material-symbols-outlined">clear</span>
			<span>Clear</span>
		</button>
		<span class="count">{selected.size} selected</span>
	</div>

	<div class="tb-group search-group">
		<label class="field">
			<span class="material-symbols-outlined field-icon">search</span>
			<input type="text" bind:value={titleQ} placeholder="Title" />
		</label>
		<label class="field">
			<span class="material-symbols-outlined field-icon">person</span>
			<input type="text" bind:value={authorQ} placeholder="Author" />
		</label>
		<label class="field">
			<span class="material-symbols-outlined field-icon">notes</span>
			<input type="text" bind:value={descQ} placeholder="Description" />
		</label>
	</div>

	<div class="tb-group">
		<label class="field wide">
			<span class="material-symbols-outlined field-icon">link</span>
			<input type="text" bind:value={addUrl} placeholder="arxiv / HF URL or ID" />
		</label>
		<button class="btn" onclick={addPaper}>
			<span class="material-symbols-outlined">add</span>
			<span>Add to DB</span>
		</button>
	</div>

	<div class="tb-group">
		<div class="filter-row">
			<span class="material-symbols-outlined filter-icon">stars</span>
			<label for="min-score">Min score</label>
			<select id="min-score" bind:value={minScore}>
				<option value={0}>All</option>
				<option value={3}>3+</option>
				<option value={5}>5+</option>
				<option value={6}>6+</option>
				<option value={7}>7+</option>
				<option value={8}>8+</option>
			</select>
		</div>
		<div class="filter-row">
			<span class="material-symbols-outlined filter-icon">event</span>
			<label for="min-date">From</label>
			<input id="min-date" type="date" bind:value={minDate} />
			{#if minDate}
				<button
					type="button"
					class="chip-clear"
					aria-label="Clear date filter"
					onclick={() => (minDate = '')}
				>
					<span class="material-symbols-outlined">close</span>
				</button>
			{/if}
		</div>
		<div class="filter-row">
			<span class="material-symbols-outlined filter-icon">category</span>
			<label for="tag-filter">Category</label>
			<select id="tag-filter" bind:value={tagFilter}>
				<option value="">All</option>
				{#each allTags as t (t)}
					<option value={t}>{t}</option>
				{/each}
			</select>
		</div>
	</div>
</div>

{#if loading}
	<p class="status">
		<span class="material-symbols-outlined spin">progress_activity</span>
		Loading…
	</p>
{:else if loadError}
	<p class="status error">
		<span class="material-symbols-outlined">error</span>
		Failed to load papers: {loadError}
	</p>
{:else}
	<p class="status">
		Showing {filtered().length.toLocaleString()} / {papers.length.toLocaleString()} papers
	</p>
	{#each grouped() as [dateKey, group] (dateKey)}
		<section class="date-group">
			<h2>
				<span class="material-symbols-outlined">calendar_month</span>
				{dateKey}
				<span class="group-count">{group.length} papers</span>
			</h2>
			{#each group as p (p.arxiv_id)}
				{@const isSel = selected.has(p.arxiv_id)}
				{@const isInt = interestedIds.has(p.arxiv_id)}
				<div
					class="paper {scoreBucket(p.score)}"
					class:selected={isSel}
					class:interested={isInt}
					role="button"
					tabindex="0"
					onclick={() => toggleSelect(p.arxiv_id)}
					onkeydown={(e) => {
						if (e.key === 'Enter' || e.key === ' ') {
							e.preventDefault();
							toggleSelect(p.arxiv_id);
						}
					}}
				>
					<div class="paper-header">
						<span class="score">{p.score}/10</span>
						{#if p.tag_category}<span class="tag-cat">{p.tag_category}</span>{/if}
						<span class="title">{p.title}</span>
						<span class="upvotes">{p.upvotes ?? 0}↑</span>
						{#if isInt}<span class="interested-badge">interested</span>{/if}
					</div>
					<div class="paper-meta">
						<span class="authors">{truncateAuthors(p)}</span>
						{#if p.org_fullname || p.organization}
							<span class="org">{p.org_fullname ?? p.organization}</span>
						{/if}
						{#if p.categories?.length}<span class="cats">{p.categories.join(', ')}</span>{/if}
						{#if p.journal_ref}<span class="journal">{p.journal_ref}</span>{/if}
						<a
							href="https://arxiv.org/abs/{p.arxiv_id}"
							target="_blank"
							rel="noreferrer"
							class="arxiv-link"
							onclick={(e) => e.stopPropagation()}>arXiv</a
						>
						{#if p.github}
							<a
								href={p.github}
								target="_blank"
								rel="noreferrer"
								class="gh-link"
								onclick={(e) => e.stopPropagation()}
								>code({p.github_stars ?? 0}⭐)</a
							>
						{/if}
					</div>
					{#if p.arxiv_comment}<div class="comment">{p.arxiv_comment}</div>{/if}
					{#if p.reason}<div class="paper-reason">{p.reason}</div>{/if}
					{#if p.similar_paper && p.similar_paper !== 'NONE'}
						<div class="similar">Similar: {p.similar_paper}</div>
					{/if}
				</div>
			{/each}
		</section>
	{/each}
{/if}

<div class="toast" class:show={toastShown}>{toastMsg}</div>

<style>
	/* Google Material-ish palette */
	.page-header {
		display: flex;
		align-items: center;
		gap: 16px;
		padding-bottom: 20px;
		margin-bottom: 16px;
		border-bottom: 1px solid #e5e5ea;
	}
	.header-icon {
		font-size: 40px !important;
		background: #e8f0fe;
		color: #1a73e8;
		border-radius: 50%;
		padding: 12px;
	}
	.page-header h1 {
		font-size: 1.75rem;
		font-weight: 500;
		color: #202124;
		letter-spacing: -0.2px;
	}
	.subtitle {
		color: #5f6368;
		font-size: 0.9rem;
		margin-top: 2px;
	}

	.toolbar {
		position: sticky;
		top: 0;
		z-index: 100;
		background: rgba(242, 242, 247, 0.92);
		backdrop-filter: saturate(180%) blur(12px);
		-webkit-backdrop-filter: saturate(180%) blur(12px);
		padding: 12px 0;
		border-bottom: 1px solid #e5e5ea;
		display: flex;
		gap: 12px;
		align-items: center;
		flex-wrap: wrap;
		margin-bottom: 16px;
	}
	.tb-group {
		display: flex;
		gap: 8px;
		align-items: center;
		flex-wrap: wrap;
	}
	.search-group {
		flex: 1;
		min-width: 240px;
	}
	.btn {
		display: inline-flex;
		align-items: center;
		gap: 6px;
		padding: 8px 16px;
		border: 1px solid #dadce0;
		border-radius: 20px;
		background: #fff;
		color: #202124;
		cursor: pointer;
		font-family: 'Google Sans', 'Roboto', sans-serif;
		font-size: 0.82rem;
		font-weight: 500;
		transition:
			background 0.15s,
			box-shadow 0.15s,
			border-color 0.15s;
	}
	.btn :global(.material-symbols-outlined) {
		font-size: 18px;
	}
	.btn:hover {
		background: #f8f9fa;
		box-shadow: 0 1px 2px rgba(60, 64, 67, 0.15);
	}
	.btn-primary {
		background: #1a73e8;
		color: #fff;
		border-color: #1a73e8;
	}
	.btn-primary:hover {
		background: #1765cc;
		border-color: #1765cc;
	}
	.btn-ghost {
		background: transparent;
		border-color: transparent;
		color: #5f6368;
	}
	.btn-ghost:hover {
		background: #f1f3f4;
	}
	.count {
		font-size: 0.8rem;
		color: #5f6368;
		padding: 0 4px;
	}

	.field {
		display: inline-flex;
		align-items: center;
		gap: 6px;
		padding: 4px 10px 4px 8px;
		border: 1px solid #dadce0;
		border-radius: 20px;
		background: #fff;
		transition: border-color 0.15s, box-shadow 0.15s;
	}
	.field:focus-within {
		border-color: #1a73e8;
		box-shadow: 0 0 0 1px #1a73e8;
	}
	.field.wide {
		min-width: 260px;
	}
	.field-icon {
		color: #5f6368;
		font-size: 18px !important;
	}
	.field input {
		border: none;
		outline: none;
		background: transparent;
		font-size: 0.85rem;
		color: #202124;
		font-family: inherit;
		padding: 4px 2px;
		width: 120px;
	}
	.field.wide input {
		width: 100%;
	}

	.filter-row {
		display: inline-flex;
		align-items: center;
		gap: 6px;
		padding: 4px 10px;
		border: 1px solid #dadce0;
		border-radius: 20px;
		background: #fff;
	}
	.filter-icon {
		color: #5f6368;
		font-size: 18px !important;
	}
	.filter-row label {
		font-size: 0.8rem;
		color: #5f6368;
	}
	.filter-row select,
	.filter-row input[type='date'] {
		border: none;
		outline: none;
		background: transparent;
		font-size: 0.82rem;
		color: #202124;
		font-family: inherit;
		padding: 0;
	}
	.filter-row input[type='date'] {
		color-scheme: light;
		min-width: 140px;
	}
	.chip-clear {
		display: inline-flex;
		align-items: center;
		justify-content: center;
		border: none;
		background: transparent;
		color: #5f6368;
		cursor: pointer;
		padding: 0;
		border-radius: 999px;
	}
	.chip-clear :global(.material-symbols-outlined) {
		font-size: 16px;
	}
	.chip-clear:hover {
		color: #202124;
		background: #f1f3f4;
	}

	.status {
		display: inline-flex;
		align-items: center;
		gap: 8px;
		font-size: 0.85rem;
		color: #5f6368;
		margin: 8px 0 16px;
	}
	.status.error {
		color: #d93025;
	}
	.spin {
		animation: spin 1s linear infinite;
	}
	@keyframes spin {
		to {
			transform: rotate(360deg);
		}
	}

	h2 {
		display: flex;
		align-items: center;
		gap: 8px;
		font-size: 0.95rem;
		font-weight: 500;
		color: #5f6368;
		text-transform: uppercase;
		letter-spacing: 0.5px;
		margin: 28px 0 12px;
		padding-bottom: 6px;
		border-bottom: 1px solid #e5e5ea;
	}
	h2 :global(.material-symbols-outlined) {
		font-size: 18px;
		color: #1a73e8;
	}
	.group-count {
		color: #9aa0a6;
		font-size: 0.78rem;
		font-weight: 400;
		text-transform: none;
		letter-spacing: 0;
		margin-left: auto;
	}

	.paper {
		padding: 14px 16px;
		margin: 8px 0;
		border-radius: 12px;
		border: 1px solid #e5e5ea;
		background: #fff;
		cursor: pointer;
		transition:
			box-shadow 0.15s,
			border-color 0.15s,
			transform 0.08s;
		content-visibility: auto;
		contain-intrinsic-size: 0 110px;
	}
	.paper:hover {
		box-shadow:
			0 1px 3px rgba(60, 64, 67, 0.12),
			0 1px 2px rgba(60, 64, 67, 0.24);
		border-color: #dadce0;
	}
	.paper:active {
		transform: scale(0.998);
	}
	.paper.selected {
		border-color: #1a73e8;
		background: #e8f0fe;
		box-shadow: 0 0 0 1px #1a73e8 inset;
	}
	.paper.interested {
		opacity: 0.55;
	}
	.paper.interested:hover {
		opacity: 0.9;
	}

	.paper-header {
		display: flex;
		align-items: center;
		gap: 10px;
		flex-wrap: wrap;
	}
	.score {
		font-family: 'Google Sans', 'Roboto', sans-serif;
		font-weight: 600;
		font-size: 0.78rem;
		min-width: 3rem;
		text-align: center;
		padding: 3px 8px;
		border-radius: 999px;
	}
	.paper.high .score {
		background: #e6f4ea;
		color: #137333;
	}
	.paper.mid .score {
		background: #fef7e0;
		color: #b06000;
	}
	.paper.low .score {
		background: #f1f3f4;
		color: #5f6368;
	}
	.title {
		font-family: 'Google Sans', 'Roboto', sans-serif;
		font-weight: 500;
		font-size: 0.98rem;
		color: #202124;
		flex: 1;
		min-width: 0;
	}
	.upvotes {
		font-size: 0.78rem;
		color: #9aa0a6;
		white-space: nowrap;
	}

	.paper-meta {
		font-size: 0.78rem;
		color: #5f6368;
		margin-top: 6px;
		display: flex;
		gap: 8px;
		align-items: center;
		flex-wrap: wrap;
	}
	.paper-meta a {
		color: #1a73e8;
		text-decoration: none;
		font-weight: 500;
	}
	.paper-meta a:hover {
		text-decoration: underline;
	}
	.authors {
		color: #5f6368;
	}
	.org {
		background: #f3e8fd;
		color: #5f259f;
		padding: 2px 8px;
		border-radius: 999px;
		font-size: 0.72rem;
		font-weight: 500;
		white-space: nowrap;
	}
	.cats {
		color: #9aa0a6;
		font-size: 0.72rem;
	}
	.tag-cat {
		background: #e8f0fe;
		color: #1a73e8;
		padding: 2px 8px;
		border-radius: 999px;
		font-size: 0.7rem;
		font-weight: 600;
		white-space: nowrap;
	}
	.journal {
		background: #e6f4ea;
		color: #137333;
		padding: 2px 8px;
		border-radius: 999px;
		font-size: 0.72rem;
		font-weight: 500;
		white-space: nowrap;
	}
	.comment {
		font-size: 0.78rem;
		color: #b06000;
		margin-top: 6px;
		line-height: 1.4;
	}
	.paper-reason {
		font-size: 0.82rem;
		color: #3c4043;
		margin-top: 6px;
		line-height: 1.45;
	}
	.similar {
		font-size: 0.76rem;
		color: #9aa0a6;
		font-style: italic;
		margin-top: 4px;
	}
	.interested-badge {
		display: inline-flex;
		align-items: center;
		background: #1a73e8;
		color: #fff;
		padding: 2px 8px;
		border-radius: 999px;
		font-size: 0.68rem;
		font-weight: 600;
		white-space: nowrap;
	}

	.toast {
		position: fixed;
		bottom: 24px;
		left: 50%;
		transform: translateX(-50%) translateY(8px);
		background: #202124;
		color: #fff;
		padding: 12px 20px;
		border-radius: 8px;
		font-size: 0.85rem;
		opacity: 0;
		transition:
			opacity 0.24s,
			transform 0.24s;
		pointer-events: none;
		box-shadow: 0 4px 12px rgba(0, 0, 0, 0.18);
	}
	.toast.show {
		opacity: 1;
		transform: translateX(-50%) translateY(0);
	}
</style>
