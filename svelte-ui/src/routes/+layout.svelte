<script lang="ts">
	let { children } = $props();
</script>

<div class="app">
	{@render children()}
</div>

<style>
	:global(*) {
		box-sizing: border-box;
		margin: 0;
		padding: 0;
	}
	:global(body) {
		font-family: 'Roboto', 'Google Sans', system-ui, -apple-system, sans-serif;
		background: #f2f2f7;
		color: #1d1d1f;
		min-height: 100vh;
		-webkit-font-smoothing: antialiased;
	}
	:global(h1, h2, h3) {
		font-family: 'Google Sans', 'Roboto', system-ui, sans-serif;
		font-weight: 500;
	}
	:global(.material-symbols-outlined) {
		font-family: 'Material Symbols Outlined';
		font-weight: normal;
		font-style: normal;
		font-size: 20px;
		line-height: 1;
		letter-spacing: normal;
		text-transform: none;
		display: inline-block;
		white-space: nowrap;
		direction: ltr;
		font-feature-settings: 'liga';
		-webkit-font-feature-settings: 'liga';
		-webkit-font-smoothing: antialiased;
	}
	.app {
		width: 100%;
		max-width: 1080px;
		margin: 0 auto;
		padding: 24px 20px 48px;
	}
</style>
