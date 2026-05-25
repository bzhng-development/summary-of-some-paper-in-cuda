import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';

const backend = process.env.PAPER_SERVER_URL ?? 'http://localhost:8787';

export default defineConfig({
	plugins: [sveltekit()],
	server: {
		port: 5173,
		proxy: {
			'/api/papers': backend,
			'/interested': backend,
			'/interested-ids': backend,
			'/add-paper': backend
		}
	}
});
