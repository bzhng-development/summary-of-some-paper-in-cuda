// Web app manifest — Next.js file convention. Makes the site installable
// as a PWA (Add to Home Screen on iOS / Install on Android).
//
// We deliberately omit icons[] for now since we don't have first-party
// art; iOS will fall back to a screenshot of the page. Color tokens match
// the Neon theme used in src/styles.
export default function manifest() {
  return {
    name: 'Paper Graph',
    short_name: 'Papers',
    description:
      'A reading-graph of paper summaries across LLMs, RL training, inference systems and architectures.',
    start_url: '/',
    scope: '/',
    display: 'standalone',
    orientation: 'portrait',
    background_color: '#0c0d0d',
    theme_color: '#0c0d0d',
    categories: ['education', 'productivity', 'reference'],
    lang: 'en-US',
  };
}
