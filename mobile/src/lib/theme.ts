// Mobile theme tokens mirrored from paper-graph-ui/tailwind.config.js
// so the iOS app reads the same visual language as the web app.

export const colors = {
  // Backgrounds
  pageBg: '#0c0d0d', // black-new
  cardBg: '#131415', // gray-new-8 — slightly lighter than page
  cardBgLifted: '#18191B', // gray-new-10
  border: '#242628', // gray-new-15
  borderStrong: '#303236', // gray-new-20

  // Text
  white: '#ffffff',
  textBody: '#AFB1B6', // gray-new-70
  textMuted: '#797D86', // gray-new-50
  textSubtle: '#61646B', // gray-new-40
  textStrong: '#E4E5E7', // gray-new-90

  // Accents
  green: '#00E599', // primary-1 / green-45 — read / success
  yellow: '#F0F075',
  orange: '#ffa64c',
  pink: '#ff4c79',
  purple: '#aa99ff',
  blue: '#259df4',
  blueDeep: '#0055ff',
} as const;

export const radii = {
  sm: 8,
  md: 12,
  lg: 16, // rounded-2xl
  xl: 20,
  pill: 999,
} as const;

export const spacing = {
  xs: 4,
  sm: 8,
  md: 12,
  lg: 16,
  xl: 20,
  '2xl': 24,
  '3xl': 32,
} as const;

export const typography = {
  // tracking-tighter ≈ -0.025em — RN uses pixels, so scale per font size
  // titleTracking: -0.5 at 24px, -0.4 at 20px, etc.
  h1: { fontSize: 28, fontWeight: '700' as const, lineHeight: 34, letterSpacing: -0.6 },
  h2: { fontSize: 22, fontWeight: '700' as const, lineHeight: 28, letterSpacing: -0.4 },
  h3: { fontSize: 17, fontWeight: '600' as const, lineHeight: 22, letterSpacing: -0.2 },
  body: { fontSize: 15, fontWeight: '400' as const, lineHeight: 22 },
  bodySm: { fontSize: 13, fontWeight: '400' as const, lineHeight: 18 },
  caption: { fontSize: 12, fontWeight: '500' as const, lineHeight: 16 },
  micro: { fontSize: 11, fontWeight: '600' as const, lineHeight: 14, letterSpacing: 0.3 },
} as const;
