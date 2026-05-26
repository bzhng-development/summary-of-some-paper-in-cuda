/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ['./src/**/*.{ts,tsx}'],
  presets: [require('nativewind/preset')],
  theme: {
    extend: {
      colors: {
        brand: {
          purple: '#aa99ff',
          green: '#00E599',
          blue: '#259df4',
          yellow: '#f0f075',
          orange: '#ffa64c',
          pink: '#ff4c79',
          muted: '#94979E',
        },
        bg: {
          primary: '#0a0a0a',
          secondary: '#111111',
          card: '#1a1a1a',
          border: '#2a2a2a',
        },
        text: {
          primary: '#ffffff',
          secondary: '#a0a0a0',
          muted: '#666666',
        },
      },
      fontFamily: {
        sans: ['System'],
        mono: ['Courier'],
      },
    },
  },
  plugins: [],
};
