import { defineConfig, globalIgnores } from 'eslint/config';
import nextVitals from 'eslint-config-next/core-web-vitals';
import nextTs from 'eslint-config-next/typescript';
import oxlint from 'eslint-plugin-oxlint';

const oxlintDisable = await oxlint.buildFromOxlintConfigFile('.oxlintrc.json');

const eslintConfig = defineConfig([
  ...nextVitals,
  ...nextTs,
  globalIgnores([
    '.next/**',
    'out/**',
    'build/**',
    'next-env.d.ts',
    'src/components/**',
    'src/hooks/**',
    'src/utils/**',
    'src/contexts/**',
    'src/constants/**',
    'src/icons/**',
    'src/fonts/**',
    'src/styles/**',
    'src/generated/**',
    'src/content/**',
    'src/lib/shiki.js',
    'src/lib/rehype-code-props.js',
    'tailwind.config.js',
    'postcss.config.js',
    'empty.js',
  ]),
  ...oxlintDisable,
]);

export default eslintConfig;
