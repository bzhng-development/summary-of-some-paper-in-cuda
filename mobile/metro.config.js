const { getDefaultConfig } = require('expo/metro-config');
const { withNativeWind } = require('nativewind/metro');

const config = getDefaultConfig(__dirname);

// Bundle .md paper bodies as static assets so require() resolves them.
// Without this, Metro treats .md as a source file (parse error).
if (!config.resolver.assetExts.includes('md')) {
  config.resolver.assetExts.push('md');
}
// And make sure .md is NOT also in sourceExts (Metro errors otherwise).
config.resolver.sourceExts = config.resolver.sourceExts.filter((e) => e !== 'md');

module.exports = withNativeWind(config, { input: './src/global.css' });
