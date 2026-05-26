# Open Issues / Blockers

## 1. Full paper markdown bodies not yet bundled as assets

**Severity:** Medium — app is usable, but paper detail screens show only the pitch/abstract (from graph.generated.json) rather than the full 7-section markdown body.

**Root cause:** The 1282 `.md` files live in `paper-graph-ui/src/content/papers/`. Metro's bundler cannot import arbitrary file-system paths outside the project root. To bundle the full text:

**Fix (one-time, ~2 min):**
```bash
cd mobile
# Copy all paper .md files into assets/papers/
rsync -a --include="*/" --include="*.md" --exclude="*" \
  ../paper-graph-ui/src/content/papers/ assets/papers/

# Then rebuild graph to pick up the new paths
node scripts/build-graph.mjs
```

After this, update `src/hooks/useMarkdownContent.ts` to load from the bundled asset path. Metro will bundle every `.md` under `assets/` because `assetBundlePatterns: ["**/*"]` is set in app.json.

## 2. iOS physical-device build requires interactive UDID setup

**Severity:** Low for overnight — simulator build is available.

**Fix:** Run `npx eas-cli@latest device:create` interactively, then rebuild with `--profile development`.

## 3. react-native-enriched-markdown requires a dev-client build

**Severity:** Expected — not a bug. The library uses Fabric native code and cannot run in Expo Go. The EAS dev-client builds in EAS_BUILD_LINKS.md are exactly the right artifact.

**Workaround:** Until the dev build is installed, the paper detail screen falls back to a plain-text renderer that parses headings/paragraphs manually (see `PlainTextFallback` in `src/app/p/[category]/[slug].tsx`). All paper metadata, actions, navigation, and topic/related sections still work.

## 4. ITSAppUsesNonExemptEncryption missing

**Severity:** Low — only matters when submitting to App Store.

**Fix:** Add to app.json:
```json
"ios": {
  "infoPlist": {
    "ITSAppUsesNonExemptEncryption": false
  }
}
```

## 5. runtimeVersion: appVersion requires expo-updates

**Severity:** Low — build succeeds, warning is cosmetic.

**Fix if needed:** Install `expo-updates` and configure OTA update URL, or switch `runtimeVersion` policy to `"sdkVersion"` in app.json.
