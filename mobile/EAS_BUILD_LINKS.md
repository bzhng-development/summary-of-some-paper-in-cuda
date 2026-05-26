# EAS Build Links — Paper Graph Mobile

## Build #3 — 2026-05-26 02:24 (1371 papers, USE THESE)

Latest. Adds the 80 curated April-May papers from the T7-T10 chain on top
of Build #2's 1306. Also includes the T2 re-categorization (852 papers
moved between dirs). 257 MB archive.

- **Android (APK):** https://expo.dev/accounts/vincentzhongy/projects/paper-graph-mobile/builds/66b18273-f720-47a7-8cbe-ea82ac9a3832
- **iOS Simulator (.app):** https://expo.dev/accounts/vincentzhongy/projects/paper-graph-mobile/builds/3642eef3-a071-4c0e-8a9a-d16539b11ebe

## Build #2 — 2026-05-26 00:14 (content-bundled, USE THESE)

These are the actual usable builds. The build-graph.mjs now rsyncs all 1297
.md files into assets/papers/ and emits src/lib/paper-asset-map.ts so Metro
bundles paper bodies as assets. Archive size jumped from ~10MB to 243MB
(bodies), and useMarkdownContent now resolves via Asset.fromModule().

- **Android (APK, profile=development-simulator):** https://expo.dev/accounts/vincentzhongy/projects/paper-graph-mobile/builds/ad5996f2-12b2-4c1d-b017-fb24380af8cb
- **iOS Simulator (.app, profile=development-simulator):** https://expo.dev/accounts/vincentzhongy/projects/paper-graph-mobile/builds/4f20d90c-3599-48be-b6aa-8507a38d7716

## Build #1 — 2026-05-26 (subagent first build — metadata only, no bodies)

Kept for reference. Paper detail screens render only the pitch — the asset
bundling wasn't wired yet. Superseded by Build #2.

- **Android (APK, profile=development):** https://expo.dev/accounts/vincentzhongy/projects/paper-graph-mobile/builds/4bcf708f-eb1c-415f-a0c8-da4b803a63ff
- **iOS Simulator (.app, profile=development-simulator):** https://expo.dev/accounts/vincentzhongy/projects/paper-graph-mobile/builds/f7241ad5-fcd3-4c00-a041-b8737b9287b5

## Notes

- iOS physical-device build (ad-hoc profile) was blocked non-interactively because EAS needs
  a provisioning profile with device UDIDs. To build for a real iPhone:
    1. Register your device UDID: `npx eas-cli@latest device:create`
    2. Rebuild: `npx eas-cli@latest build --platform ios --profile development --no-wait`
- The `development-simulator` profile produces a `.app` bundle for the Xcode Simulator —
  works without any Apple Developer account setup.

## Re-running builds
```bash
cd mobile
# Android physical device (APK)
npx eas-cli@latest build --platform android --profile development --no-wait

# iOS simulator
npx eas-cli@latest build --platform ios --profile development-simulator --no-wait

# iOS physical device (requires device UDID registered first)
npx eas-cli@latest device:create
npx eas-cli@latest build --platform ios --profile development --no-wait
```
