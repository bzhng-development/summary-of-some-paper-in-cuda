# Open Issues / Blockers

## Resolved (kept for history)

- **Full paper markdown bodies bundled + rendered.** 1320 `.md` files are copied into `assets/papers/` by `scripts/build-graph.mjs`, `src/lib/paper-asset-map.ts` emits `require()` per paper, and `src/hooks/useMarkdownContent.ts` resolves via `Asset.fromModule` → `expo-file-system` `File`. The detail screen (`src/app/p/[category]/[slug].tsx`) calls the hook and renders via `EnrichedMarkdownText` from `react-native-enriched-markdown`.
- **iOS device build flow works locally.** Personal Team `937557U4CK`, paired iPhone (UDID `00008150-00124866140A401C`), trusted developer profile. Release config builds via direct `xcodebuild -allowProvisioningUpdates ...` (see note below).
- **`runtimeVersion` switched to `sdkVersion` policy** so the missing-`expo-updates` warning is gone.
- **`ITSAppUsesNonExemptEncryption: false`** added to `ios.infoPlist` — App Store submission will not block on it.

## Active

### 1. `expo run:ios --device` doesn't pass `-allowProvisioningUpdates`

**Severity:** Low — a documented Expo CLI papercut.

Expo's CLI doesn't pass `-allowProvisioningUpdates` to `xcodebuild`, so first-time provisioning (no profile cached) fails. Workarounds:

- Run `xcodebuild` directly with the flag (the current path), or
- Build once via the Xcode GUI Run button (passes the flag implicitly), then subsequent `npx expo run:ios --device` calls succeed because the profile is cached.

### 2. `DEVELOPMENT_TEAM` is hardcoded in pbxproj

**Severity:** Low — survives all normal builds.

`DEVELOPMENT_TEAM = 937557U4CK` lives in `ios/PaperGraph.xcodeproj/project.pbxproj`. It is **lost on `npx expo prebuild --clean`**. Either:

- Re-edit after every `prebuild --clean`, or
- Add a config plugin that injects it, or
- Configure signing through EAS Build (`eas.json` ios profile + `eas credentials`).

### 3. 254 MB app bundle

**Severity:** Low for sideload / TestFlight; medium for App Store.

The Release `.app` weighs 254 MB because all 1320 paper `.md` bodies (~185 MB) are baked in. App Store cellular-download cap is 500 MB. If paper count grows materially:

- Lazy-fetch markdown from a CDN at runtime, or
- gzip-compress the bundled `.md` files pre-bundle (~3-4× shrink), or
- Split into `expo-updates` asset bundles.

### 4. `iosMath-mathFonts` pod deployment target = 6.0

**Severity:** Cosmetic.

Transitive pod `iosMath (0.9.4)` declares `IPHONEOS_DEPLOYMENT_TARGET = 6.0`, below the supported range 12.0–26.2.99. Xcode logs a warning per build. No runtime impact; bump (or replace the dependency) only if the warning bothers you.

### 5. Several build-phase scripts marked "Based on dependency analysis = unchecked"

**Severity:** Cosmetic build-perf hit.

Expo's own build phases run on every incremental build because they don't declare outputs:

- `[Expo] Configure project`
- `[Expo Dev Launcher] Strip Local Network Keys for Release`
- `[CP-User] [Hermes] Replace Hermes for the right configuration, if needed`
- `[CP-User] [Expo] Switch * XCFramework for build configuration`
- `Bundle React Native code and images`

These are Expo's intentional configuration; changing them risks breaking incremental builds. Tolerate until Expo fixes upstream.
