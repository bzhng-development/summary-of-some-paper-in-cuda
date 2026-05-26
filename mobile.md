Modern Expo Content-Library App Workflow (May 2026)

  This is the verbose corrected workflow for spinning up a Markdown-reader/content-library app on Expo SDK 56. Every step explains the why, not just the command.

  ---
  1. Project initialization

  npx create-expo-app@latest --template default@sdk-56 my-content-library
  cd my-content-library

  Why --template default@sdk-56 and not --template blank-typescript?

  Two reasons:

  (a) SDK pinning. During the SDK 56 transition window, npx create-expo-app@latest with no template flag still generates an SDK 54 project. This is intentional — Expo Go on physical
  devices currently ships SDK 54, so the bare command keeps Expo Go users from getting a broken app. If you want the actual 2026 release (RN 0.85, React 19.2, iOS 16.4 minimum, all the
   new build perf wins), you have to ask for it explicitly with default@sdk-56.

  (b) What "default" actually means. The default template is no longer a stripped-down hello-world. As of SDK 56 it ships with:

  - TypeScript fully configured (no separate "blank-typescript" needed — the old template is now redundant for most apps)
  - expo-router pre-installed with config plugin wired up, plus a working tabs layout under src/app/(tabs)/
  - Expo UI (expo-ui) — native UI primitives, stable in SDK 56, also available in Expo Go
  - AGENTS.md, CLAUDE.md, and .claude/settings.json — pre-populated with Expo-specific guidance so Claude Code / coding agents have the right context without you writing it
  - A src/ source root with src/app/, src/components/, etc. — the canonical layout

  The old blank-typescript template is the bare-minimum scaffold and exists mostly for people who want to build their navigation stack manually. For a Markdown-reader you'd just be
  reinstalling everything default already gives you.

  ---
  2. What's already installed (skip these — they're in the template)

  These come pre-wired in default@sdk-56, so you don't need to expo install them:

  - expo-router + react-native-safe-area-context + react-native-screens
  - react-native-reanimated + react-native-gesture-handler
  - expo-ui (native components)
  - TypeScript, ESLint, Prettier

  Running npx expo install <thing-already-there> is harmless but it's noise — it makes your setup script look like you don't know what's in the template.

  ---
  3. Picking a navigation system: expo-router only

  In SDK 56 this stopped being a choice. The framing "expo-router OR React Navigation" is outdated.

  What changed in SDK 56:

  - Expo Router now forks the parts of React Navigation it depends on. The expo-router package no longer declares @react-navigation/* as a dependency.
  - Importing from @react-navigation/native, @react-navigation/native-stack, @react-navigation/drawer, etc. in your own app code is no longer supported. You must import from
  expo-router entry points. There's a codemod and a migration guide for SDK 55→56.
  - Third-party libraries that still pull from @react-navigation/core are handled by a compatibility shim in Expo CLI that rewrites those imports inside node_modules. So most libraries
   keep working, but if you hit a weird navigation bug, suspect a dep that's still on the old packages.

  Practical implication: delete the line that said npx expo install @react-navigation/native  # if you prefer not using Expo Router — it's a footgun in SDK 56. Use expo-router's
  <Stack>, <Tabs>, and <Drawer> components instead.

  Tab implementation: SDK 56's default template uses a split — src/components/app-tabs.native.tsx renders native tabs (platform-native tab bar) on iOS/Android, and
  src/components/app-tabs.tsx renders custom tabs from expo-router/ui on web. Expo's bundler resolves the .native.tsx vs .tsx extension automatically. Keep this pattern — don't replace
   it with a single JS tabs implementation unless you have a reason.

  ---
  4. Markdown rendering: use react-native-enriched-markdown

  pnpm add react-native-enriched-markdown

  Why not react-native-markdown-display?

  The upstream repo (iamacup/react-native-markdown-display) officially recommends migration. Their README now points users at react-native-enriched-markdown as the path forward. The
  original library:

  - Uses Paper (the legacy architecture); not Fabric-native
  - Hasn't kept pace with RN 0.82+ which runs entirely on the New Architecture
  - Has a graveyard of forks (jonasmerlin, RonRadtke, willmac321, jeremybarbet…) — each fixed one specific incompatibility, none are canonically maintained

  Why react-native-enriched-markdown?

  - Built by Software Mansion (the company behind Reanimated, Gesture Handler, Screens — i.e. the people who maintain a huge chunk of the RN ecosystem). Maintenance signal is strong.
  - Fabric-native (New Architecture), which is mandatory in SDK 55+ anyway
  - Uses md4c for parsing — fast, CommonMark + GFM compliant
  - Supports iOS, Android, macOS, and web
  - Provides both a renderer and a Markdown-output rich-text input (useful if your reader ever grows note-taking features)
  - Has native text selection, accessibility hooks, RTL support — all the things you'd otherwise wire up yourself

  Why not @mattermost/react-native-markdown?

  It isn't a maintained npm package you can install. Mattermost's markdown code lives inside their mobile app repo as an internal module. The name shows up in old "RN markdown options"
   posts but you can't npm install it.

  ---
  5. Storage: pick one SQLite library

  # Option A — recommended for this app
  npx expo install expo-sqlite

  # Option B — only if perf-critical
  pnpm add @op-engineering/op-sqlite

  Why not both, and why not sqlite3

  - sqlite3 (no namespace) is a Node.js package for backends. It will install but never work in a React Native app. Drop it entirely from your list.
  - Installing both expo-sqlite and @op-engineering/op-sqlite is a known footgun: they each try to compile/link SQLite from source. Even when both compile, the flags can differ and you
   get runtime errors. Pick one.

  expo-sqlite (recommended for a content library)

  Why this is the right default for a reader app:

  - DevTools inspector built in. Press Shift+M in the Expo CLI terminal, select "Open expo-sqlite" — you get a browser-based table viewer, row editor, query runner, and DB export. No
  third-party tool. For a content library this is huge: you'll be debugging "is this paper in the DB?" constantly.
  - Includes expo-sqlite/kv-store as a drop-in replacement for AsyncStorage, backed by SQLite. Saves you a dependency for "remember last-read position" / preferences.
  - Works on web, with a metro.config.js tweak for .wasm files and SharedArrayBuffer headers. Run npx expo customize metro.config.js if you don't have one.
  - Frictionless setup in managed workflow — no native linking
  - For a reader app, your write loads are tiny (mark-as-read, bookmark, last-opened). The 5–8× perf advantage of op-sqlite is invisible at this scale.

  op-sqlite (only if you have a perf reason)

  Choose this if:
  - You're doing full-text search across thousands of papers with FTS5 and feeling latency
  - You're running on-device LLM embeddings or similarity search over the library
  - You're batch-inserting millions of rows on first launch

  Tradeoffs:
  - 5–8× faster reads/writes via JSI (direct sync calls, no bridge serialization)
  - 1/4 memory consumption on big queries (1.2 GB → 250 MB in the upstream benchmark)
  - Memory mapping option for even faster I/O (skip kernel)
  - Has op-sqlcipher fork if you need transparent AES-256 encryption
  - But: conflicts with expo-updates and other Expo packages that link SQLite. Fix is to add "expo.updates.useThirdPartySQLitePod": "true" to ios/Podfile.properties.json so
  expo-updates uses the third-party pod instead of its own.
  - Web support is async-only (no executeSync()); SQLCipher/libsql/Turso unsupported on web.

  For a Markdown reader: start with expo-sqlite. Migrate later only if you actually hit perf limits.

  ---
  6. Search: fuse.js for fuzzy, expo-sqlite FTS5 for full-text

  pnpm add fuse.js

  Not npx expo install fuse.js — Fuse is pure JavaScript, no native code. expo install exists to version-pin libraries against the active SDK's native compatibility matrix; for pure-JS
   deps it's just a slower pnpm add.

  When to use which:
  - Fuse.js: in-memory fuzzy search across <10k items. Good for "find a paper by title approximately". Loads everything into RAM, builds an index per session.
  - expo-sqlite + FTS5: full-text search across paper bodies/summaries. SQLite has a built-in FTS5 virtual table; you create one alongside your main table and query with MATCH.
  Persistent, doesn't need to rebuild on launch, scales to millions of rows. Slightly more setup.

  Use both if needed — they solve different problems.

  ---
  7. File handling

  npx expo install expo-file-system expo-document-picker expo-sharing

  These three are correct as-is in your original. Quick notes:

  - expo-file-system ships a new next-gen API (expo-file-system/next) with a synchronous File/Directory class-based interface. Prefer it over the legacy API for new code — much nicer
  than the old FileSystem.readAsStringAsync pattern.
  - expo-document-picker for importing .md files from Files / Drive
  - expo-sharing for exporting / share-sheet

  ---
  8. Styling: pick one, don't layer them

  # Option A — Tailwind for RN
  npx expo install nativewind
  pnpm add -D tailwindcss

  NativeWind v4 is the current generation. Setup requires tailwind.config.js plus a Babel/Metro plugin entry — follow their docs. Works with the New Architecture.

  Alternatives (don't combine):
  - Tamagui — heavier, more opinionated, better at design systems and animations, more complex setup
  - Gluestack v2 — component library with utility-first styling; good if you want pre-built accessible primitives

  For a reader app, NativeWind is the lowest-friction choice — you mostly want typography styling, not a component kit.

  ---
  9. Project structure

  my-content-library/
  ├── src/
  │   ├── app/                        # expo-router routes ONLY
  │   │   ├── _layout.tsx             # root layout
  │   │   ├── (tabs)/
  │   │   │   ├── _layout.tsx
  │   │   │   ├── library.tsx         # /library
  │   │   │   ├── search.tsx          # /search
  │   │   │   └── settings.tsx
  │   │   └── reader/
  │   │       └── [id].tsx            # /reader/<paper-id>
  │   ├── components/
  │   │   ├── app-tabs.native.tsx     # iOS/Android native tabs
  │   │   └── app-tabs.tsx            # Web custom tabs
  │   ├── hooks/
  │   │   ├── useMarkdownLoader.ts
  │   │   └── useSearch.ts
  │   ├── lib/
  │   │   ├── db.ts                   # expo-sqlite setup + schema
  │   │   ├── markdown.ts             # parser helpers
  │   │   └── search.ts               # Fuse + FTS5 dispatch
  │   └── types/
  ├── assets/
  ├── app.json
  ├── AGENTS.md                       # auto-generated by template
  ├── CLAUDE.md                       # auto-generated by template
  ├── .claude/settings.json           # auto-generated
  ├── package.json
  └── tsconfig.json

  Key rule (commonly violated)

  src/app/ is exclusively for route files. Don't put components, hooks, or utilities inside it. Expo Router treats every file in src/app/ as a potential route and will get confused.
  Anything non-route goes in src/components/, src/hooks/, src/lib/, etc.

  Special files

  - _layout.tsx — wraps everything below it. Root _layout.tsx is where you put font loading, theme providers, DB initialization. Replaces the old App.tsx entry point.
  - (tabs)/ — parentheses mean route group: the segment doesn't appear in the URL. Used to scope a layout to a subset of routes.
  - [id].tsx — dynamic segment. Available as useLocalSearchParams<{ id: string }>() inside the component.
  - index.tsx — the default route at a level. src/app/index.tsx is /.

  ---
  10. app.json — the parts that matter in SDK 56

  {
    "expo": {
      "name": "My Content Library",
      "slug": "content-library",
      "scheme": "contentlib",

      "runtimeVersion": {
        "policy": "appVersion"
      },

      "assetBundlePatterns": ["**/*"],

      "ios": {
        "supportsTablet": true,
        "bundleIdentifier": "com.you.contentlib"
      },

      "android": {
        "package": "com.you.contentlib",
        "adaptiveIcon": { "foregroundImage": "./assets/adaptive-icon.png", "backgroundColor": "#ffffff" }
      },

      "plugins": [
        "expo-router",
        [
          "expo-build-properties",
          {
            "ios": { "deploymentTarget": "16.4" },
            "android": { "minSdkVersion": 26 }
          }
        ]
      ]
    }
  }

  Things to leave out

  - "newArchEnabled": true — no-op in SDK 55+. The New Architecture is always on and can't be disabled. SDK 54 was the last version where you could turn it off. Setting it does
  nothing; remove it.
  - Top-level "ios": { "minimumOSVersion": "16.4" } — Expo SDK 56 already enforces iOS 16.4 as the floor. You only need to override if you want to raise it further. If you want to be
  explicit, set deploymentTarget via expo-build-properties (shown above), which is the supported channel.
  - "minSdkVersion": 28 at the top of the android block — Expo's default minSdk varies by SDK release; pinning via expo-build-properties is the supported pattern. 24 is the React
  Native 0.82+ minimum; 26 is a safe modern choice (drops Android 7); 28 drops Android 8 too. Pick based on your audience, not because someone wrote 28 in a guide.

  runtimeVersion: appVersion

  This ties OTA updates to your app version — if you bump version in app.json, you get a new runtime version automatically. Good default. The alternative "sdkVersion" policy ties
  updates to the Expo SDK; "fingerprint" ties to the actual native code fingerprint (more correct, more complex).

  scheme

  You'll want this for deep links. Expo Router auto-generates deep links for every route based on your file structure (/reader/[id] → contentlib://reader/abc123). Set the scheme here.

  ---
  11. Running the app

  Iterative dev with Expo Go

  npx expo start

  This is fine for SDK 54 projects if you're using Expo Go on a physical device. For SDK 56, Expo Go's bundled native code may not match — you'll hit module mismatch errors as soon as
  you use anything beyond the core.

  Development build (recommended for SDK 56)

  npx expo install expo-dev-client
  npx expo run:ios       # first build: ~5–10 min (now 50%+ faster than SDK 54 thanks to prebuilt XCFrameworks)
  npx expo run:android   # first build: faster than before too, especially if you enable usePrecompiledHeaders

  The dev client is a custom Expo Go that includes your project's exact native modules. After the first run:ios/run:android, subsequent JS changes are hot-reloaded via npx expo start
  against the dev client — you don't rebuild natively unless you add a new native module.

  Android build perf in SDK 56

  Opt-in flag in expo-build-properties:

  [
    "expo-build-properties",
    {
      "android": { "usePrecompiledHeaders": true }
    }
  ]

  Applies CMake precompiled headers to autolinked native modules. Upstream benchmark dropped :app:buildCMakeDebug from 17m 10s to 6m 06s (2.81× speedup). Default project gets ~1.3×
  speedup. More autolinked native modules = bigger win.

  ---
  12. The complete corrected install script

  # 1. Init with the real default template
  npx create-expo-app@latest --template default@sdk-56 my-content-library
  cd my-content-library

  # 2. Reader-app native deps (versioned to SDK 56)
  npx expo install \
    expo-sqlite \
    expo-file-system \
    expo-document-picker \
    expo-sharing \
    expo-build-properties

  # 3. Pure-JS deps
  pnpm add react-native-enriched-markdown fuse.js

  # 4. Styling (optional)
  npx expo install nativewind
  pnpm add -D tailwindcss

  # 5. Dev client + first native build
  npx expo install expo-dev-client
  npx expo run:ios
