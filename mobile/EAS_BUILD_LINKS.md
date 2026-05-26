# EAS Build Links — Paper Graph Mobile

Builds kicked off: 2026-05-26

## Android (APK, internal distribution)
- **Profile:** development
- **Status at kick-off:** IN_PROGRESS
- **Build URL:** https://expo.dev/accounts/vincentzhongy/projects/paper-graph-mobile/builds/4bcf708f-eb1c-415f-a0c8-da4b803a63ff
- **Install:** Download the APK from the build URL above and sideload on any Android device

## iOS (Simulator build, .app)
- **Profile:** development-simulator
- **Status at kick-off:** IN_QUEUE
- **Build URL:** https://expo.dev/accounts/vincentzhongy/projects/paper-graph-mobile/builds/f7241ad5-fcd3-4c00-a041-b8737b9287b5
- **Install:** Download the .tar.gz, extract, drag into Xcode Simulator

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
