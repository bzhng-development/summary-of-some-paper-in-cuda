import { Stack } from 'expo-router';
import { StatusBar } from 'expo-status-bar';
import { useColorScheme } from 'react-native';

export default function RootLayout() {
  return (
    <>
      <StatusBar style="light" />
      <Stack
        screenOptions={{
          headerStyle: { backgroundColor: '#0a0a0a' },
          headerTintColor: '#ffffff',
          headerTitleStyle: { fontWeight: '700', color: '#ffffff' },
          contentStyle: { backgroundColor: '#0a0a0a' },
          headerBackTitle: 'Back',
        }}
      >
        <Stack.Screen name="(tabs)" options={{ headerShown: false }} />
        <Stack.Screen name="c/[category]" options={{ title: 'Category' }} />
        <Stack.Screen name="p/[category]/[slug]" options={{ title: 'Paper' }} />
        <Stack.Screen name="topic/[topic]" options={{ title: 'Topic' }} />
        <Stack.Screen name="graph" options={{ title: 'Domain Graph', presentation: 'modal' }} />
      </Stack>
    </>
  );
}
