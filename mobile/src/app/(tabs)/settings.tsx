import React, { useState } from 'react';
import { View, Text, Pressable, StyleSheet, ScrollView } from 'react-native';
import { getAllReadIds, getStreak, getTheme, setTheme, getFontSize, setFontSize } from '../../lib/db';
import { graph } from '../../lib/graph';

export default function SettingsScreen() {
  const [theme, setThemeState] = useState(() => getTheme());
  const [fontSize, setFontSizeState] = useState(() => getFontSize());
  const readCount = getAllReadIds().length;
  const streak = getStreak();

  const handleThemeToggle = () => {
    const next = theme === 'dark' ? 'light' : 'dark';
    setTheme(next);
    setThemeState(next);
  };

  const handleFontSize = (delta: number) => {
    const next = Math.max(12, Math.min(22, fontSize + delta));
    setFontSize(next);
    setFontSizeState(next);
  };

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.content}>
      {/* Stats */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Reading Progress</Text>
        <View style={styles.card}>
          <Row label="Papers read" value={`${readCount} / ${graph.counts.papers}`} />
          <Row label="Day streak" value={`${streak} days`} accent="#00E599" />
          <Row label="Categories" value={`${graph.counts.categories}`} />
        </View>
      </View>

      {/* Theme */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Appearance</Text>
        <View style={styles.card}>
          <View style={styles.row}>
            <Text style={styles.rowLabel}>Theme</Text>
            <Pressable style={styles.toggle} onPress={handleThemeToggle}>
              <Text style={styles.toggleText}>{theme === 'dark' ? 'Dark' : 'Light'}</Text>
            </Pressable>
          </View>
          <View style={styles.row}>
            <Text style={styles.rowLabel}>Font size ({fontSize}px)</Text>
            <View style={styles.stepper}>
              <Pressable style={styles.stepBtn} onPress={() => handleFontSize(-1)}>
                <Text style={styles.stepText}>−</Text>
              </Pressable>
              <Pressable style={styles.stepBtn} onPress={() => handleFontSize(1)}>
                <Text style={styles.stepText}>+</Text>
              </Pressable>
            </View>
          </View>
        </View>
      </View>

      {/* About */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>About</Text>
        <View style={styles.card}>
          <Row label="App" value="Paper Graph Mobile" />
          <Row label="Papers" value={String(graph.counts.papers)} />
          <Row label="Generated" value={graph.generatedAt.slice(0, 10)} />
          <Row label="Source" value="paper-graph-ui.vercel.app" />
        </View>
      </View>
    </ScrollView>
  );
}

function Row({ label, value, accent }: { label: string; value: string; accent?: string }) {
  return (
    <View style={styles.row}>
      <Text style={styles.rowLabel}>{label}</Text>
      <Text style={[styles.rowValue, accent ? { color: accent } : {}]}>{value}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#0a0a0a',
  },
  content: {
    padding: 16,
    paddingBottom: 40,
    gap: 24,
  },
  section: {
    gap: 8,
  },
  sectionTitle: {
    fontSize: 13,
    fontWeight: '600',
    textTransform: 'uppercase',
    letterSpacing: 0.8,
    color: '#666666',
  },
  card: {
    backgroundColor: '#1a1a1a',
    borderRadius: 10,
    overflow: 'hidden',
  },
  row: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: 14,
    borderBottomWidth: 1,
    borderBottomColor: '#2a2a2a',
  },
  rowLabel: {
    fontSize: 15,
    color: '#ffffff',
  },
  rowValue: {
    fontSize: 14,
    color: '#a0a0a0',
  },
  toggle: {
    backgroundColor: '#2a2a2a',
    borderRadius: 6,
    paddingHorizontal: 12,
    paddingVertical: 6,
  },
  toggleText: {
    color: '#aa99ff',
    fontWeight: '600',
    fontSize: 14,
  },
  stepper: {
    flexDirection: 'row',
    gap: 8,
  },
  stepBtn: {
    backgroundColor: '#2a2a2a',
    borderRadius: 6,
    width: 32,
    height: 32,
    alignItems: 'center',
    justifyContent: 'center',
  },
  stepText: {
    color: '#aa99ff',
    fontSize: 18,
    fontWeight: '600',
  },
});
