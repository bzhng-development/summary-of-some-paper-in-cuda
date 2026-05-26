import React, { useMemo, useCallback } from 'react';
import { View, Text, Pressable, StyleSheet, Dimensions, ScrollView } from 'react-native';
import { useRouter } from 'expo-router';
import Svg, { Circle, Line, Text as SvgText, G } from 'react-native-svg';
import { graph, listCategories } from '../lib/graph';
import type { CategoryMeta } from '../lib/types';

const { width: SCREEN_W, height: SCREEN_H } = Dimensions.get('window');
const CANVAS_W = SCREEN_W - 32;
const CANVAS_H = SCREEN_H * 0.6;

// Layout categories in a circular arrangement
function layoutNodes(categories: CategoryMeta[]) {
  const total = categories.length;
  const cx = CANVAS_W / 2;
  const cy = CANVAS_H / 2;
  const maxCount = Math.max(...categories.map((c) => c.count));
  const radius = Math.min(CANVAS_W, CANVAS_H) * 0.38;

  return categories.map((cat, i) => {
    const angle = (i / total) * 2 * Math.PI - Math.PI / 2;
    const x = cx + radius * Math.cos(angle);
    const y = cy + radius * Math.sin(angle);
    const r = 12 + (cat.count / maxCount) * 28; // bubble radius proportional to count
    return { cat, x, y, r };
  });
}

export default function GraphScreen() {
  const router = useRouter();
  const categories = useMemo(() => listCategories(), []);
  const nodes = useMemo(() => layoutNodes(categories), [categories]);

  // Build node position index for edge drawing
  const nodeMap = useMemo(() => {
    const m: Record<string, { x: number; y: number; r: number; color: string }> = {};
    for (const n of nodes) {
      m[n.cat.slug] = { x: n.x, y: n.y, r: n.r, color: n.cat.color };
    }
    return m;
  }, [nodes]);

  const handleNodePress = useCallback((slug: string) => {
    router.push(`/c/${slug}` as never);
  }, [router]);

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.content}>
      <Text style={styles.heading}>Domain Graph</Text>
      <Text style={styles.subheading}>
        {graph.counts.categories} domains · {graph.counts.papers} papers · bubble size = paper count
      </Text>

      <View style={styles.canvas}>
        <Svg width={CANVAS_W} height={CANVAS_H}>
          {/* Domain bridge edges */}
          {graph.domainBridges.map((bridge, i) => {
            const a = nodeMap[bridge.a];
            const b = nodeMap[bridge.b];
            if (!a || !b) return null;
            return (
              <Line
                key={`bridge-${i}`}
                x1={a.x} y1={a.y}
                x2={b.x} y2={b.y}
                stroke="#2a2a2a"
                strokeWidth={1}
                opacity={0.6}
              />
            );
          })}

          {/* Nodes */}
          {nodes.map(({ cat, x, y, r }) => (
            <G
              key={cat.slug}
              onPress={() => handleNodePress(cat.slug)}
            >
              <Circle
                cx={x}
                cy={y}
                r={r}
                fill={cat.color + '33'}
                stroke={cat.color}
                strokeWidth={1.5}
              />
              <SvgText
                x={x}
                y={y - r - 4}
                textAnchor="middle"
                fill={cat.color}
                fontSize={9}
                fontWeight="600"
              >
                {cat.title.split(' ').slice(0, 2).join(' ')}
              </SvgText>
              <SvgText
                x={x}
                y={y + 4}
                textAnchor="middle"
                fill="#ffffff"
                fontSize={10}
                fontWeight="700"
              >
                {cat.count}
              </SvgText>
            </G>
          ))}
        </Svg>
      </View>

      {/* Category list below the graph */}
      <Text style={styles.listTitle}>All Domains</Text>
      <View style={styles.catGrid}>
        {categories.map((cat) => (
          <Pressable
            key={cat.slug}
            style={({ pressed }) => [
              styles.catItem,
              { borderLeftColor: cat.color, opacity: pressed ? 0.8 : 1 },
            ]}
            onPress={() => router.push(`/c/${cat.slug}` as never)}
          >
            <Text style={[styles.catItemTitle, { color: cat.color }]} numberOfLines={1}>
              {cat.title}
            </Text>
            <Text style={styles.catItemCount}>{cat.count}</Text>
          </Pressable>
        ))}
      </View>
    </ScrollView>
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
    gap: 12,
  },
  heading: {
    fontSize: 24,
    fontWeight: '700',
    color: '#ffffff',
  },
  subheading: {
    fontSize: 13,
    color: '#666666',
  },
  canvas: {
    backgroundColor: '#111111',
    borderRadius: 12,
    overflow: 'hidden',
    borderWidth: 1,
    borderColor: '#2a2a2a',
  },
  listTitle: {
    fontSize: 17,
    fontWeight: '700',
    color: '#ffffff',
    marginTop: 8,
  },
  catGrid: {
    gap: 8,
  },
  catItem: {
    backgroundColor: '#1a1a1a',
    borderRadius: 8,
    borderLeftWidth: 3,
    paddingHorizontal: 12,
    paddingVertical: 10,
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  catItemTitle: {
    fontSize: 14,
    fontWeight: '600',
    flex: 1,
  },
  catItemCount: {
    fontSize: 13,
    color: '#666666',
    marginLeft: 8,
  },
});
