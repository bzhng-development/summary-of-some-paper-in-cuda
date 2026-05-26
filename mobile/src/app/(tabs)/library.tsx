import React, { useMemo } from 'react';
import { View, Text, FlatList, Pressable, StyleSheet } from 'react-native';
import { useRouter } from 'expo-router';
import { listCategories, graph, getPaperById } from '../../lib/graph';
import { getAllReadIds, getRecentIds, getStreak } from '../../lib/db';
import CategoryCard from '../../components/CategoryCard';
import PaperCard from '../../components/PaperCard';
import type { CategoryMeta, Paper } from '../../lib/types';

export default function LibraryScreen() {
  const router = useRouter();
  const categories = useMemo(() => listCategories(), []);
  const readIds = useMemo(() => new Set(getAllReadIds()), []);
  const recentIds = useMemo(() => getRecentIds(), []);
  const streak = useMemo(() => getStreak(), []);

  const recentPapers = useMemo(() =>
    recentIds.slice(0, 6).map((id) => getPaperById(id)).filter((p): p is Paper => p != null),
    [recentIds]
  );

  const readCountByCategory = useMemo(() => {
    const map: Record<string, number> = {};
    for (const id of readIds) {
      const cat = id.split('/')[0];
      map[cat] = (map[cat] ?? 0) + 1;
    }
    return map;
  }, [readIds]);

  type ListItem =
    | { type: 'header-stats' }
    | { type: 'section-title'; label: string; action?: () => void }
    | { type: 'recent-paper'; paper: Paper }
    | { type: 'category'; cat: CategoryMeta }
    | { type: 'graph-btn' };

  const listData: ListItem[] = useMemo(() => {
    const items: ListItem[] = [];
    items.push({ type: 'header-stats' });

    if (recentPapers.length > 0) {
      items.push({ type: 'section-title', label: 'Recently Opened' });
      for (const paper of recentPapers) {
        items.push({ type: 'recent-paper', paper });
      }
    }

    items.push({
      type: 'section-title',
      label: 'Domains',
      action: () => router.push('/graph' as never),
    });

    for (const cat of categories) {
      items.push({ type: 'category', cat });
    }

    items.push({ type: 'graph-btn' });
    return items;
  }, [recentPapers, categories, router]);

  const renderItem = ({ item }: { item: ListItem }) => {
    switch (item.type) {
      case 'header-stats':
        return (
          <View style={styles.statsBar}>
            <View style={styles.statItem}>
              <Text style={styles.statNum}>{graph.counts.papers}</Text>
              <Text style={styles.statLabel}>Papers</Text>
            </View>
            <View style={styles.statDivider} />
            <View style={styles.statItem}>
              <Text style={styles.statNum}>{readIds.size}</Text>
              <Text style={styles.statLabel}>Read</Text>
            </View>
            <View style={styles.statDivider} />
            <View style={styles.statItem}>
              <Text style={[styles.statNum, { color: '#00E599' }]}>{streak}</Text>
              <Text style={styles.statLabel}>Day Streak</Text>
            </View>
          </View>
        );

      case 'section-title':
        return (
          <View style={styles.sectionHeader}>
            <Text style={styles.sectionTitle}>{item.label}</Text>
            {item.action && (
              <Pressable onPress={item.action}>
                <Text style={styles.sectionAction}>View graph →</Text>
              </Pressable>
            )}
          </View>
        );

      case 'recent-paper':
        return (
          <PaperCard
            paper={item.paper}
            category={graph.categories[item.paper.category]}
            isRead={readIds.has(item.paper.id)}
            compact
          />
        );

      case 'category':
        return (
          <CategoryCard
            category={item.cat}
            readCount={readCountByCategory[item.cat.slug] ?? 0}
          />
        );

      case 'graph-btn':
        return (
          <Pressable style={styles.graphBtn} onPress={() => router.push('/graph' as never)}>
            <Text style={styles.graphBtnText}>Open Domain Graph</Text>
          </Pressable>
        );

      default:
        return null;
    }
  };

  return (
    <FlatList
      data={listData}
      keyExtractor={(item, i) => {
        if (item.type === 'recent-paper') return `recent-${item.paper.id}`;
        if (item.type === 'category') return `cat-${item.cat.slug}`;
        return `${item.type}-${i}`;
      }}
      renderItem={renderItem}
      contentContainerStyle={styles.list}
      showsVerticalScrollIndicator={false}
    />
  );
}

const styles = StyleSheet.create({
  list: {
    padding: 16,
    paddingBottom: 40,
  },
  statsBar: {
    flexDirection: 'row',
    backgroundColor: '#1a1a1a',
    borderRadius: 10,
    padding: 16,
    marginBottom: 20,
    justifyContent: 'space-around',
  },
  statItem: {
    alignItems: 'center',
    gap: 4,
  },
  statNum: {
    fontSize: 22,
    fontWeight: '700',
    color: '#aa99ff',
  },
  statLabel: {
    fontSize: 12,
    color: '#666666',
  },
  statDivider: {
    width: 1,
    backgroundColor: '#2a2a2a',
  },
  sectionHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 10,
    marginTop: 6,
  },
  sectionTitle: {
    fontSize: 17,
    fontWeight: '700',
    color: '#ffffff',
  },
  sectionAction: {
    fontSize: 13,
    color: '#aa99ff',
  },
  graphBtn: {
    marginTop: 16,
    backgroundColor: '#1a1a1a',
    borderRadius: 10,
    padding: 16,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: '#2a2a2a',
  },
  graphBtnText: {
    color: '#aa99ff',
    fontWeight: '600',
    fontSize: 15,
  },
});
