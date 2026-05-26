import React, { useMemo } from 'react';
import { View, Text, FlatList, StyleSheet } from 'react-native';
import { useLocalSearchParams, useNavigation } from 'expo-router';
import { getPapersForCategory, graph } from '../../lib/graph';
import { getAllReadIds } from '../../lib/db';
import PaperCard from '../../components/PaperCard';
import type { Paper } from '../../lib/types';

export default function CategoryScreen() {
  const { category } = useLocalSearchParams<{ category: string }>();
  const navigation = useNavigation();

  const catMeta = category ? graph.categories[category] : undefined;
  const papers = useMemo(() =>
    category ? getPapersForCategory(category) : [],
    [category]
  );
  const readIds = useMemo(() => new Set(getAllReadIds()), []);

  // Update header title
  React.useEffect(() => {
    if (catMeta) {
      navigation.setOptions({ title: catMeta.title });
    }
  }, [catMeta, navigation]);

  if (!catMeta) {
    return (
      <View style={styles.error}>
        <Text style={styles.errorText}>Category not found</Text>
      </View>
    );
  }

  const renderHeader = () => (
    <View style={[styles.header, { borderLeftColor: catMeta.color }]}>
      <Text style={[styles.headerTitle, { color: catMeta.color }]}>{catMeta.title}</Text>
      <Text style={styles.headerBlurb}>{catMeta.blurb}</Text>
      <Text style={styles.headerCount}>{catMeta.count} papers</Text>
    </View>
  );

  const renderItem = ({ item }: { item: Paper }) => (
    <PaperCard
      paper={item}
      category={catMeta}
      isRead={readIds.has(item.id)}
    />
  );

  return (
    <FlatList
      data={papers}
      keyExtractor={(item) => item.id}
      renderItem={renderItem}
      ListHeaderComponent={renderHeader}
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
  header: {
    borderLeftWidth: 3,
    paddingLeft: 12,
    marginBottom: 16,
    gap: 4,
  },
  headerTitle: {
    fontSize: 22,
    fontWeight: '700',
  },
  headerBlurb: {
    fontSize: 14,
    color: '#a0a0a0',
    lineHeight: 20,
  },
  headerCount: {
    fontSize: 13,
    color: '#666666',
  },
  error: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#0a0a0a',
  },
  errorText: {
    color: '#666666',
    fontSize: 16,
  },
});
