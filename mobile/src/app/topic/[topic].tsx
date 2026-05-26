import React, { useMemo, useEffect } from 'react';
import { View, Text, FlatList, StyleSheet } from 'react-native';
import { useLocalSearchParams, useNavigation } from 'expo-router';
import { getPapersForTopic, graph } from '../../lib/graph';
import { getAllReadIds } from '../../lib/db';
import PaperCard from '../../components/PaperCard';
import type { Paper } from '../../lib/types';

export default function TopicScreen() {
  const { topic } = useLocalSearchParams<{ topic: string }>();
  const navigation = useNavigation();

  const topicMeta = useMemo(
    () => graph.topics.find((t) => t.id === topic),
    [topic]
  );
  const papers = useMemo(
    () => (topic ? getPapersForTopic(topic) : []),
    [topic]
  );
  const readIds = useMemo(() => new Set(getAllReadIds()), []);

  useEffect(() => {
    if (topicMeta) {
      navigation.setOptions({ title: topicMeta.label });
    }
  }, [topicMeta, navigation]);

  const renderItem = ({ item }: { item: Paper }) => (
    <PaperCard
      paper={item}
      category={graph.categories[item.category]}
      isRead={readIds.has(item.id)}
    />
  );

  return (
    <FlatList
      data={papers}
      keyExtractor={(item) => item.id}
      renderItem={renderItem}
      ListHeaderComponent={
        <View style={styles.header}>
          <Text style={styles.label}>Topic Thread</Text>
          <Text style={styles.title}>{topicMeta?.label ?? topic}</Text>
          <Text style={styles.count}>{papers.length} papers across domains</Text>
        </View>
      }
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
    marginBottom: 16,
    gap: 4,
  },
  label: {
    fontSize: 11,
    textTransform: 'uppercase',
    letterSpacing: 0.8,
    color: '#666666',
    fontWeight: '600',
  },
  title: {
    fontSize: 22,
    fontWeight: '700',
    color: '#aa99ff',
  },
  count: {
    fontSize: 13,
    color: '#666666',
  },
});
