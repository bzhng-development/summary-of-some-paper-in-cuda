import React, { memo } from 'react';
import { View, Text, Pressable, StyleSheet } from 'react-native';
import { useRouter } from 'expo-router';
import type { Paper, CategoryMeta } from '../lib/types';

interface Props {
  paper: Paper;
  category?: CategoryMeta;
  isRead?: boolean;
  compact?: boolean;
}

const PaperCard = memo(function PaperCard({ paper, category, isRead = false, compact = false }: Props) {
  const router = useRouter();
  const accentColor = category?.color ?? '#aa99ff';

  const handlePress = () => {
    router.push(`/p/${paper.category}/${paper.slug}` as never);
  };

  return (
    <Pressable
      style={({ pressed }) => [
        styles.card,
        { borderLeftColor: accentColor, opacity: pressed ? 0.85 : 1 },
        isRead && styles.cardRead,
      ]}
      onPress={handlePress}
      accessibilityRole="button"
      accessibilityLabel={paper.title}
    >
      <View style={styles.header}>
        <Text style={[styles.category, { color: accentColor }]} numberOfLines={1}>
          {category?.title ?? paper.category}
        </Text>
        <Text style={styles.year}>{paper.year}</Text>
      </View>

      <Text style={[styles.title, isRead && styles.titleRead]} numberOfLines={compact ? 2 : 3}>
        {paper.title}
      </Text>

      {!compact && paper.pitch && (
        <Text style={styles.pitch} numberOfLines={3}>
          {paper.pitch}
        </Text>
      )}

      <View style={styles.footer}>
        <Text style={styles.meta}>{paper.readTimeMin} min read</Text>
        {paper.score != null && (
          <View style={[styles.scorePill, { backgroundColor: accentColor + '22' }]}>
            <Text style={[styles.scoreText, { color: accentColor }]}>★ {paper.score}</Text>
          </View>
        )}
        {paper.upvotes != null && (
          <Text style={styles.meta}>↑ {paper.upvotes}</Text>
        )}
        {isRead && <Text style={styles.readBadge}>✓ Read</Text>}
      </View>
    </Pressable>
  );
});

export default PaperCard;

const styles = StyleSheet.create({
  card: {
    backgroundColor: '#1a1a1a',
    borderRadius: 10,
    borderLeftWidth: 3,
    padding: 14,
    marginBottom: 10,
    gap: 6,
  },
  cardRead: {
    opacity: 0.7,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  category: {
    fontSize: 11,
    fontWeight: '600',
    textTransform: 'uppercase',
    letterSpacing: 0.5,
    flex: 1,
  },
  year: {
    fontSize: 12,
    color: '#666666',
    marginLeft: 8,
  },
  title: {
    fontSize: 15,
    fontWeight: '600',
    color: '#ffffff',
    lineHeight: 21,
  },
  titleRead: {
    color: '#888888',
  },
  pitch: {
    fontSize: 13,
    color: '#a0a0a0',
    lineHeight: 18,
  },
  footer: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    flexWrap: 'wrap',
    marginTop: 2,
  },
  meta: {
    fontSize: 12,
    color: '#666666',
  },
  scorePill: {
    borderRadius: 6,
    paddingHorizontal: 6,
    paddingVertical: 2,
  },
  scoreText: {
    fontSize: 11,
    fontWeight: '600',
  },
  readBadge: {
    fontSize: 11,
    color: '#00E599',
  },
});
