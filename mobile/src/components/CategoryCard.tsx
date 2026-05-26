import React, { memo } from 'react';
import { View, Text, Pressable, StyleSheet } from 'react-native';
import { useRouter } from 'expo-router';
import type { CategoryMeta } from '../lib/types';

interface Props {
  category: CategoryMeta;
  readCount?: number;
}

const CategoryCard = memo(function CategoryCard({ category, readCount = 0 }: Props) {
  const router = useRouter();
  const progress = category.count > 0 ? readCount / category.count : 0;

  return (
    <Pressable
      style={({ pressed }) => [styles.card, { opacity: pressed ? 0.85 : 1 }]}
      onPress={() => router.push(`/c/${category.slug}` as never)}
      accessibilityRole="button"
      accessibilityLabel={`${category.title} category, ${category.count} papers`}
    >
      {/* Gradient border top */}
      <View style={[styles.topBar, { backgroundColor: category.color }]} />

      <View style={styles.body}>
        <Text style={[styles.title, { color: category.color }]} numberOfLines={2}>
          {category.title}
        </Text>
        <Text style={styles.blurb} numberOfLines={2}>{category.blurb}</Text>

        <View style={styles.footer}>
          <Text style={styles.count}>{category.count} papers</Text>
          {readCount > 0 && (
            <Text style={[styles.progress, { color: category.color }]}>
              {readCount}/{category.count}
            </Text>
          )}
        </View>

        {/* Progress bar */}
        {progress > 0 && (
          <View style={styles.progressBar}>
            <View style={[styles.progressFill, { width: `${Math.round(progress * 100)}%`, backgroundColor: category.color }]} />
          </View>
        )}
      </View>
    </Pressable>
  );
});

export default CategoryCard;

const styles = StyleSheet.create({
  card: {
    backgroundColor: '#1a1a1a',
    borderRadius: 10,
    overflow: 'hidden',
    marginBottom: 10,
  },
  topBar: {
    height: 3,
  },
  body: {
    padding: 14,
    gap: 6,
  },
  title: {
    fontSize: 16,
    fontWeight: '700',
  },
  blurb: {
    fontSize: 13,
    color: '#a0a0a0',
    lineHeight: 18,
  },
  footer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginTop: 4,
  },
  count: {
    fontSize: 12,
    color: '#666666',
  },
  progress: {
    fontSize: 12,
    fontWeight: '600',
  },
  progressBar: {
    height: 2,
    backgroundColor: '#2a2a2a',
    borderRadius: 1,
    marginTop: 6,
  },
  progressFill: {
    height: 2,
    borderRadius: 1,
  },
});
