import React, { memo } from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { useRouter } from 'expo-router';
import type { CategoryMeta } from '../lib/types';
import { colors, radii, spacing, typography } from '../lib/theme';
import Card from './ui/Card';
import GradientPill from './ui/GradientPill';

interface Props {
  category: CategoryMeta;
  readCount?: number;
}

const CategoryCard = memo(function CategoryCard({ category, readCount = 0 }: Props) {
  const router = useRouter();
  const progress = category.count > 0 ? readCount / category.count : 0;
  const accent = category.color;

  return (
    <Card
      onPress={() => router.push(`/c/${category.slug}` as never)}
      accessibilityLabel={`${category.title} category, ${category.count} papers`}
      style={styles.card}
      padding="xl"
    >
      <View style={styles.header}>
        <View style={[styles.dot, { backgroundColor: accent }]} />
        <GradientPill color={accent}>{category.count} papers</GradientPill>
      </View>

      <Text style={[styles.title, { color: colors.white }]} numberOfLines={2}>
        {category.title}
      </Text>
      <Text style={styles.blurb} numberOfLines={2}>
        {category.blurb}
      </Text>

      {progress > 0 && (
        <View style={styles.progressRow}>
          <View style={styles.progressBar}>
            <View
              style={[
                styles.progressFill,
                { width: `${Math.round(progress * 100)}%`, backgroundColor: accent },
              ]}
            />
          </View>
          <Text style={[styles.progressLabel, { color: accent }]}>
            {readCount}/{category.count}
          </Text>
        </View>
      )}
    </Card>
  );
});

export default CategoryCard;

const styles = StyleSheet.create({
  card: {
    marginBottom: spacing.md,
    gap: spacing.sm,
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.sm,
  },
  dot: {
    width: 10,
    height: 10,
    borderRadius: 5,
  },
  title: {
    ...typography.h2,
  },
  blurb: {
    ...typography.bodySm,
    color: colors.textBody,
  },
  progressRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.sm,
    marginTop: spacing.sm,
  },
  progressBar: {
    flex: 1,
    height: 3,
    backgroundColor: colors.borderStrong,
    borderRadius: 2,
    overflow: 'hidden',
  },
  progressFill: {
    height: 3,
    borderRadius: 2,
  },
  progressLabel: {
    ...typography.caption,
    fontWeight: '600',
  },
});
