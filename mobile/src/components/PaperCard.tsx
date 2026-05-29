import React, { memo } from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { useRouter } from 'expo-router';
import type { Paper, CategoryMeta } from '../lib/types';
import { colors, radii, spacing, typography } from '../lib/theme';
import Card from './ui/Card';
import GradientPill from './ui/GradientPill';

interface Props {
  paper: Paper;
  category?: CategoryMeta;
  isRead?: boolean;
  compact?: boolean;
}

const PaperCard = memo(function PaperCard({ paper, category, isRead = false, compact = false }: Props) {
  const router = useRouter();
  const accent = category?.color ?? colors.purple;

  return (
    <Card
      onPress={() => router.push(`/p/${paper.category}/${paper.slug}` as never)}
      accessibilityLabel={paper.title}
      style={[styles.card, isRead && styles.cardRead] as never}
      padding="xl"
    >
      <View style={styles.header}>
        <View style={styles.headerLeft}>
          <View style={[styles.dot, { backgroundColor: accent }]} />
          <Text style={[styles.category, { color: accent }]} numberOfLines={1}>
            {category?.title ?? paper.category}
          </Text>
        </View>
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
          <GradientPill color={accent}>★ {paper.score}</GradientPill>
        )}
        {paper.upvotes != null && (
          <Text style={styles.meta}>↑ {paper.upvotes}</Text>
        )}
        {isRead && (
          <GradientPill color={colors.green}>✓ Read</GradientPill>
        )}
      </View>
    </Card>
  );
});

export default PaperCard;

const styles = StyleSheet.create({
  card: {
    marginBottom: spacing.md,
    gap: spacing.sm,
  },
  cardRead: {
    opacity: 0.7,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  headerLeft: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.sm,
    flex: 1,
  },
  dot: {
    width: 8,
    height: 8,
    borderRadius: 4,
  },
  category: {
    ...typography.micro,
    textTransform: 'uppercase',
    flex: 1,
  },
  year: {
    ...typography.caption,
    color: colors.textMuted,
    marginLeft: spacing.sm,
  },
  title: {
    ...typography.h3,
    color: colors.white,
  },
  titleRead: {
    color: colors.textBody,
  },
  pitch: {
    ...typography.bodySm,
    color: colors.textBody,
  },
  footer: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.sm,
    flexWrap: 'wrap',
    marginTop: spacing.xs,
  },
  meta: {
    ...typography.caption,
    color: colors.textMuted,
  },
});
