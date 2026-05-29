import React, { useMemo, useEffect } from 'react';
import {
  View,
  Text,
  ScrollView,
  Pressable,
  StyleSheet,
  Linking,
  ActivityIndicator,
} from 'react-native';
import { useLocalSearchParams, useNavigation, useRouter } from 'expo-router';
import { getPaper, getAdjacentPapers, getRelatedPapers, graph } from '../../../lib/graph';
import { recordOpened } from '../../../lib/db';
import { useReadState } from '../../../hooks/useReadState';
import { useMarkdownContent } from '../../../hooks/useMarkdownContent';
import { colors, radii, spacing, typography } from '../../../lib/theme';
import Card from '../../../components/ui/Card';
import GradientPill from '../../../components/ui/GradientPill';

// Markdown rendering — react-native-enriched-markdown
// Note: requires dev build (not compatible with Expo Go)
import type { MarkdownStyle } from 'react-native-enriched-markdown';
let MarkdownRenderer:
  | React.ComponentType<{ markdown: string; markdownStyle?: MarkdownStyle }>
  | null = null;
try {
  // eslint-disable-next-line @typescript-eslint/no-var-requires
  const mod = require('react-native-enriched-markdown');
  MarkdownRenderer = mod.EnrichedMarkdownText ?? null;
} catch {
  MarkdownRenderer = null;
}

const darkMarkdownStyle: MarkdownStyle = {
  paragraph: { color: colors.textBody, fontSize: 15, lineHeight: 24, marginBottom: 8 },
  h1: { color: colors.white, fontSize: 24, fontWeight: '700', marginTop: 12, marginBottom: 8 },
  h2: { color: colors.white, fontSize: 20, fontWeight: '700', marginTop: 20, marginBottom: 6 },
  h3: { color: colors.textStrong, fontSize: 17, fontWeight: '600', marginTop: 16, marginBottom: 4 },
  h4: { color: colors.textStrong, fontSize: 15, fontWeight: '600', marginTop: 14, marginBottom: 4 },
  h5: { color: colors.textBody, fontSize: 14, fontWeight: '600' },
  h6: { color: colors.textBody, fontSize: 13, fontWeight: '600' },
  strong: { color: colors.white, fontWeight: 'bold' },
  em: { color: colors.textStrong, fontStyle: 'italic' },
  link: { color: colors.purple, underline: true },
  code: { color: colors.green, backgroundColor: colors.cardBg, fontFamily: 'Menlo', fontSize: 14 },
  codeBlock: {
    color: colors.textStrong,
    backgroundColor: colors.cardBg,
    fontFamily: 'Menlo',
    fontSize: 13,
    padding: 12,
    borderRadius: radii.md,
    marginTop: 8,
    marginBottom: 8,
  },
  blockquote: {
    color: colors.textBody,
    borderColor: colors.purple,
    borderWidth: 3,
    gapWidth: 8,
    fontSize: 15,
  },
  list: { color: colors.textStrong, bulletColor: colors.green, markerColor: colors.green, fontSize: 15 },
  thematicBreak: { color: colors.border, height: 1 },
};

export default function PaperDetailScreen() {
  const { category, slug } = useLocalSearchParams<{ category: string; slug: string }>();
  const navigation = useNavigation();
  const router = useRouter();

  const paper = useMemo(
    () => (category && slug ? getPaper(category, slug) : undefined),
    [category, slug]
  );

  const { read, toggle: toggleRead } = useReadState(paper?.id ?? '');
  const catMeta = paper ? graph.categories[paper.category] : undefined;

  const { prev, next } = useMemo(
    () => (paper ? getAdjacentPapers(paper) : { prev: null, next: null }),
    [paper]
  );
  const related = useMemo(() => (paper ? getRelatedPapers(paper) : []), [paper]);

  useEffect(() => {
    if (paper) {
      recordOpened(paper.id);
      navigation.setOptions({
        title: paper.title.slice(0, 40) + (paper.title.length > 40 ? '…' : ''),
      });
    }
  }, [paper, navigation]);

  // Hook must be called unconditionally — pass safe placeholders when paper is missing.
  const { content: bodyContent, loading: bodyLoading, error: bodyError } = useMarkdownContent(
    paper?.category ?? '',
    paper?.slug ?? ''
  );

  if (!paper) {
    return (
      <View style={styles.error}>
        <Text style={styles.errorText}>Paper not found</Text>
      </View>
    );
  }

  const accent = catMeta?.color ?? colors.purple;

  const openArxiv = () => paper.arxivId && Linking.openURL(`https://arxiv.org/abs/${paper.arxivId}`);
  const openGithub = () => paper.github && Linking.openURL(paper.github);

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.content}>
      {/* Header card — mirrors web's GradientBorder header */}
      <Card padding="2xl" style={styles.headerCard}>
        <Pressable
          style={styles.catRow}
          onPress={() => router.push(`/c/${paper.category}` as never)}
        >
          <View style={[styles.dot, { backgroundColor: accent }]} />
          <Text style={[styles.catLabel, { color: accent }]} numberOfLines={1}>
            {catMeta?.title ?? paper.category}
          </Text>
        </Pressable>

        <Text style={styles.title}>{paper.title}</Text>

        <View style={styles.pillRow}>
          <GradientPill color={colors.textBody}>{paper.year}</GradientPill>
          {paper.month != null && (
            <GradientPill color={colors.textBody}>
              {paper.month.toString().padStart(2, '0')}
            </GradientPill>
          )}
          <GradientPill color={colors.textBody}>{paper.readTimeMin} min</GradientPill>
          {paper.score != null && (
            <GradientPill color={accent}>★ {paper.score}/10</GradientPill>
          )}
          {paper.upvotes != null && (
            <GradientPill color={colors.textBody}>↑ {paper.upvotes}</GradientPill>
          )}
        </View>

        {paper.organization && (
          <Text style={styles.org}>{paper.organization}</Text>
        )}

        <View style={styles.actions}>
          <Pressable
            style={[
              styles.actionBtn,
              read ? { backgroundColor: colors.green + '22' } : styles.actionBtnDefault,
            ]}
            onPress={toggleRead}
          >
            <Text
              style={[styles.actionBtnText, { color: read ? colors.green : colors.white }]}
            >
              {read ? '✓ Marked Read' : 'Mark as Read'}
            </Text>
          </Pressable>

          {paper.arxivId && (
            <Pressable style={styles.actionBtnSecondary} onPress={openArxiv}>
              <Text style={styles.actionBtnSecondaryText}>ArXiv ↗</Text>
            </Pressable>
          )}

          {paper.github && (
            <Pressable style={styles.actionBtnSecondary} onPress={openGithub}>
              <Text style={styles.actionBtnSecondaryText}>GitHub ↗</Text>
            </Pressable>
          )}
        </View>
      </Card>

      {/* Paper body */}
      <View style={styles.body}>
        {bodyLoading ? (
          <ActivityIndicator color={accent} style={{ marginTop: spacing['2xl'] }} />
        ) : bodyError ? (
          <Text style={styles.bodyError}>Failed to load paper body: {bodyError}</Text>
        ) : bodyContent ? (
          MarkdownRenderer ? (
            <MarkdownRenderer markdown={bodyContent} markdownStyle={darkMarkdownStyle} />
          ) : (
            <PlainTextFallback content={bodyContent} />
          )
        ) : null}
      </View>

      {/* Topics */}
      {paper.topics.length > 0 && (
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Topics</Text>
          <View style={styles.topicChips}>
            {paper.topics.map((t) => (
              <Pressable
                key={t}
                onPress={() => router.push(`/topic/${t}` as never)}
              >
                <GradientPill color={colors.textBody}>{t}</GradientPill>
              </Pressable>
            ))}
          </View>
        </View>
      )}

      {/* Related papers */}
      {related.length > 0 && (
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Related Papers</Text>
          {related.slice(0, 4).map((rp) => {
            const rpAccent = graph.categories[rp.category]?.color ?? colors.purple;
            return (
              <Card
                key={rp.id}
                padding="lg"
                style={styles.relatedItem}
                onPress={() => router.push(`/p/${rp.category}/${rp.slug}` as never)}
              >
                <View style={styles.catRow}>
                  <View style={[styles.dotSm, { backgroundColor: rpAccent }]} />
                  <Text style={[styles.relatedCat, { color: rpAccent }]}>
                    {graph.categories[rp.category]?.title ?? rp.category}
                  </Text>
                </View>
                <Text style={styles.relatedItemTitle} numberOfLines={2}>{rp.title}</Text>
              </Card>
            );
          })}
        </View>
      )}

      {/* Prev / Next navigation */}
      <View style={styles.navRow}>
        {prev && (
          <Card
            padding="lg"
            style={styles.navBtn}
            onPress={() => router.push(`/p/${prev.category}/${prev.slug}` as never)}
          >
            <Text style={styles.navBtnLabel}>← Older</Text>
            <Text style={styles.navBtnTitle} numberOfLines={1}>{prev.title}</Text>
          </Card>
        )}
        {next && (
          <Card
            padding="lg"
            style={[styles.navBtn, styles.navBtnRight] as never}
            onPress={() => router.push(`/p/${next.category}/${next.slug}` as never)}
          >
            <Text style={styles.navBtnLabel}>Newer →</Text>
            <Text style={styles.navBtnTitle} numberOfLines={1}>{next.title}</Text>
          </Card>
        )}
      </View>
    </ScrollView>
  );
}

function PlainTextFallback({ content }: { content: string }) {
  const lines = content.split('\n');
  return (
    <View style={{ gap: spacing.sm }}>
      {lines.map((line, i) => {
        if (line.startsWith('# ')) return <Text key={i} style={plain.h1}>{line.slice(2)}</Text>;
        if (line.startsWith('## ')) return <Text key={i} style={plain.h2}>{line.slice(3)}</Text>;
        if (line.startsWith('### ')) return <Text key={i} style={plain.h3}>{line.slice(4)}</Text>;
        if (line.startsWith('**') && line.endsWith('**'))
          return <Text key={i} style={plain.bold}>{line.slice(2, -2)}</Text>;
        if (line.startsWith('---')) return <View key={i} style={plain.divider} />;
        if (!line.trim()) return <View key={i} style={{ height: spacing.sm }} />;
        return <Text key={i} style={plain.body}>{line}</Text>;
      })}
    </View>
  );
}

const plain = StyleSheet.create({
  h1: { ...typography.h2, color: colors.white, marginTop: 8 },
  h2: { ...typography.h3, color: colors.textStrong, marginTop: 16 },
  h3: { fontSize: 16, fontWeight: '600', color: colors.textStrong, marginTop: 12 },
  bold: { fontSize: 15, fontWeight: '600', color: colors.white },
  body: { ...typography.body, color: colors.textBody },
  divider: { height: 1, backgroundColor: colors.border, marginVertical: 12 },
});

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: colors.pageBg,
  },
  content: {
    padding: spacing.lg,
    paddingBottom: 60,
    gap: spacing.lg,
  },
  headerCard: {
    gap: spacing.md,
  },
  catRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.sm,
  },
  dot: {
    width: 10,
    height: 10,
    borderRadius: 5,
  },
  dotSm: {
    width: 8,
    height: 8,
    borderRadius: 4,
  },
  catLabel: {
    ...typography.micro,
    textTransform: 'uppercase',
  },
  title: {
    ...typography.h1,
    color: colors.white,
  },
  pillRow: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: spacing.xs,
  },
  org: {
    ...typography.bodySm,
    color: colors.textBody,
  },
  actions: {
    flexDirection: 'row',
    gap: spacing.sm,
    flexWrap: 'wrap',
    marginTop: spacing.xs,
  },
  actionBtn: {
    borderRadius: radii.md,
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.md,
  },
  actionBtnDefault: {
    backgroundColor: colors.border,
  },
  actionBtnText: {
    ...typography.body,
    fontWeight: '600',
  },
  actionBtnSecondary: {
    backgroundColor: 'transparent',
    borderWidth: 1,
    borderColor: colors.border,
    borderRadius: radii.md,
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.md,
  },
  actionBtnSecondaryText: {
    ...typography.body,
    color: colors.textBody,
    fontWeight: '500',
  },
  body: {
    paddingHorizontal: spacing.xs,
  },
  bodyError: {
    ...typography.body,
    color: colors.pink,
  },
  section: {
    gap: spacing.sm,
  },
  sectionTitle: {
    ...typography.h3,
    color: colors.white,
  },
  topicChips: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: spacing.sm,
  },
  relatedItem: {
    gap: spacing.xs,
  },
  relatedCat: {
    ...typography.micro,
    textTransform: 'uppercase',
  },
  relatedItemTitle: {
    ...typography.body,
    color: colors.white,
  },
  navRow: {
    flexDirection: 'row',
    gap: spacing.sm,
  },
  navBtn: {
    flex: 1,
    gap: spacing.xs,
  },
  navBtnRight: {
    alignItems: 'flex-end',
  },
  navBtnLabel: {
    ...typography.micro,
    color: colors.textMuted,
    textTransform: 'uppercase',
  },
  navBtnTitle: {
    ...typography.bodySm,
    color: colors.white,
  },
  error: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: colors.pageBg,
  },
  errorText: {
    ...typography.body,
    color: colors.textMuted,
  },
});
