import React, { useMemo } from 'react';
import { View, Text, SectionList, StyleSheet } from 'react-native';
import { getAvailableYears, getPapersForYear, graph } from '../../lib/graph';
import { getAllReadIds } from '../../lib/db';
import PaperCard from '../../components/PaperCard';
import type { Paper } from '../../lib/types';

interface Section {
  year: number;
  data: Paper[];
}

export default function TimelineScreen() {
  const readIds = useMemo(() => new Set(getAllReadIds()), []);

  const sections: Section[] = useMemo(() => {
    const years = getAvailableYears();
    return years.map((year) => ({
      year,
      data: getPapersForYear(year),
    }));
  }, []);

  const renderItem = ({ item }: { item: Paper }) => (
    <PaperCard
      paper={item}
      category={graph.categories[item.category]}
      isRead={readIds.has(item.id)}
    />
  );

  const renderSectionHeader = ({ section }: { section: Section }) => (
    <View style={styles.yearHeader}>
      <Text style={styles.yearText}>{section.year}</Text>
      <Text style={styles.yearCount}>{section.data.length} papers</Text>
    </View>
  );

  return (
    <SectionList
      sections={sections}
      keyExtractor={(item) => item.id}
      renderItem={renderItem}
      renderSectionHeader={renderSectionHeader}
      contentContainerStyle={styles.list}
      showsVerticalScrollIndicator={false}
      stickySectionHeadersEnabled
    />
  );
}

const styles = StyleSheet.create({
  list: {
    padding: 16,
    paddingBottom: 40,
  },
  yearHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    backgroundColor: '#0a0a0a',
    paddingVertical: 10,
    paddingHorizontal: 0,
    marginBottom: 4,
    borderBottomWidth: 1,
    borderBottomColor: '#2a2a2a',
  },
  yearText: {
    fontSize: 20,
    fontWeight: '700',
    color: '#aa99ff',
  },
  yearCount: {
    fontSize: 13,
    color: '#666666',
  },
});
