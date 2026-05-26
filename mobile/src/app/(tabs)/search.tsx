import React, { useState, useMemo, useCallback } from 'react';
import { View, Text, TextInput, FlatList, StyleSheet, ActivityIndicator } from 'react-native';
import { searchPapers } from '../../lib/search';
import { graph } from '../../lib/graph';
import { getAllReadIds } from '../../lib/db';
import PaperCard from '../../components/PaperCard';
import type { Paper } from '../../lib/types';

export default function SearchScreen() {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<Paper[]>([]);
  const [searching, setSearching] = useState(false);

  const readIds = useMemo(() => new Set(getAllReadIds()), []);

  const handleQuery = useCallback((text: string) => {
    setQuery(text);
    if (!text.trim()) {
      setResults([]);
      return;
    }
    setSearching(true);
    // Fuse is synchronous; wrap in setTimeout to not block keypress
    setTimeout(() => {
      const found = searchPapers(text, 40);
      setResults(found);
      setSearching(false);
    }, 0);
  }, []);

  const renderItem = useCallback(({ item }: { item: Paper }) => (
    <PaperCard
      paper={item}
      category={graph.categories[item.category]}
      isRead={readIds.has(item.id)}
    />
  ), [readIds]);

  return (
    <View style={styles.container}>
      <View style={styles.searchBox}>
        <TextInput
          style={styles.input}
          placeholder="Search papers, authors, topics..."
          placeholderTextColor="#666666"
          value={query}
          onChangeText={handleQuery}
          autoCorrect={false}
          autoCapitalize="none"
          clearButtonMode="while-editing"
        />
      </View>

      {searching && (
        <ActivityIndicator color="#aa99ff" style={{ marginTop: 20 }} />
      )}

      {!searching && query.trim() !== '' && results.length === 0 && (
        <View style={styles.empty}>
          <Text style={styles.emptyText}>No results for "{query}"</Text>
          <Text style={styles.emptyHint}>Try shorter terms or category names</Text>
        </View>
      )}

      {!searching && query.trim() === '' && (
        <View style={styles.hint}>
          <Text style={styles.hintText}>
            Search across {graph.counts.papers} papers by title, category, or organization
          </Text>
        </View>
      )}

      <FlatList
        data={results}
        keyExtractor={(item) => item.id}
        renderItem={renderItem}
        contentContainerStyle={styles.list}
        showsVerticalScrollIndicator={false}
        keyboardShouldPersistTaps="handled"
      />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#0a0a0a',
  },
  searchBox: {
    padding: 16,
    paddingBottom: 8,
  },
  input: {
    backgroundColor: '#1a1a1a',
    borderRadius: 10,
    padding: 12,
    fontSize: 15,
    color: '#ffffff',
    borderWidth: 1,
    borderColor: '#2a2a2a',
  },
  list: {
    padding: 16,
    paddingTop: 8,
    paddingBottom: 40,
  },
  empty: {
    padding: 24,
    alignItems: 'center',
    gap: 6,
  },
  emptyText: {
    fontSize: 15,
    color: '#ffffff',
    fontWeight: '600',
  },
  emptyHint: {
    fontSize: 13,
    color: '#666666',
  },
  hint: {
    paddingHorizontal: 20,
    paddingTop: 8,
    paddingBottom: 4,
  },
  hintText: {
    fontSize: 13,
    color: '#666666',
    textAlign: 'center',
  },
});
