import React from 'react';
import { View, Text, StyleSheet, ViewStyle } from 'react-native';
import { colors, radii, typography } from '../../lib/theme';

interface Props {
  children: React.ReactNode;
  // Either pick a theme color, or pass a custom hex (e.g. category color)
  color?: string;
  style?: ViewStyle;
}

// Mirrors paper-graph-ui's GradientLabel: fully-rounded pill with a
// semi-transparent tinted background and a matching tinted text color.
// Web does this with CSS gradient layers; we approximate with two flat
// alpha-blended layers (background + color), which is visually equivalent
// at typical pill sizes.
export default function GradientPill({ children, color = colors.green, style }: Props) {
  return (
    <View style={[styles.wrap, { backgroundColor: color + '1A', borderColor: color + '40' }, style]}>
      <Text style={[styles.text, { color }]} numberOfLines={1}>
        {children}
      </Text>
    </View>
  );
}

const styles = StyleSheet.create({
  wrap: {
    alignSelf: 'flex-start',
    borderRadius: radii.pill,
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderWidth: 1,
  },
  text: {
    ...typography.caption,
    fontWeight: '600',
  },
});
