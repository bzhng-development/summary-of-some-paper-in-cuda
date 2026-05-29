import React from 'react';
import { View, Pressable, StyleSheet, ViewStyle, AccessibilityRole } from 'react-native';
import { colors, radii, spacing } from '../../lib/theme';

interface Props {
  children: React.ReactNode;
  onPress?: () => void;
  style?: ViewStyle;
  padding?: keyof typeof spacing;
  accessibilityRole?: AccessibilityRole;
  accessibilityLabel?: string;
}

// Mirrors paper-graph-ui's "rounded-2xl + GradientBorder" card style.
// We can't paint a true CSS gradient border in RN cheaply; instead we
// stack a darker hairline border on the lifted card bg, which reads as
// a subtle outline at typical iPhone densities.
export default function Card({
  children,
  onPress,
  style,
  padding = 'xl',
  accessibilityRole,
  accessibilityLabel,
}: Props) {
  const content = (
    <View style={[styles.card, { padding: spacing[padding] }, style]}>{children}</View>
  );

  if (onPress) {
    return (
      <Pressable
        onPress={onPress}
        style={({ pressed }) => ({ opacity: pressed ? 0.85 : 1 })}
        accessibilityRole={accessibilityRole ?? 'button'}
        accessibilityLabel={accessibilityLabel}
      >
        {content}
      </Pressable>
    );
  }
  return content;
}

const styles = StyleSheet.create({
  card: {
    backgroundColor: colors.cardBg,
    borderRadius: radii.lg,
    borderWidth: 1,
    borderColor: colors.border,
    overflow: 'hidden',
  },
});
