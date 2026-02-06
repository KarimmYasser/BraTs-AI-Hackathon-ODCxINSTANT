/**
 * Statistics Card Component
 */

import React from "react";
import { View, Text, StyleSheet } from "react-native";
import { colors, spacing, borderRadius, typography } from "../theme/colors";

interface StatCardProps {
  value: number;
  label: string;
  color?: string;
}

export const StatCard: React.FC<StatCardProps> = ({ value, label, color }) => {
  return (
    <View style={styles.card}>
      <Text style={[styles.value, color ? { color } : null]}>
        {value.toLocaleString()}
      </Text>
      <Text style={styles.label}>{label}</Text>
    </View>
  );
};

const styles = StyleSheet.create({
  card: {
    backgroundColor: colors.bgTertiary,
    borderWidth: 1,
    borderColor: colors.borderColor,
    borderRadius: borderRadius.md,
    padding: spacing.lg,
    alignItems: "center",
    flex: 1,
    marginHorizontal: spacing.xs,
  },
  value: {
    color: colors.accentPrimary,
    fontSize: 24,
    fontWeight: "700",
    marginBottom: spacing.xs,
  },
  label: {
    color: colors.textSecondary,
    ...typography.small,
    textAlign: "center",
  },
});

export default StatCard;
