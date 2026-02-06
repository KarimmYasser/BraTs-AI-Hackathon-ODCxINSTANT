/**
 * Model Selection Card Component
 */

import React from "react";
import { View, Text, StyleSheet, TouchableOpacity } from "react-native";
import { colors, spacing, borderRadius, typography } from "../theme/colors";

interface ModelCardProps {
  id: string;
  name: string;
  description: string;
  isSelected: boolean;
  onPress: () => void;
}

export const ModelCard: React.FC<ModelCardProps> = ({
  id,
  name,
  description,
  isSelected,
  onPress,
}) => {
  const iconLetter = id === "unet" ? "U" : "S";

  return (
    <TouchableOpacity
      style={[styles.card, isSelected && styles.cardSelected]}
      onPress={onPress}
      activeOpacity={0.8}
    >
      <View style={styles.iconContainer}>
        <Text style={styles.iconText}>{iconLetter}</Text>
      </View>
      <View style={styles.info}>
        <Text style={styles.name}>{name}</Text>
        <Text style={styles.description}>{description}</Text>
      </View>
      <View style={[styles.checkmark, isSelected && styles.checkmarkSelected]}>
        <Text style={styles.checkmarkText}>✓</Text>
      </View>
    </TouchableOpacity>
  );
};

const styles = StyleSheet.create({
  card: {
    flexDirection: "row",
    alignItems: "center",
    backgroundColor: colors.bgTertiary,
    borderWidth: 2,
    borderColor: colors.borderColor,
    borderRadius: borderRadius.md,
    padding: spacing.lg,
    marginBottom: spacing.md,
  },
  cardSelected: {
    borderColor: colors.accentSecondary,
    shadowColor: colors.accentSecondary,
    shadowOffset: { width: 0, height: 0 },
    shadowOpacity: 0.3,
    shadowRadius: 10,
    elevation: 5,
  },
  iconContainer: {
    width: 50,
    height: 50,
    borderRadius: borderRadius.sm,
    backgroundColor: colors.accentPrimary,
    alignItems: "center",
    justifyContent: "center",
    marginRight: spacing.lg,
  },
  iconText: {
    color: colors.textPrimary,
    fontSize: 20,
    fontWeight: "700",
  },
  info: {
    flex: 1,
  },
  name: {
    color: colors.textPrimary,
    ...typography.h3,
    marginBottom: spacing.xs,
  },
  description: {
    color: colors.textSecondary,
    ...typography.caption,
  },
  checkmark: {
    width: 28,
    height: 28,
    borderRadius: 14,
    backgroundColor: colors.bgTertiary,
    borderWidth: 2,
    borderColor: colors.borderColor,
    alignItems: "center",
    justifyContent: "center",
    opacity: 0,
  },
  checkmarkSelected: {
    opacity: 1,
    backgroundColor: colors.accentPrimary,
    borderColor: colors.accentPrimary,
  },
  checkmarkText: {
    color: colors.textPrimary,
    fontSize: 14,
    fontWeight: "bold",
  },
});

export default ModelCard;
