/**
 * File Upload Card Component
 */

import React from "react";
import { View, Text, StyleSheet, TouchableOpacity } from "react-native";
import { colors, spacing, borderRadius, typography } from "../theme/colors";

interface FileUploadCardProps {
  modality: string;
  label: string;
  fileName?: string;
  isSelected: boolean;
  onPress: () => void;
}

export const FileUploadCard: React.FC<FileUploadCardProps> = ({
  modality,
  label,
  fileName,
  isSelected,
  onPress,
}) => {
  return (
    <TouchableOpacity
      style={[styles.card, isSelected && styles.cardSelected]}
      onPress={onPress}
      activeOpacity={0.8}
    >
      <View style={styles.iconContainer}>
        <Text style={styles.iconText}>{modality}</Text>
      </View>
      <Text style={styles.label}>{label}</Text>
      <Text style={styles.fileName} numberOfLines={1}>
        {fileName || "Tap to select file"}
      </Text>
      {isSelected && <Text style={styles.checkmark}>✓</Text>}
    </TouchableOpacity>
  );
};

const styles = StyleSheet.create({
  card: {
    backgroundColor: colors.bgTertiary,
    borderWidth: 2,
    borderColor: colors.borderColor,
    borderStyle: "dashed",
    borderRadius: borderRadius.md,
    padding: spacing.lg,
    alignItems: "center",
    minWidth: 150,
    flex: 1,
  },
  cardSelected: {
    borderColor: colors.accentSecondary,
    borderStyle: "solid",
    shadowColor: colors.accentSecondary,
    shadowOffset: { width: 0, height: 0 },
    shadowOpacity: 0.3,
    shadowRadius: 10,
    elevation: 5,
  },
  iconContainer: {
    width: 60,
    height: 60,
    borderRadius: 30,
    backgroundColor: colors.accentPrimary,
    alignItems: "center",
    justifyContent: "center",
    marginBottom: spacing.md,
  },
  iconText: {
    color: colors.textPrimary,
    fontSize: 16,
    fontWeight: "700",
  },
  label: {
    color: colors.textPrimary,
    ...typography.body,
    fontWeight: "500",
    marginBottom: spacing.sm,
  },
  fileName: {
    color: colors.textMuted,
    ...typography.small,
    textAlign: "center",
  },
  checkmark: {
    position: "absolute",
    top: spacing.sm,
    right: spacing.sm,
    color: colors.success,
    fontSize: 18,
    fontWeight: "bold",
  },
});

export default FileUploadCard;
