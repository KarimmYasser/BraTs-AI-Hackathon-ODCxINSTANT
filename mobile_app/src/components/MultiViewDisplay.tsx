/**
 * Multi-View Display Component
 * Shows axial, coronal, and sagittal views for 3D visualization
 */

import React from "react";
import { View, Text, Image, StyleSheet, Dimensions } from "react-native";
import Slider from "@react-native-community/slider";
import { colors, spacing, borderRadius, typography } from "../theme/colors";
import { MultiviewImages, VolumeDimensions } from "../types";

interface MultiViewDisplayProps {
  images: MultiviewImages;
  dimensions: VolumeDimensions;
  slices: {
    axial: number;
    coronal: number;
    sagittal: number;
  };
  showOverlay: boolean;
  onSliceChange: (
    view: "axial" | "coronal" | "sagittal",
    value: number,
  ) => void;
}

const { width } = Dimensions.get("window");
const panelWidth = (width - spacing.lg * 2 - spacing.md * 2) / 3;

export const MultiViewDisplay: React.FC<MultiViewDisplayProps> = ({
  images,
  dimensions,
  slices,
  showOverlay,
  onSliceChange,
}) => {
  const views = [
    {
      key: "axial" as const,
      title: "Axial",
      subtitle: "Top-Down",
      max: dimensions.width - 1,
      value: slices.axial,
      image: showOverlay ? images.axial : images.axial_no_overlay,
    },
    {
      key: "coronal" as const,
      title: "Coronal",
      subtitle: "Front-Back",
      max: dimensions.height - 1,
      value: slices.coronal,
      image: showOverlay ? images.coronal : images.coronal_no_overlay,
    },
    {
      key: "sagittal" as const,
      title: "Sagittal",
      subtitle: "Side",
      max: dimensions.depth - 1,
      value: slices.sagittal,
      image: showOverlay ? images.sagittal : images.sagittal_no_overlay,
    },
  ];

  return (
    <View style={styles.container}>
      <View style={styles.gridContainer}>
        {views.map((view) => (
          <View key={view.key} style={styles.panel}>
            <View style={styles.panelHeader}>
              <Text style={styles.panelTitle}>{view.title}</Text>
              <Text style={styles.panelSubtitle}>{view.subtitle}</Text>
            </View>

            <View style={styles.imageContainer}>
              <Image
                source={{ uri: `data:image/png;base64,${view.image}` }}
                style={styles.image}
                resizeMode="contain"
              />
            </View>

            <View style={styles.sliderContainer}>
              <Text style={styles.sliceLabel}>{view.value}</Text>
              <Slider
                style={styles.slider}
                minimumValue={0}
                maximumValue={view.max}
                value={view.value}
                onValueChange={(val) =>
                  onSliceChange(view.key, Math.round(val))
                }
                minimumTrackTintColor={colors.accentPrimary}
                maximumTrackTintColor={colors.bgTertiary}
                thumbTintColor={colors.accentPrimary}
              />
            </View>
          </View>
        ))}
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  gridContainer: {
    flexDirection: "row",
    flexWrap: "wrap",
    justifyContent: "space-between",
    gap: spacing.sm,
  },
  panel: {
    flex: 1,
    minWidth: panelWidth,
    maxWidth: "100%",
    backgroundColor: colors.bgTertiary,
    borderRadius: borderRadius.md,
    borderWidth: 1,
    borderColor: colors.borderColor,
    overflow: "hidden",
    marginBottom: spacing.sm,
  },
  panelHeader: {
    backgroundColor: colors.bgSecondary,
    paddingVertical: spacing.sm,
    paddingHorizontal: spacing.md,
    borderBottomWidth: 1,
    borderBottomColor: colors.borderColor,
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
  },
  panelTitle: {
    color: colors.accentPrimary,
    ...typography.body,
    fontWeight: "600",
  },
  panelSubtitle: {
    color: colors.textMuted,
    ...typography.small,
  },
  imageContainer: {
    aspectRatio: 1,
    backgroundColor: "#000",
    justifyContent: "center",
    alignItems: "center",
  },
  image: {
    width: "100%",
    height: "100%",
  },
  sliderContainer: {
    backgroundColor: colors.bgSecondary,
    paddingVertical: spacing.sm,
    paddingHorizontal: spacing.md,
    flexDirection: "row",
    alignItems: "center",
    gap: spacing.sm,
  },
  sliceLabel: {
    color: colors.textSecondary,
    ...typography.small,
    minWidth: 30,
    textAlign: "center",
  },
  slider: {
    flex: 1,
    height: 30,
  },
});

export default MultiViewDisplay;
