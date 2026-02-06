/**
 * Slice Viewer Component
 */

import React from "react";
import { View, Text, Image, StyleSheet, ScrollView } from "react-native";
import { colors, spacing, borderRadius, typography } from "../theme/colors";
import { SliceImages } from "../types";

interface SliceViewerProps {
  images: SliceImages;
  showOverlay: boolean;
}

export const SliceViewer: React.FC<SliceViewerProps> = ({
  images,
  showOverlay,
}) => {
  const modalities = [
    { key: "t1", label: "T1" },
    { key: "t1ce", label: "T1ce" },
    { key: "t2", label: "T2" },
    { key: "flair", label: "FLAIR" },
  ];

  return (
    <ScrollView style={styles.container} showsVerticalScrollIndicator={false}>
      <View style={styles.grid}>
        {modalities.map(({ key, label }) => {
          const imageKey = showOverlay ? key : `${key}_no_overlay`;
          const imageData = images[imageKey as keyof SliceImages];

          return (
            <View key={key} style={styles.imageCard}>
              <View style={styles.imageTitle}>
                <Text style={styles.imageTitleText}>{label}</Text>
              </View>
              {imageData && (
                <Image
                  source={{ uri: `data:image/png;base64,${imageData}` }}
                  style={styles.image}
                  resizeMode="contain"
                />
              )}
            </View>
          );
        })}

        {images.segmentation && (
          <View style={[styles.imageCard, styles.segmentationCard]}>
            <View style={styles.imageTitle}>
              <Text style={styles.imageTitleText}>Segmentation</Text>
            </View>
            <Image
              source={{ uri: `data:image/png;base64,${images.segmentation}` }}
              style={styles.image}
              resizeMode="contain"
            />
          </View>
        )}
      </View>
    </ScrollView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  grid: {
    flexDirection: "row",
    flexWrap: "wrap",
    justifyContent: "space-between",
    paddingBottom: spacing.lg,
  },
  imageCard: {
    backgroundColor: colors.bgTertiary,
    borderWidth: 1,
    borderColor: colors.borderColor,
    borderRadius: borderRadius.md,
    overflow: "hidden",
    width: "48%",
    marginBottom: spacing.md,
  },
  segmentationCard: {
    width: "100%",
  },
  imageTitle: {
    backgroundColor: colors.bgSecondary,
    padding: spacing.sm,
    borderBottomWidth: 1,
    borderBottomColor: colors.borderColor,
  },
  imageTitleText: {
    color: colors.textPrimary,
    ...typography.caption,
    fontWeight: "600",
    textAlign: "center",
  },
  image: {
    width: "100%",
    aspectRatio: 1,
    backgroundColor: "#000",
  },
});

export default SliceViewer;
