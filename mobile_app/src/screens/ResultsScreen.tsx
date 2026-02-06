/**
 * Results Screen - View segmentation results with slice navigation
 * Supports both 2D slice view and 3D multi-view modes
 */

import React, { useState, useEffect, useCallback } from "react";
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  Switch,
  ActivityIndicator,
  ScrollView,
} from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import { NativeStackScreenProps } from "@react-navigation/native-stack";
import Slider from "@react-native-community/slider";
import { colors, spacing, borderRadius, typography } from "../theme/colors";
import { StatCard } from "../components/StatCard";
import { SliceViewer } from "../components/SliceViewer";
import { MultiViewDisplay } from "../components/MultiViewDisplay";
import apiClient from "../api/client";
import {
  RootStackParamList,
  SliceImages,
  MultiviewImages,
  VolumeDimensions,
} from "../types";

type Props = NativeStackScreenProps<RootStackParamList, "Results">;

export const ResultsScreen: React.FC<Props> = ({ navigation, route }) => {
  const { sessionId, numSlices, tumorStats } = route.params;

  // View mode state
  const [viewMode, setViewMode] = useState<"2d" | "3d">("2d");

  // 2D slice state
  const [currentSlice, setCurrentSlice] = useState(Math.floor(numSlices / 2));
  const [showOverlay, setShowOverlay] = useState(true);
  const [images, setImages] = useState<SliceImages | null>(null);
  const [loading, setLoading] = useState(true);

  // 3D multi-view state
  const [multiviewImages, setMultiviewImages] =
    useState<MultiviewImages | null>(null);
  const [multiviewDimensions, setMultiviewDimensions] =
    useState<VolumeDimensions | null>(null);
  const [multiviewSlices, setMultiviewSlices] = useState({
    axial: 64,
    coronal: 64,
    sagittal: 64,
  });
  const [multiviewLoading, setMultiviewLoading] = useState(false);

  const loadSlice = useCallback(
    async (sliceIdx: number) => {
      try {
        const data = await apiClient.getSlice(sessionId, sliceIdx, showOverlay);
        setImages(data.images);
      } catch (error) {
        console.error("Failed to load slice:", error);
      } finally {
        setLoading(false);
      }
    },
    [sessionId, showOverlay],
  );

  const loadMultiview = useCallback(async () => {
    setMultiviewLoading(true);
    try {
      const data = await apiClient.getMultiview(
        sessionId,
        multiviewSlices.axial,
        multiviewSlices.coronal,
        multiviewSlices.sagittal,
        "flair",
        showOverlay,
      );
      setMultiviewImages(data.images);
      if (!multiviewDimensions) {
        setMultiviewDimensions(data.dimensions);
        // Initialize slices to center
        setMultiviewSlices({
          axial: Math.floor(data.dimensions.width / 2),
          coronal: Math.floor(data.dimensions.height / 2),
          sagittal: Math.floor(data.dimensions.depth / 2),
        });
      }
    } catch (error) {
      console.error("Failed to load multi-view:", error);
    } finally {
      setMultiviewLoading(false);
    }
  }, [sessionId, multiviewSlices, showOverlay, multiviewDimensions]);

  useEffect(() => {
    if (viewMode === "2d") {
      loadSlice(currentSlice);
    } else {
      loadMultiview();
    }
  }, [viewMode, currentSlice, showOverlay, loadSlice, loadMultiview]);

  const handleSliceChange = (value: number) => {
    const newSlice = Math.round(value);
    if (newSlice !== currentSlice) {
      setCurrentSlice(newSlice);
    }
  };

  const handleMultiviewSliceChange = (
    view: "axial" | "coronal" | "sagittal",
    value: number,
  ) => {
    setMultiviewSlices((prev) => ({ ...prev, [view]: value }));
  };

  const handleNewScan = async () => {
    try {
      await apiClient.deleteSession(sessionId);
    } catch (error) {
      // Ignore cleanup errors
    }
    navigation.popToTop();
  };

  return (
    <SafeAreaView style={styles.container}>
      <ScrollView
        style={styles.scrollView}
        contentContainerStyle={styles.scrollContent}
      >
        {/* Header */}
        <View style={styles.header}>
          <Text style={styles.title}>Segmentation Results</Text>

          {/* Legend */}
          <View style={styles.legend}>
            <View style={styles.legendItem}>
              <View
                style={[
                  styles.legendColor,
                  { backgroundColor: colors.ncrColor },
                ]}
              />
              <Text style={styles.legendText}>NCR</Text>
            </View>
            <View style={styles.legendItem}>
              <View
                style={[
                  styles.legendColor,
                  { backgroundColor: colors.edColor },
                ]}
              />
              <Text style={styles.legendText}>ED</Text>
            </View>
            <View style={styles.legendItem}>
              <View
                style={[
                  styles.legendColor,
                  { backgroundColor: colors.etColor },
                ]}
              />
              <Text style={styles.legendText}>ET</Text>
            </View>
          </View>
        </View>

        {/* Statistics */}
        <View style={styles.statsRow}>
          <StatCard
            value={tumorStats.ncr}
            label="NCR Voxels"
            color={colors.ncrColor}
          />
          <StatCard
            value={tumorStats.ed}
            label="ED Voxels"
            color={colors.edColor}
          />
          <StatCard
            value={tumorStats.et}
            label="ET Voxels"
            color={colors.etColor}
          />
        </View>

        {/* View Mode Toggle */}
        <View style={styles.viewModeContainer}>
          <TouchableOpacity
            style={[
              styles.viewModeButton,
              viewMode === "2d" && styles.viewModeButtonActive,
            ]}
            onPress={() => setViewMode("2d")}
          >
            <Text
              style={[
                styles.viewModeButtonText,
                viewMode === "2d" && styles.viewModeButtonTextActive,
              ]}
            >
              2D Slice
            </Text>
          </TouchableOpacity>
          <TouchableOpacity
            style={[
              styles.viewModeButton,
              viewMode === "3d" && styles.viewModeButtonActive,
            ]}
            onPress={() => setViewMode("3d")}
          >
            <Text
              style={[
                styles.viewModeButtonText,
                viewMode === "3d" && styles.viewModeButtonTextActive,
              ]}
            >
              3D Multi-View
            </Text>
          </TouchableOpacity>
        </View>

        {/* Overlay Toggle */}
        <View style={styles.toggleRow}>
          <Text style={styles.toggleLabel}>Show Overlay</Text>
          <Switch
            value={showOverlay}
            onValueChange={setShowOverlay}
            trackColor={{
              false: colors.bgTertiary,
              true: colors.accentPrimary,
            }}
            thumbColor={colors.textPrimary}
          />
        </View>

        {/* 2D Slice Viewer */}
        {viewMode === "2d" && (
          <>
            {/* Slice Controls */}
            <View style={styles.sliceControls}>
              <TouchableOpacity
                style={styles.sliceButton}
                onPress={() => setCurrentSlice(Math.max(0, currentSlice - 1))}
              >
                <Text style={styles.sliceButtonText}>◀</Text>
              </TouchableOpacity>

              <View style={styles.sliderContainer}>
                <Slider
                  style={styles.slider}
                  minimumValue={0}
                  maximumValue={numSlices - 1}
                  value={currentSlice}
                  onValueChange={handleSliceChange}
                  minimumTrackTintColor={colors.accentPrimary}
                  maximumTrackTintColor={colors.bgTertiary}
                  thumbTintColor={colors.accentPrimary}
                />
              </View>

              <TouchableOpacity
                style={styles.sliceButton}
                onPress={() =>
                  setCurrentSlice(Math.min(numSlices - 1, currentSlice + 1))
                }
              >
                <Text style={styles.sliceButtonText}>▶</Text>
              </TouchableOpacity>

              <Text style={styles.sliceInfo}>
                {currentSlice} / {numSlices - 1}
              </Text>
            </View>

            {/* Slice Viewer */}
            <View style={styles.viewerContainer}>
              {loading ? (
                <ActivityIndicator size="large" color={colors.accentPrimary} />
              ) : images ? (
                <SliceViewer images={images} showOverlay={showOverlay} />
              ) : null}
            </View>
          </>
        )}

        {/* 3D Multi-View Display */}
        {viewMode === "3d" && (
          <View style={styles.multiviewContainer}>
            {multiviewLoading ? (
              <ActivityIndicator size="large" color={colors.accentPrimary} />
            ) : multiviewImages && multiviewDimensions ? (
              <MultiViewDisplay
                images={multiviewImages}
                dimensions={multiviewDimensions}
                slices={multiviewSlices}
                showOverlay={showOverlay}
                onSliceChange={handleMultiviewSliceChange}
              />
            ) : null}
          </View>
        )}

        {/* New Scan Button */}
        <TouchableOpacity style={styles.newScanButton} onPress={handleNewScan}>
          <Text style={styles.newScanButtonText}>Analyze New Scan</Text>
        </TouchableOpacity>
      </ScrollView>
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: colors.bgPrimary,
  },
  scrollView: {
    flex: 1,
  },
  scrollContent: {
    padding: spacing.lg,
  },
  header: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: spacing.lg,
    flexWrap: "wrap",
    gap: spacing.sm,
  },
  title: {
    color: colors.textPrimary,
    ...typography.h2,
  },
  legend: {
    flexDirection: "row",
    gap: spacing.md,
  },
  legendItem: {
    flexDirection: "row",
    alignItems: "center",
    gap: spacing.xs,
  },
  legendColor: {
    width: 12,
    height: 12,
    borderRadius: 3,
  },
  legendText: {
    color: colors.textSecondary,
    ...typography.small,
  },
  statsRow: {
    flexDirection: "row",
    marginBottom: spacing.lg,
  },
  viewModeContainer: {
    flexDirection: "row",
    backgroundColor: colors.bgTertiary,
    borderRadius: borderRadius.md,
    padding: spacing.xs,
    marginBottom: spacing.md,
    borderWidth: 1,
    borderColor: colors.borderColor,
  },
  viewModeButton: {
    flex: 1,
    paddingVertical: spacing.sm,
    alignItems: "center",
    borderRadius: borderRadius.sm,
  },
  viewModeButtonActive: {
    backgroundColor: colors.accentPrimary,
  },
  viewModeButtonText: {
    color: colors.textSecondary,
    ...typography.body,
    fontWeight: "500",
  },
  viewModeButtonTextActive: {
    color: colors.textPrimary,
  },
  toggleRow: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    backgroundColor: colors.bgTertiary,
    padding: spacing.md,
    borderRadius: borderRadius.md,
    marginBottom: spacing.md,
  },
  toggleLabel: {
    color: colors.textPrimary,
    ...typography.body,
  },
  sliceControls: {
    flexDirection: "row",
    alignItems: "center",
    backgroundColor: colors.bgTertiary,
    padding: spacing.md,
    borderRadius: borderRadius.md,
    marginBottom: spacing.md,
  },
  sliceButton: {
    width: 40,
    height: 40,
    backgroundColor: colors.bgSecondary,
    borderRadius: borderRadius.sm,
    alignItems: "center",
    justifyContent: "center",
    borderWidth: 1,
    borderColor: colors.borderColor,
  },
  sliceButtonText: {
    color: colors.textPrimary,
    fontSize: 16,
  },
  sliderContainer: {
    flex: 1,
    marginHorizontal: spacing.md,
  },
  slider: {
    width: "100%",
    height: 40,
  },
  sliceInfo: {
    color: colors.textSecondary,
    ...typography.caption,
    minWidth: 60,
    textAlign: "right",
  },
  viewerContainer: {
    minHeight: 300,
    marginBottom: spacing.md,
  },
  multiviewContainer: {
    minHeight: 400,
    marginBottom: spacing.md,
  },
  newScanButton: {
    backgroundColor: colors.bgTertiary,
    paddingVertical: spacing.md,
    paddingHorizontal: spacing.xl,
    borderRadius: borderRadius.md,
    alignItems: "center",
    borderWidth: 1,
    borderColor: colors.borderColor,
  },
  newScanButtonText: {
    color: colors.textPrimary,
    ...typography.body,
    fontWeight: "600",
  },
});

export default ResultsScreen;
