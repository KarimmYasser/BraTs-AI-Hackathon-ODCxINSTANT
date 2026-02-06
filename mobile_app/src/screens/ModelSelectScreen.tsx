/**
 * Model Selection Screen
 */

import React, { useState } from "react";
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  ActivityIndicator,
  Alert,
} from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import { NativeStackScreenProps } from "@react-navigation/native-stack";
import { colors, spacing, borderRadius, typography } from "../theme/colors";
import { ModelCard } from "../components/ModelCard";
import apiClient from "../api/client";
import { RootStackParamList } from "../types";

type Props = NativeStackScreenProps<RootStackParamList, "ModelSelect">;

const MODELS = [
  {
    id: "unet",
    name: "U-Net 3D",
    description: "3D U-Net architecture for volumetric segmentation",
  },
  {
    id: "segresnet",
    name: "SegResNet",
    description: "Residual encoder-decoder network for segmentation",
  },
  {
    id: "mednext",
    name: "MedNeXt",
    description: "Transformer-inspired ConvNet with sliding window inference",
  },
];

export const ModelSelectScreen: React.FC<Props> = ({ navigation, route }) => {
  const { sessionId, numSlices } = route.params;
  const [selectedModel, setSelectedModel] = useState("unet");
  const [loading, setLoading] = useState(false);

  const runSegmentation = async () => {
    setLoading(true);
    try {
      const result = await apiClient.runSegmentation(sessionId, selectedModel);

      navigation.navigate("Results", {
        sessionId: result.sessionId,
        numSlices: result.numSlices,
        tumorStats: result.tumorStatistics,
      });
    } catch (error: any) {
      Alert.alert("Segmentation Failed", error.message || "Please try again");
    } finally {
      setLoading(false);
    }
  };

  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.content}>
        {/* Header */}
        <View style={styles.header}>
          <Text style={styles.title}>Select Model</Text>
          <Text style={styles.subtitle}>
            Choose a segmentation model for analysis
          </Text>
        </View>

        {/* Model Cards */}
        <View style={styles.modelList}>
          {MODELS.map((model) => (
            <ModelCard
              key={model.id}
              id={model.id}
              name={model.name}
              description={model.description}
              isSelected={selectedModel === model.id}
              onPress={() => setSelectedModel(model.id)}
            />
          ))}
        </View>

        {/* Run Button */}
        <TouchableOpacity
          style={styles.button}
          onPress={runSegmentation}
          disabled={loading}
        >
          {loading ? (
            <>
              <ActivityIndicator
                color={colors.textPrimary}
                style={styles.buttonLoader}
              />
              <Text style={styles.buttonText}>Processing...</Text>
            </>
          ) : (
            <Text style={styles.buttonText}>Run Segmentation</Text>
          )}
        </TouchableOpacity>

        {loading && (
          <Text style={styles.loadingHint}>This may take a moment...</Text>
        )}
      </View>
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: colors.bgPrimary,
  },
  content: {
    flex: 1,
    padding: spacing.lg,
  },
  header: {
    marginBottom: spacing.xl,
  },
  title: {
    color: colors.textPrimary,
    ...typography.h1,
    marginBottom: spacing.sm,
  },
  subtitle: {
    color: colors.textSecondary,
    ...typography.body,
  },
  modelList: {
    flex: 1,
  },
  button: {
    backgroundColor: colors.accentPrimary,
    paddingVertical: spacing.md,
    paddingHorizontal: spacing.xl,
    borderRadius: borderRadius.md,
    alignItems: "center",
    flexDirection: "row",
    justifyContent: "center",
    shadowColor: colors.accentPrimary,
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 8,
    elevation: 5,
    marginTop: spacing.lg,
  },
  buttonLoader: {
    marginRight: spacing.sm,
  },
  buttonText: {
    color: colors.textPrimary,
    ...typography.body,
    fontWeight: "600",
  },
  loadingHint: {
    color: colors.textMuted,
    ...typography.caption,
    textAlign: "center",
    marginTop: spacing.md,
  },
});

export default ModelSelectScreen;
