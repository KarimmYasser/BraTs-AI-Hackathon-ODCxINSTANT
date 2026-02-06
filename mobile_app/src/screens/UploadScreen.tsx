/**
 * Upload Screen - Select 4 MRI modality files
 */

import React, { useState } from "react";
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  ActivityIndicator,
  Alert,
  ScrollView,
} from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import * as DocumentPicker from "expo-document-picker";
import { NativeStackScreenProps } from "@react-navigation/native-stack";
import { colors, spacing, borderRadius, typography } from "../theme/colors";
import { FileUploadCard } from "../components/FileUploadCard";
import { DeviceSelector } from "../components/DeviceSelector";
import apiClient from "../api/client";
import { RootStackParamList } from "../types";

type Props = NativeStackScreenProps<RootStackParamList, "Upload">;

interface FileInfo {
  uri: string;
  name: string;
}

interface Files {
  t1: FileInfo | null;
  t1ce: FileInfo | null;
  t2: FileInfo | null;
  flair: FileInfo | null;
}

export const UploadScreen: React.FC<Props> = ({ navigation }) => {
  const [files, setFiles] = useState<Files>({
    t1: null,
    t1ce: null,
    t2: null,
    flair: null,
  });
  const [loading, setLoading] = useState(false);

  const pickFile = async (modality: keyof Files) => {
    try {
      const result = await DocumentPicker.getDocumentAsync({
        type: "*/*",
        copyToCacheDirectory: true,
      });

      if (result.canceled || !result.assets || result.assets.length === 0) {
        return;
      }

      const file = result.assets[0];

      // Check if it's a NIfTI file
      if (!file.name.endsWith(".nii") && !file.name.endsWith(".nii.gz")) {
        Alert.alert(
          "Invalid File",
          "Please select a NIfTI file (.nii or .nii.gz)",
        );
        return;
      }

      setFiles((prev) => ({
        ...prev,
        [modality]: { uri: file.uri, name: file.name },
      }));
    } catch (error) {
      Alert.alert("Error", "Failed to pick file");
    }
  };

  const allFilesSelected = Object.values(files).every((f) => f !== null);

  const uploadFiles = async () => {
    if (!allFilesSelected) return;

    setLoading(true);
    try {
      const result = await apiClient.uploadFiles({
        t1: files.t1!,
        t1ce: files.t1ce!,
        t2: files.t2!,
        flair: files.flair!,
      });

      navigation.navigate("ModelSelect", {
        sessionId: result.sessionId,
        numSlices: result.numSlices,
      });
    } catch (error: any) {
      Alert.alert("Upload Failed", error.message || "Please try again");
    } finally {
      setLoading(false);
    }
  };

  const modalities = [
    { key: "t1" as const, label: "Native T1", short: "T1" },
    { key: "t1ce" as const, label: "Post-contrast T1", short: "T1ce" },
    { key: "t2" as const, label: "T2-weighted", short: "T2" },
    { key: "flair" as const, label: "T2 FLAIR", short: "FLAIR" },
  ];

  return (
    <SafeAreaView style={styles.container}>
      <ScrollView contentContainerStyle={styles.scrollContent}>
        {/* Header */}
        <View style={styles.header}>
          <Text style={styles.title}>Brain Tumor Segmentation</Text>
          <DeviceSelector />
        </View>

        {/* Upload Section */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Upload MRI Modalities</Text>
          <Text style={styles.sectionDescription}>
            Select the 4 required NIfTI (.nii or .nii.gz) files
          </Text>

          <View style={styles.uploadGrid}>
            {modalities.map(({ key, label, short }) => (
              <FileUploadCard
                key={key}
                modality={short}
                label={label}
                fileName={files[key]?.name}
                isSelected={files[key] !== null}
                onPress={() => pickFile(key)}
              />
            ))}
          </View>

          <TouchableOpacity
            style={[styles.button, !allFilesSelected && styles.buttonDisabled]}
            onPress={uploadFiles}
            disabled={!allFilesSelected || loading}
          >
            {loading ? (
              <ActivityIndicator color={colors.textPrimary} />
            ) : (
              <Text style={styles.buttonText}>Upload Files</Text>
            )}
          </TouchableOpacity>
        </View>
      </ScrollView>
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: colors.bgPrimary,
  },
  scrollContent: {
    padding: spacing.lg,
  },
  header: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: spacing.xl,
    flexWrap: "wrap",
    gap: spacing.md,
  },
  title: {
    color: colors.accentPrimary,
    ...typography.h1,
  },
  section: {
    backgroundColor: colors.bgCard,
    borderWidth: 1,
    borderColor: colors.borderColor,
    borderRadius: borderRadius.xl,
    padding: spacing.lg,
  },
  sectionTitle: {
    color: colors.textPrimary,
    ...typography.h2,
    marginBottom: spacing.sm,
  },
  sectionDescription: {
    color: colors.textSecondary,
    ...typography.body,
    marginBottom: spacing.lg,
  },
  uploadGrid: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: spacing.md,
    marginBottom: spacing.lg,
  },
  button: {
    backgroundColor: colors.accentPrimary,
    paddingVertical: spacing.md,
    paddingHorizontal: spacing.xl,
    borderRadius: borderRadius.md,
    alignItems: "center",
    shadowColor: colors.accentPrimary,
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 8,
    elevation: 5,
  },
  buttonDisabled: {
    opacity: 0.5,
  },
  buttonText: {
    color: colors.textPrimary,
    ...typography.body,
    fontWeight: "600",
  },
});

export default UploadScreen;
