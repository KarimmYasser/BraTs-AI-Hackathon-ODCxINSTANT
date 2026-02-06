/**
 * Device Selector Component - Switch between CPU and CUDA
 */

import React, { useState, useEffect } from "react";
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  ActivityIndicator,
  Alert,
} from "react-native";
import { colors, spacing, borderRadius, typography } from "../theme/colors";
import apiClient from "../api/client";
import { DeviceInfo } from "../types";

interface DeviceSelectorProps {
  onDeviceChange?: (device: string) => void;
}

export const DeviceSelector: React.FC<DeviceSelectorProps> = ({
  onDeviceChange,
}) => {
  const [devices, setDevices] = useState<DeviceInfo[]>([]);
  const [currentDevice, setCurrentDevice] = useState<string>("");
  const [loading, setLoading] = useState(false);
  const [initialLoading, setInitialLoading] = useState(true);

  useEffect(() => {
    loadDevices();
  }, []);

  const loadDevices = async () => {
    try {
      const response = await apiClient.getDevices();
      setDevices(response.devices);
      setCurrentDevice(response.current_device);
    } catch (error) {
      console.error("Failed to load devices:", error);
    } finally {
      setInitialLoading(false);
    }
  };

  const handleDeviceChange = async (deviceId: string) => {
    if (deviceId === currentDevice || loading) return;

    const device = devices.find((d) => d.id === deviceId);
    if (!device?.available) {
      Alert.alert(
        "Device Unavailable",
        `${device?.name || deviceId} is not available on this server.`,
      );
      return;
    }

    setLoading(true);
    try {
      const response = await apiClient.setDevice(deviceId);
      setCurrentDevice(response.device);
      onDeviceChange?.(response.device);
    } catch (error: any) {
      Alert.alert("Error", error.message || "Failed to change device");
    } finally {
      setLoading(false);
    }
  };

  const getDeviceIcon = (deviceId: string) => {
    return deviceId === "cuda" ? "⚡" : "💻";
  };

  const isSelected = (deviceId: string) => {
    return currentDevice.includes(deviceId);
  };

  if (initialLoading) {
    return (
      <View style={styles.container}>
        <ActivityIndicator size="small" color={colors.accentPrimary} />
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <Text style={styles.label}>Device:</Text>
      <View style={styles.buttonGroup}>
        {devices.map((device) => (
          <TouchableOpacity
            key={device.id}
            style={[
              styles.deviceButton,
              isSelected(device.id) && styles.deviceButtonSelected,
              !device.available && styles.deviceButtonDisabled,
            ]}
            onPress={() => handleDeviceChange(device.id)}
            disabled={!device.available || loading}
          >
            <Text style={styles.deviceIcon}>{getDeviceIcon(device.id)}</Text>
            <Text
              style={[
                styles.deviceText,
                isSelected(device.id) && styles.deviceTextSelected,
                !device.available && styles.deviceTextDisabled,
              ]}
            >
              {device.id.toUpperCase()}
            </Text>
          </TouchableOpacity>
        ))}
      </View>
      {loading && (
        <ActivityIndicator
          size="small"
          color={colors.accentPrimary}
          style={styles.loader}
        />
      )}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flexDirection: "row",
    alignItems: "center",
    backgroundColor: colors.bgTertiary,
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.sm,
    borderRadius: borderRadius.sm,
    borderWidth: 1,
    borderColor: colors.borderColor,
    gap: spacing.sm,
  },
  label: {
    color: colors.textSecondary,
    ...typography.small,
    fontWeight: "500",
  },
  buttonGroup: {
    flexDirection: "row",
    gap: spacing.xs,
  },
  deviceButton: {
    flexDirection: "row",
    alignItems: "center",
    paddingHorizontal: spacing.sm,
    paddingVertical: spacing.xs,
    borderRadius: borderRadius.sm,
    backgroundColor: colors.bgSecondary,
    borderWidth: 1,
    borderColor: colors.borderColor,
    gap: 4,
  },
  deviceButtonSelected: {
    backgroundColor: colors.accentPrimary,
    borderColor: colors.accentPrimary,
  },
  deviceButtonDisabled: {
    opacity: 0.4,
  },
  deviceIcon: {
    fontSize: 12,
  },
  deviceText: {
    color: colors.textSecondary,
    ...typography.small,
    fontWeight: "500",
  },
  deviceTextSelected: {
    color: colors.textPrimary,
  },
  deviceTextDisabled: {
    color: colors.textMuted,
  },
  loader: {
    marginLeft: spacing.xs,
  },
});

export default DeviceSelector;
