/**
 * API client for FastAPI backend
 */

import Constants from "expo-constants";
import {
  SessionData,
  SegmentationResult,
  SliceData,
  ModelInfo,
  HealthCheck,
  MultiviewData,
  DevicesResponse,
} from "../types";

// Get API URL from environment or use default
const API_BASE_URL =
  Constants.expoConfig?.extra?.apiBaseUrl || "http://localhost:8000";

class ApiClient {
  private baseUrl: string;

  constructor(baseUrl: string = API_BASE_URL) {
    this.baseUrl = baseUrl;
  }

  setBaseUrl(url: string) {
    this.baseUrl = url;
  }

  async healthCheck(): Promise<HealthCheck> {
    const response = await fetch(`${this.baseUrl}/health`);
    if (!response.ok) throw new Error("Health check failed");
    return response.json();
  }

  async getModels(): Promise<{ models: ModelInfo[] }> {
    const response = await fetch(`${this.baseUrl}/models`);
    if (!response.ok) throw new Error("Failed to fetch models");
    return response.json();
  }

  async uploadFiles(files: {
    t1: { uri: string; name: string };
    t1ce: { uri: string; name: string };
    t2: { uri: string; name: string };
    flair: { uri: string; name: string };
  }): Promise<SessionData> {
    const formData = new FormData();

    // Add each file to the form data
    formData.append("t1", {
      uri: files.t1.uri,
      type: "application/octet-stream",
      name: files.t1.name,
    } as any);

    formData.append("t1ce", {
      uri: files.t1ce.uri,
      type: "application/octet-stream",
      name: files.t1ce.name,
    } as any);

    formData.append("t2", {
      uri: files.t2.uri,
      type: "application/octet-stream",
      name: files.t2.name,
    } as any);

    formData.append("flair", {
      uri: files.flair.uri,
      type: "application/octet-stream",
      name: files.flair.name,
    } as any);

    const response = await fetch(`${this.baseUrl}/upload`, {
      method: "POST",
      body: formData,
      headers: {
        "Content-Type": "multipart/form-data",
      },
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || "Upload failed");
    }

    const data = await response.json();
    return {
      sessionId: data.session_id,
      numSlices: data.num_slices,
    };
  }

  async runSegmentation(
    sessionId: string,
    modelName: string,
  ): Promise<SegmentationResult> {
    const formData = new FormData();
    formData.append("model_name", modelName);

    const response = await fetch(`${this.baseUrl}/segment/${sessionId}`, {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || "Segmentation failed");
    }

    const data = await response.json();
    return {
      sessionId: data.session_id,
      modelUsed: data.model_used,
      tumorStatistics: data.tumor_statistics,
      numSlices: data.num_slices,
    };
  }

  async getSlice(
    sessionId: string,
    sliceIdx: number,
    overlay: boolean = true,
  ): Promise<SliceData> {
    const response = await fetch(
      `${this.baseUrl}/slice/${sessionId}/${sliceIdx}?overlay=${overlay}`,
    );

    if (!response.ok) {
      throw new Error("Failed to load slice");
    }

    const data = await response.json();
    return {
      sessionId: data.session_id,
      sliceIdx: data.slice_idx,
      numSlices: data.num_slices,
      images: data.images,
      hasSegmentation: data.has_segmentation,
    };
  }

  async getMultiview(
    sessionId: string,
    axial: number,
    coronal: number,
    sagittal: number,
    modality: string = "flair",
    overlay: boolean = true,
  ): Promise<MultiviewData> {
    const url =
      `${this.baseUrl}/multiview/${sessionId}?` +
      `axial=${axial}&coronal=${coronal}&sagittal=${sagittal}&` +
      `modality=${modality}&overlay=${overlay}`;

    const response = await fetch(url);

    if (!response.ok) {
      throw new Error("Failed to load multi-view");
    }

    const data = await response.json();
    return {
      sessionId: data.session_id,
      dimensions: data.dimensions,
      currentSlices: data.current_slices,
      modality: data.modality,
      images: data.images,
      hasSegmentation: data.has_segmentation,
    };
  }

  async getDevices(): Promise<DevicesResponse> {
    const response = await fetch(`${this.baseUrl}/devices`);
    if (!response.ok) throw new Error("Failed to fetch devices");
    return response.json();
  }

  async setDevice(
    deviceName: string,
  ): Promise<{ message: string; device: string }> {
    const formData = new FormData();
    formData.append("device_name", deviceName);

    const response = await fetch(`${this.baseUrl}/device`, {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || "Failed to change device");
    }

    return response.json();
  }

  async deleteSession(sessionId: string): Promise<void> {
    await fetch(`${this.baseUrl}/session/${sessionId}`, {
      method: "DELETE",
    });
  }
}

export const apiClient = new ApiClient();
export default apiClient;
