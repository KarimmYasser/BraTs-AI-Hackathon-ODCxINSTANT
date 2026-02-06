/**
 * TypeScript types for the application
 */

export interface SessionData {
  sessionId: string;
  numSlices: number;
}

export interface TumorStatistics {
  ncr: number;
  ed: number;
  et: number;
}

export interface SegmentationResult {
  sessionId: string;
  modelUsed: string;
  tumorStatistics: TumorStatistics;
  numSlices: number;
}

export interface SliceImages {
  t1: string;
  t1ce: string;
  t2: string;
  flair: string;
  segmentation?: string;
  t1_no_overlay: string;
  t1ce_no_overlay: string;
  t2_no_overlay: string;
  flair_no_overlay: string;
}

export interface SliceData {
  sessionId: string;
  sliceIdx: number;
  numSlices: number;
  images: SliceImages;
  hasSegmentation: boolean;
}

export interface ModelInfo {
  id: string;
  name: string;
  description: string;
  available: boolean;
}

export interface DeviceInfo {
  id: string;
  name: string;
  available: boolean;
}

export interface DevicesResponse {
  devices: DeviceInfo[];
  current_device: string;
}

export interface HealthCheck {
  status: string;
  device: string;
  cuda_available: boolean;
  modelsAvailable: {
    unet: boolean;
    segresnet: boolean;
    mednext: boolean;
  };
}

// Multi-view types for 3D visualization
export interface VolumeDimensions {
  depth: number;
  height: number;
  width: number;
}

export interface MultiviewImages {
  axial: string;
  axial_no_overlay: string;
  coronal: string;
  coronal_no_overlay: string;
  sagittal: string;
  sagittal_no_overlay: string;
}

export interface MultiviewData {
  sessionId: string;
  dimensions: VolumeDimensions;
  currentSlices: {
    axial: number;
    coronal: number;
    sagittal: number;
  };
  modality: string;
  images: MultiviewImages;
  hasSegmentation: boolean;
}

export type RootStackParamList = {
  Upload: undefined;
  ModelSelect: { sessionId: string; numSlices: number };
  Results: {
    sessionId: string;
    numSlices: number;
    tumorStats: TumorStatistics;
  };
};
