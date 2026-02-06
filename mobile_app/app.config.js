import "dotenv/config";

export default {
  expo: {
    name: "Brain Tumor Segmentation",
    slug: "brain-tumor-segmentation",
    version: "1.0.0",
    orientation: "portrait",
    userInterfaceStyle: "dark",
    splash: {
      backgroundColor: "#0a0a1a",
    },
    assetBundlePatterns: ["**/*"],
    ios: {
      supportsTablet: true,
    },
    android: {
      adaptiveIcon: {
        backgroundColor: "#0a0a1a",
      },
    },
    extra: {
      apiBaseUrl: process.env.API_BASE_URL || "http://localhost:8000",
    },
  },
};
