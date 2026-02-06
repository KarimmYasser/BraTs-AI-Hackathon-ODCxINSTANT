/**
 * Theme colors matching the web application
 */

export const colors = {
  // Backgrounds
  bgPrimary: "#0a0a1a",
  bgSecondary: "#12122a",
  bgTertiary: "#1a1a3a",
  bgCard: "rgba(30, 30, 60, 0.8)",
  bgCardHover: "rgba(40, 40, 80, 0.9)",

  // Accent colors
  accentPrimary: "#6366f1",
  accentSecondary: "#8b5cf6",
  accentTertiary: "#a855f7",

  // Text colors
  textPrimary: "#ffffff",
  textSecondary: "#a0a0c0",
  textMuted: "#6b6b8a",

  // Border
  borderColor: "rgba(99, 102, 241, 0.3)",
  borderGlow: "rgba(139, 92, 246, 0.5)",

  // Tumor class colors
  ncrColor: "#ef4444", // Necrotic Core - Red
  edColor: "#22c55e", // Edema - Green
  etColor: "#eab308", // Enhancing Tumor - Yellow

  // Status colors
  success: "#22c55e",
  error: "#ef4444",
  warning: "#eab308",
};

export const spacing = {
  xs: 4,
  sm: 8,
  md: 16,
  lg: 24,
  xl: 32,
  xxl: 48,
};

export const borderRadius = {
  sm: 8,
  md: 12,
  lg: 16,
  xl: 24,
  full: 9999,
};

export const typography = {
  h1: {
    fontSize: 28,
    fontWeight: "700" as const,
  },
  h2: {
    fontSize: 22,
    fontWeight: "600" as const,
  },
  h3: {
    fontSize: 18,
    fontWeight: "600" as const,
  },
  body: {
    fontSize: 16,
    fontWeight: "400" as const,
  },
  caption: {
    fontSize: 14,
    fontWeight: "400" as const,
  },
  small: {
    fontSize: 12,
    fontWeight: "400" as const,
  },
};
