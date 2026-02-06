# 🧠 Brain Tumor Segmentation - Mobile App

React Native mobile application for brain tumor segmentation visualization.

![React Native](https://img.shields.io/badge/React_Native-0.73-blue)
![Expo](https://img.shields.io/badge/Expo-50-black)
![TypeScript](https://img.shields.io/badge/TypeScript-5.1-blue)

## Features

- 📱 **Cross-platform** - iOS and Android support
- 🖼️ **2D Slice Viewer** - Navigate through brain MRI slices
- 🎯 **3D Multi-View** - Axial, coronal, and sagittal visualization
- 📊 **Tumor Statistics** - NCR, ED, and ET voxel counts
- 🎨 **Dark Theme** - Matching the web app design

## Prerequisites

- Node.js 18+
- Expo Go app on your phone
- Backend server running (see [webapp repo](../webapp))

## Quick Start

```bash
# Install dependencies
npm install

# Copy environment file
cp .env.example .env

# Edit .env with your backend URL
# API_BASE_URL=http://YOUR_SERVER_IP:8000

# Start Expo
npx expo start
```

Scan the QR code with Expo Go app.

## Configuration

Edit `.env` to set your backend URL:

```env
API_BASE_URL=http://192.168.1.100:8000
```

> **Note**: Use your computer's local IP, not `localhost`, when testing on a physical device.

## Project Structure

```
mobile/
├── App.tsx                 # Navigation setup
├── app.config.js           # Expo config with env vars
├── src/
│   ├── api/
│   │   └── client.ts       # API client
│   ├── components/
│   │   ├── SliceViewer.tsx
│   │   └── MultiViewDisplay.tsx
│   ├── screens/
│   │   ├── UploadScreen.tsx
│   │   ├── ModelSelectScreen.tsx
│   │   └── ResultsScreen.tsx
│   ├── theme/
│   │   └── colors.ts       # Theme colors
│   └── types/
│       └── index.ts        # TypeScript types
```

## Building for Production

```bash
# Build for Android
npx expo build:android

# Build for iOS
npx expo build:ios
```

## License

MIT
