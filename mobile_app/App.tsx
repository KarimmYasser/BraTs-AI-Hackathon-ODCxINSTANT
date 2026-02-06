/**
 * Brain Tumor Segmentation - React Native Mobile App
 * Main entry point with navigation
 */

import React from "react";
import { StatusBar } from "expo-status-bar";
import { NavigationContainer, DefaultTheme } from "@react-navigation/native";
import { createNativeStackNavigator } from "@react-navigation/native-stack";
import { SafeAreaProvider } from "react-native-safe-area-context";
import { GestureHandlerRootView } from "react-native-gesture-handler";

import { UploadScreen } from "./src/screens/UploadScreen";
import { ModelSelectScreen } from "./src/screens/ModelSelectScreen";
import { ResultsScreen } from "./src/screens/ResultsScreen";
import { colors } from "./src/theme/colors";
import { RootStackParamList } from "./src/types";

const Stack = createNativeStackNavigator<RootStackParamList>();

// Custom dark theme matching web app
const DarkTheme = {
  ...DefaultTheme,
  dark: true,
  colors: {
    ...DefaultTheme.colors,
    primary: colors.accentPrimary,
    background: colors.bgPrimary,
    card: colors.bgSecondary,
    text: colors.textPrimary,
    border: colors.borderColor,
    notification: colors.accentSecondary,
  },
};

export default function App() {
  return (
    <GestureHandlerRootView style={{ flex: 1 }}>
      <SafeAreaProvider>
        <NavigationContainer theme={DarkTheme}>
          <StatusBar style="light" />
          <Stack.Navigator
            initialRouteName="Upload"
            screenOptions={{
              headerStyle: {
                backgroundColor: colors.bgSecondary,
              },
              headerTintColor: colors.textPrimary,
              headerTitleStyle: {
                fontWeight: "600",
              },
              headerShadowVisible: false,
              animation: "slide_from_right",
            }}
          >
            <Stack.Screen
              name="Upload"
              component={UploadScreen}
              options={{ headerShown: false }}
            />
            <Stack.Screen
              name="ModelSelect"
              component={ModelSelectScreen}
              options={{
                title: "Select Model",
                headerBackTitle: "Back",
              }}
            />
            <Stack.Screen
              name="Results"
              component={ResultsScreen}
              options={{
                title: "Results",
                headerBackVisible: false,
              }}
            />
          </Stack.Navigator>
        </NavigationContainer>
      </SafeAreaProvider>
    </GestureHandlerRootView>
  );
}
