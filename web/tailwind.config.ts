import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
    "./lib/**/*.{ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // Claude-inspired warm-cream palette
        cream: {
          50: "#fbfaf7",
          100: "#f7f5ee",
          200: "#ece7d8",
          300: "#ddd6c2",
        },
        ink: {
          900: "#1f1e1c",
          800: "#2d2c29",
          700: "#3d3b37",
          600: "#52504a",
          500: "#74716a",
          400: "#9a978f",
          300: "#bfbcb3",
        },
        copper: {
          400: "#e07a55",
          500: "#c25b3f",
          600: "#a64931",
          700: "#7f3724",
        },
      },
      fontFamily: {
        sans: [
          "ui-sans-serif",
          "-apple-system",
          "BlinkMacSystemFont",
          "Inter",
          "Segoe UI",
          "Roboto",
          "sans-serif",
        ],
        serif: ["ui-serif", "Georgia", "Cambria", "Times New Roman", "serif"],
      },
      boxShadow: {
        soft: "0 2px 24px -8px rgba(31, 30, 28, 0.12)",
        ring: "0 0 0 1px rgba(31, 30, 28, 0.06)",
      },
      borderRadius: {
        bubble: "22px",
      },
    },
  },
  plugins: [],
};

export default config;
