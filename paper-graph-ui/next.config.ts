import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  reactCompiler: true,
  experimental: {
    turbopackFileSystemCacheForDev: true,
    turbopackFileSystemCacheForBuild: true,
  },
  transpilePackages: ["geist", "react-icons"],
  turbopack: {
    rules: {
      "*.inline.svg": {
        loaders: [
          {
            loader: "@svgr/webpack",
            options: {
              svgo: true,
              svgoConfig: {
                plugins: [
                  { name: "preset-default", params: { overrides: { removeViewBox: false } } },
                  "prefixIds",
                ],
              },
            },
          },
        ],
        as: "*.js",
      },
    },
    resolveAlias: {
      fs: { browser: "./empty.js" },
      module: { browser: "./empty.js" },
    },
  },
  images: {
    formats: ["image/avif", "image/webp"],
    qualities: [75, 85, 90, 95, 99, 100],
    remotePatterns: [{ protocol: "https", hostname: "**" }],
  },
};

export default nextConfig;
