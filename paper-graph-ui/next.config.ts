import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  reactCompiler: true,
  // /p/[category]/[slug] uses generateStaticParams + readPaperBody to
  // pre-render every paper detail page to HTML at build time. Once
  // rendered the runtime serverless function never reads those .md files
  // again, but Next's tracer still tries to bundle them which blows the
  // 250MB function-size limit (329MB of paper content for 2193 papers).
  // Excluding them from the runtime trace is safe because the build
  // output already contains every paper as static HTML.
  outputFileTracingExcludes: {
    "*": [
      "src/content/papers/**/*",
    ],
  },
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
