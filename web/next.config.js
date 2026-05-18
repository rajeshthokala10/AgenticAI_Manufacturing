/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // Long blocking calls (e.g. /api/diagnostic on local Ollama can take
  // 5–6 minutes) overrun the dev server's default ~30s proxy timeout and
  // surface to the browser as a 500 ``socket hang up`` error. Allow up to
  // 10 minutes for proxied requests.
  experimental: {
    proxyTimeout: 600_000,
  },
  async rewrites() {
    const apiOrigin = process.env.NEXT_PUBLIC_API_ORIGIN || "http://localhost:8000";
    return [
      {
        source: "/api/:path*",
        destination: `${apiOrigin}/api/:path*`,
      },
    ];
  },
};

module.exports = nextConfig;
