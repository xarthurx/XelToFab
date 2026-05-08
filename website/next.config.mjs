import { createMDX } from 'fumadocs-mdx/next';

const withMDX = createMDX();

/** @type {import('next').NextConfig} */
const config = {
  reactStrictMode: true,
  async redirects() {
    const host = process.env.VERCEL_PROJECT_PRODUCTION_URL || '';
    if (host.includes('vercel.app')) {
      return [
        {
          source: '/:path*',
          destination: 'https://xeltofab.ethz.ch/:path*',
          permanent: true,
        },
      ];
    }
    return [];
  },
  async rewrites() {
    return [
      {
        source: '/docs/:path*.mdx',
        destination: '/llms.mdx/docs/:path*',
      },
    ];
  },
};

export default withMDX(config);
