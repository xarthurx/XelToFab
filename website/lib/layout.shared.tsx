import type { BaseLayoutProps } from 'fumadocs-ui/layouts/shared';

export const shikiThemes = {
  light: 'ayu-light',
  dark: 'ayu-dark',
} as const;

export const gitConfig = {
  user: 'xarthurx',
  repo: 'XelToFab',
};

export function baseOptions(): BaseLayoutProps {
  return {
    nav: {
      title: (
        <>
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src="/branding/icon-xf-nav.svg" alt="" width={24} height={24} />
          XelToFab
        </>
      ),
    },
    links: [
      {
        text: 'Documentation',
        url: '/docs',
        active: 'nested-url',
      },
      {
        text: 'API Reference',
        url: '/docs/api/process',
        active: 'nested-url',
      },
      {
        text: 'Cite',
        url: '/docs/citation',
      },
    ],
    githubUrl: `https://github.com/${gitConfig.user}/${gitConfig.repo}`,
  };
}
