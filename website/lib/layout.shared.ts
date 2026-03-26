import type { BaseLayoutProps } from 'fumadocs-ui/layouts/shared';

export const shikiThemes = {
  light: 'ayu-light',
  dark: 'ayu-dark',
} as const;

export const gitConfig = {
  user: 'xarthurx',
  repo: 'XelToFab',
  branch: 'main',
};

export function baseOptions(): BaseLayoutProps {
  return {
    nav: {
      title: 'XelToFab',
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
    ],
    githubUrl: `https://github.com/${gitConfig.user}/${gitConfig.repo}`,
  };
}
