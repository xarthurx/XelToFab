import type { BaseLayoutProps } from 'fumadocs-ui/layouts/shared';

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
    githubUrl: `https://github.com/${gitConfig.user}/${gitConfig.repo}`,
    links: [
      {
        text: 'Docs',
        url: '/docs',
        active: 'nested-url',
      },
    ],
  };
}
