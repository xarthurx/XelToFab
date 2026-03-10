import { docs } from '@/.source/server';
import { loader } from 'fumadocs-core/source';

export const source = loader({
  baseUrl: '/docs',
  source: docs.toFumadocsSource(),
});

export function getPageImage(page: ReturnType<typeof source.getPage>) {
  return {
    url: `/og/docs/${page?.slugs.join('/') ?? ''}`,
    width: 1200,
    height: 630,
  };
}

export async function getLLMText(
  page: ReturnType<typeof source.getPage>,
): Promise<string | undefined> {
  return page?.data.getText('processed');
}
