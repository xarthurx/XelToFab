import type { MetadataRoute } from 'next';
import { source } from '@/lib/source';
import { SITE_URL } from '@/lib/site';

function toAbsoluteUrl(path: string): string {
  return new URL(path, SITE_URL).toString();
}

export default function sitemap(): MetadataRoute.Sitemap {
  const docsEntries = source.getPages().map((page) => {
    const docPath = page.slugs.length > 0 ? `/docs/${page.slugs.join('/')}` : '/docs';

    return {
      url: toAbsoluteUrl(docPath),
      changeFrequency: 'weekly' as const,
      priority: docPath === '/docs' ? 0.9 : 0.7,
    };
  });

  const staticEntries: MetadataRoute.Sitemap = [
    {
      url: toAbsoluteUrl('/'),
      changeFrequency: 'weekly',
      priority: 1,
    },
  ];

  return [...staticEntries, ...docsEntries];
}
