import { RootProvider } from 'fumadocs-ui/provider/next';
import './global.css';
import { Source_Sans_3, JetBrains_Mono } from 'next/font/google';
import type { Metadata } from 'next';

const body = Source_Sans_3({
  subsets: ['latin'],
  variable: '--font-body',
});

const mono = JetBrains_Mono({
  subsets: ['latin'],
  variable: '--font-code',
});

export const metadata: Metadata = {
  title: {
    default: 'XelToFab',
    template: '%s | XelToFab',
  },
  description:
    'Design fields to fabrication-ready geometry. Post-processing pipeline for topology optimization, neural fields, and computational design.',
};

export default function Layout({ children }: LayoutProps<'/'>) {
  return (
    <html
      lang="en"
      className={`${body.variable} ${mono.variable}`}
      suppressHydrationWarning
    >
      <body className="flex min-h-screen flex-col font-sans">
        <RootProvider i18n={{ locale: 'en', translations: { toc: 'Outline' } }}>
          {children}
        </RootProvider>
      </body>
    </html>
  );
}
