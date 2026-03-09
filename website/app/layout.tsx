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
  variable: '--font-mono',
});

export const metadata: Metadata = {
  title: {
    default: 'XelToFab',
    template: '%s | XelToFab',
  },
  description:
    'Topology optimization post-processing pipeline: density fields to fabrication-ready meshes.',
};

export default function Layout({ children }: LayoutProps<'/'>) {
  return (
    <html
      lang="en"
      className={`${body.variable} ${mono.variable}`}
      suppressHydrationWarning
    >
      <body className="flex flex-col min-h-screen font-[family-name:var(--font-body)]">
        <RootProvider>{children}</RootProvider>
      </body>
    </html>
  );
}
