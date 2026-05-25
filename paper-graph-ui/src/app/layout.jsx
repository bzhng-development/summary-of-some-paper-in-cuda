import 'styles/globals.css';
import 'styles/app.css';
import 'katex/dist/katex.min.css';
import { GeistMono } from 'geist/font/mono';
import { Suspense } from 'react';

import { TabsProvider } from 'contexts/tabs-context';

import QueuePill from './_components/queue-pill';
import ResumeBanner from './_components/resume-banner';
import SiteHeader from './_components/site-header';
import { inter, esbuild } from './fonts';

export const metadata = {
  title: 'Paper Graph — LLM, Systems & RL',
  description:
    'A reading-graph of paper summaries across LLMs, RL training, inference systems, and architectures. Built for mobile reading.',
};

export const viewport = {
  width: 'device-width',
  initialScale: 1,
  viewportFit: 'cover',
  themeColor: '#0c0d0d',
};

const RootLayout = ({ children }) => (
  <html
    lang="en"
    className={`${inter.variable} ${esbuild.variable} ${GeistMono.variable} dark`}
    suppressHydrationWarning
  >
    <body className="bg-black-pure text-white">
      <TabsProvider>
        <SiteHeader />
        <Suspense fallback={null}>
          <ResumeBanner />
        </Suspense>
        {children}
        <QueuePill />
      </TabsProvider>
    </body>
  </html>
);

export default RootLayout;
