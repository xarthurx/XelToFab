import Link from 'next/link';
import { codeToHtml } from 'shiki';

const codeExample = `from xeltofab.io import load_density, save_mesh
from xeltofab.pipeline import process

result = process(load_density("field.npy"))
save_mesh(result, "output.stl")`;

const pipelineSteps = [
  {
    num: '01',
    label: 'Preprocess',
    detail: 'Gaussian smooth, threshold, morphological cleanup',
    input: 'density field [0,1]',
  },
  {
    num: '02',
    label: 'Extract',
    detail: 'Marching cubes (3D) or marching squares (2D)',
    input: 'binary volume',
  },
  {
    num: '03',
    label: 'Smooth',
    detail: 'Taubin filtering removes staircase artifacts',
    input: 'triangle mesh',
  },
  {
    num: '04',
    label: 'Output',
    detail: 'STL, OBJ, PLY — fabrication-ready geometry',
    input: 'smoothed mesh',
  },
];

export default async function HomePage() {
  const highlighted = await codeToHtml(codeExample, {
    lang: 'python',
    themes: { light: 'catppuccin-latte', dark: 'github-dark' },
    defaultColor: false,
  });
  return (
    <main className="flex flex-1 flex-col">
      {/* Hero — left-aligned, asymmetric */}
      <section className="mx-auto w-full max-w-6xl px-6 pt-32 pb-20">
        <p className="mb-4 font-[family-name:var(--font-mono)] text-sm font-medium uppercase tracking-widest text-brand-500">
          Topology Optimization Post-Processing
        </p>
        <h1 className="mb-6 text-[clamp(2.5rem,6vw,4.5rem)] font-bold leading-[1.05] tracking-tight">
          Density fields to
          <br />
          fabrication-ready meshes.
        </h1>
        <p className="mb-10 max-w-lg text-lg text-fd-muted-foreground leading-relaxed">
          <strong className="font-semibold text-fd-foreground">XelToFab</strong> handles the full pipeline — preprocessing, extraction,
          smoothing, and quality analysis — so your topology optimization
          results are ready for FEA validation or 3D printing.
        </p>
        <div className="flex flex-wrap gap-3">
          <Link
            href="/docs"
            className="rounded-md bg-brand-600 px-5 py-2.5 font-[family-name:var(--font-mono)] text-sm font-medium text-white transition-colors hover:bg-brand-700 dark:bg-brand-400 dark:text-brand-900 dark:hover:bg-brand-300"
          >
            Get Started
          </Link>
          <Link
            href="/docs/api/process"
            className="rounded-md border border-fd-border px-5 py-2.5 font-[family-name:var(--font-mono)] text-sm font-medium transition-colors hover:bg-fd-accent"
          >
            API Reference
          </Link>
        </div>
      </section>

      {/* Pipeline — numbered steps, left-aligned, vertical rhythm */}
      <section className="border-y border-fd-border bg-fd-card/50">
        <div className="mx-auto w-full max-w-6xl px-6 py-20">
          <p className="mb-2 font-[family-name:var(--font-mono)] text-sm font-medium uppercase tracking-widest text-brand-500">
            How it works
          </p>
          <h2 className="mb-12 text-3xl font-bold tracking-tight">
            Four stages, one function call.
          </h2>
          <div className="grid gap-0 sm:grid-cols-2 lg:grid-cols-4">
            {pipelineSteps.map((step, i) => (
              <div
                key={step.num}
                className={`relative py-6 ${i < pipelineSteps.length - 1 ? 'lg:border-r lg:border-fd-border lg:pr-8' : ''} ${i > 0 ? 'lg:pl-8' : ''}`}
              >
                <span className="font-[family-name:var(--font-mono)] text-4xl font-bold text-brand-200 dark:text-brand-800">
                  {step.num}
                </span>
                <h3 className="mt-2 text-lg font-semibold">{step.label}</h3>
                <p className="mt-1 text-sm text-fd-muted-foreground leading-relaxed">
                  {step.detail}
                </p>
                <p className="mt-3 font-[family-name:var(--font-mono)] text-xs text-fd-muted-foreground">
                  Input: {step.input}
                </p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Code example — full-width band */}
      <section className="mx-auto w-full max-w-6xl px-6 py-20">
        <div className="grid items-center gap-12 lg:grid-cols-2">
          <div>
            <p className="mb-2 font-[family-name:var(--font-mono)] text-sm font-medium uppercase tracking-widest text-brand-500">
              Minimal API
            </p>
            <h2 className="mb-4 text-3xl font-bold tracking-tight">
              Three lines to a mesh.
            </h2>
            <p className="text-fd-muted-foreground leading-relaxed">
              Load any density field — NumPy, MATLAB, VTK, HDF5 — run the
              pipeline, save the result. Parameters are sensible defaults you
              can override when needed.
            </p>
            <div className="mt-6 flex flex-wrap gap-x-6 gap-y-2 font-[family-name:var(--font-mono)] text-sm text-fd-muted-foreground">
              <span>.npy</span>
              <span>.mat</span>
              <span>.vtk</span>
              <span>.h5</span>
              <span>.csv</span>
              <span>.npz</span>
            </div>
          </div>
          <div
            className="overflow-x-auto rounded-lg border border-fd-border text-sm leading-relaxed [&_pre]:p-6 [&_pre]:!bg-fd-card"
            dangerouslySetInnerHTML={{ __html: highlighted }}
          />
        </div>
      </section>

      {/* Capabilities — alternating layout, not a card grid */}
      <section className="border-t border-fd-border bg-fd-card/50">
        <div className="mx-auto w-full max-w-6xl px-6 py-20">
          <p className="mb-2 font-[family-name:var(--font-mono)] text-sm font-medium uppercase tracking-widest text-brand-500">
            Capabilities
          </p>
          <h2 className="mb-16 text-3xl font-bold tracking-tight">
            Built for the topology optimization workflow.
          </h2>

          <div className="grid gap-16 sm:grid-cols-2">
            {/* Left column — stacked items with generous spacing */}
            <div className="flex flex-col gap-12">
              <div>
                <h3 className="text-lg font-semibold">2D &amp; 3D Support</h3>
                <p className="mt-2 text-sm text-fd-muted-foreground leading-relaxed">
                  Extract contours from 2D fields or triangle meshes from 3D
                  volumes. SDF fields from neural models are supported with
                  direct extraction at the zero level set.
                </p>
              </div>
              <div>
                <h3 className="text-lg font-semibold">Quality Metrics</h3>
                <p className="mt-2 text-sm text-fd-muted-foreground leading-relaxed">
                  Measure aspect ratio, minimum angle, and scaled Jacobian via
                  PyVista. Validate meshes against FEA requirements before
                  downstream use.
                </p>
              </div>
              <div>
                <h3 className="text-lg font-semibold">
                  Extensible Architecture
                </h3>
                <p className="mt-2 text-sm text-fd-muted-foreground leading-relaxed">
                  Immutable pipeline state, pure stage functions. Add
                  decimation, isotropic remeshing, or CAD export as new stages
                  without touching existing code.
                </p>
              </div>
            </div>

            {/* Right column — offset for visual rhythm */}
            <div className="flex flex-col gap-12 sm:pt-8">
              <div>
                <h3 className="text-lg font-semibold">Multi-Format Input</h3>
                <p className="mt-2 text-sm text-fd-muted-foreground leading-relaxed">
                  Load density fields from NumPy, MATLAB, VTK, HDF5, CSV, and
                  more. Auto-detection of common topology optimization variable
                  names.
                </p>
              </div>
              <div>
                <h3 className="text-lg font-semibold">CLI &amp; Python API</h3>
                <p className="mt-2 text-sm text-fd-muted-foreground leading-relaxed">
                  Use the{' '}
                  <code className="rounded bg-fd-card px-1.5 py-0.5 text-xs border border-fd-border">
                    xtf
                  </code>{' '}
                  command-line tool for batch processing, or import{' '}
                  <code className="rounded bg-fd-card px-1.5 py-0.5 text-xs border border-fd-border">
                    xeltofab
                  </code>{' '}
                  directly in Python scripts and notebooks.
                </p>
              </div>
              <div>
                <h3 className="text-lg font-semibold">
                  Automatic Pipeline
                </h3>
                <p className="mt-2 text-sm text-fd-muted-foreground leading-relaxed">
                  Gaussian smoothing, thresholding, marching cubes, and Taubin
                  mesh smoothing — all configured with sensible defaults and
                  overridable per-stage.
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* CTA */}
      <section className="mx-auto w-full max-w-6xl px-6 py-24">
        <div className="flex flex-col items-start gap-4 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <h2 className="text-2xl font-bold tracking-tight">
              Ready to process your first field?
            </h2>
            <p className="mt-1 text-fd-muted-foreground">
              Install with pip or uv and run in under a minute.
            </p>
          </div>
          <Link
            href="/docs/getting-started/installation"
            className="shrink-0 rounded-md bg-brand-600 px-5 py-2.5 font-[family-name:var(--font-mono)] text-sm font-medium text-white transition-colors hover:bg-brand-700 dark:bg-brand-400 dark:text-brand-900 dark:hover:bg-brand-300"
          >
            Installation Guide
          </Link>
        </div>
      </section>
    </main>
  );
}
