import Link from 'next/link';

const features = [
  {
    title: 'Multi-Format Input',
    description:
      'Load density fields from NumPy, MATLAB, VTK, HDF5, CSV, and more.',
  },
  {
    title: 'Automatic Pipeline',
    description:
      'Gaussian smoothing, thresholding, marching cubes extraction, and Taubin mesh smoothing in one call.',
  },
  {
    title: '2D & 3D Support',
    description:
      'Extract contours from 2D fields or triangle meshes from 3D volumes. SDF fields supported.',
  },
  {
    title: 'Quality Metrics',
    description:
      'Measure aspect ratio, min angle, and scaled Jacobian via PyVista. Benchmark and compare.',
  },
  {
    title: 'CLI & Python API',
    description:
      'Use the xtf command-line tool or import xeltofab directly in Python scripts.',
  },
  {
    title: 'Extensible Architecture',
    description:
      'Immutable pipeline state, pure stage functions, easy to add decimation, remeshing, or CAD export.',
  },
];

const pipelineSteps = [
  { label: 'Density Field', sub: 'numpy array [0,1]' },
  { label: 'Preprocess', sub: 'smooth, threshold, morphology' },
  { label: 'Extract', sub: 'marching cubes / squares' },
  { label: 'Smooth', sub: 'Taubin filtering' },
  { label: 'Output', sub: 'STL / OBJ / PLY' },
];

export default function HomePage() {
  return (
    <main className="flex flex-1 flex-col">
      {/* Hero */}
      <section className="flex flex-col items-center justify-center px-4 py-24 text-center">
        <h1 className="mb-4 text-5xl font-bold tracking-tight">XelToFab</h1>
        <p className="mb-2 text-xl text-fd-muted-foreground">
          Topology optimization post-processing pipeline
        </p>
        <p className="mb-8 max-w-xl text-fd-muted-foreground">
          Convert density fields from topology optimization solvers into clean,
          fabrication-ready triangle meshes.
        </p>
        <div className="flex gap-4">
          <Link
            href="/docs"
            className="rounded-lg bg-fd-primary px-6 py-3 font-medium text-fd-primary-foreground transition-colors hover:bg-fd-primary/90"
          >
            Get Started
          </Link>
          <Link
            href="/docs/api/process"
            className="rounded-lg border border-fd-border px-6 py-3 font-medium transition-colors hover:bg-fd-accent"
          >
            API Reference
          </Link>
        </div>
      </section>

      {/* Pipeline diagram */}
      <section className="mx-auto w-full max-w-4xl px-4 py-12">
        <h2 className="mb-8 text-center text-2xl font-semibold">Pipeline</h2>
        <div className="flex flex-col items-center gap-2 sm:flex-row sm:justify-between">
          {pipelineSteps.map((step, i) => (
            <div key={step.label} className="flex items-center gap-2">
              <div className="flex flex-col items-center rounded-lg border border-fd-border bg-fd-card px-4 py-3 text-center">
                <span className="font-medium">{step.label}</span>
                <span className="text-xs text-fd-muted-foreground">
                  {step.sub}
                </span>
              </div>
              {i < pipelineSteps.length - 1 && (
                <span className="hidden text-fd-muted-foreground sm:block">
                  &rarr;
                </span>
              )}
            </div>
          ))}
        </div>
      </section>

      {/* Features grid */}
      <section className="mx-auto w-full max-w-5xl px-4 py-12">
        <div className="grid gap-6 sm:grid-cols-2 lg:grid-cols-3">
          {features.map((f) => (
            <div
              key={f.title}
              className="rounded-lg border border-fd-border bg-fd-card p-6"
            >
              <h3 className="mb-2 font-semibold">{f.title}</h3>
              <p className="text-sm text-fd-muted-foreground">
                {f.description}
              </p>
            </div>
          ))}
        </div>
      </section>

      {/* Quick code example */}
      <section className="mx-auto w-full max-w-3xl px-4 py-12">
        <h2 className="mb-6 text-center text-2xl font-semibold">
          Three Lines to a Mesh
        </h2>
        <pre className="overflow-x-auto rounded-lg border border-fd-border bg-fd-card p-6 text-sm">
          <code>{`from xeltofab.io import load_density, save_mesh
from xeltofab.pipeline import process

result = process(load_density("density_field.npy"))
save_mesh(result, "output.stl")`}</code>
        </pre>
      </section>
    </main>
  );
}
