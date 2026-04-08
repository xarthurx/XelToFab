const concepts = [
  {
    name: 'Reference-Based X/F Monogram',
    file: '/branding/icon-xf-nav.svg',
    note: 'A navy x-stroke is crossed by a lighter accent stroke that also becomes the F crossbar, then dissolves into dots before ending as a solid mark.',
  },
] as const;

const chipSizes = [256, 64, 32] as const;

export default function IconPreviewPage() {
  return (
    <main className="mx-auto flex w-full max-w-6xl flex-1 flex-col px-6 py-16">
      <div className="max-w-3xl">
        <p className="font-mono text-sm font-medium uppercase tracking-widest text-brand-500">
          Brand Study
        </p>
        <h1 className="mt-3 text-4xl font-bold tracking-tight">
          Monogram concept preview
        </h1>
        <p className="mt-4 text-fd-muted-foreground leading-relaxed">
          Updated direction based on your sketch: one stroke reads as x, the
          contrasting stroke reads as f, and the lower-right tail transitions
          from subtle dots into a solid terminal form.
        </p>
      </div>

      <div className="mt-12 grid gap-8 lg:grid-cols-1">
        {concepts.map((concept) => (
          <section
            key={concept.name}
            className="rounded-2xl border border-fd-border bg-fd-card p-6"
          >
            <h2 className="text-2xl font-bold tracking-tight">{concept.name}</h2>
            <p className="mt-3 text-sm text-fd-muted-foreground leading-relaxed">
              {concept.note}
            </p>

            <div className="mt-6 rounded-2xl border border-fd-border bg-white p-6">
              <img
                src={concept.file}
                alt={concept.name}
                width={256}
                height={256}
                className="mx-auto h-48 w-48"
              />
            </div>

            <div className="mt-6 grid gap-4 sm:grid-cols-3">
              {chipSizes.map((size) => (
                <div
                  key={size}
                  className="rounded-xl border border-fd-border bg-fd-background p-4"
                >
                  <p className="font-mono text-xs uppercase tracking-widest text-fd-muted-foreground">
                    {size} px
                  </p>
                  <div className="mt-3 flex min-h-24 items-center justify-center rounded-lg border border-dashed border-fd-border bg-white">
                    <img
                      src={concept.file}
                      alt={`${concept.name} ${size}px`}
                      width={size}
                      height={size}
                    />
                  </div>
                </div>
              ))}
            </div>

            <div className="mt-6 rounded-xl border border-fd-border bg-slate-950 p-4">
              <p className="font-mono text-xs uppercase tracking-widest text-slate-400">
                Dark background check
              </p>
              <div className="mt-3 flex min-h-24 items-center justify-center rounded-lg border border-slate-800 bg-slate-900">
                <img
                  src={concept.file}
                  alt={`${concept.name} on dark background`}
                  width={64}
                  height={64}
                />
              </div>
            </div>
          </section>
        ))}
      </div>
    </main>
  );
}
