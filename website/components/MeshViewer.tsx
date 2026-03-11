'use client';

import { Canvas } from '@react-three/fiber';
import { OrbitControls, Center } from '@react-three/drei';
import { memo, Suspense, useEffect, useState, type ReactNode } from 'react';
import { BackSide, type BufferGeometry } from 'three';
import { STLLoader } from 'three/examples/jsm/loaders/STLLoader.js';

function LoadingFallback(): ReactNode {
  return (
    <div className="flex h-full items-center justify-center">
      <p className="font-mono text-xs tracking-wide text-fd-muted-foreground">
        Loading model&hellip;
      </p>
    </div>
  );
}

function Mesh({ url }: { url: string }) {
  const [geometry, setGeometry] = useState<BufferGeometry | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    const loader = new STLLoader();
    loader.load(
      url,
      (geo) => {
        if (cancelled) {
          geo.dispose();
          return;
        }
        geo.computeVertexNormals();
        setGeometry((prev) => {
          prev?.dispose();
          return geo;
        });
        setError(null);
      },
      undefined,
      () => {
        if (!cancelled) setError(`Failed to load ${url}`);
      },
    );

    return () => {
      cancelled = true;
      setGeometry((prev) => {
        prev?.dispose();
        return null;
      });
    };
  }, [url]);

  if (error) return null;
  if (!geometry) return null;

  return (
    <Center>
      <mesh geometry={geometry}>
        <meshStandardMaterial color="#6b9fd4" />
      </mesh>
      <mesh geometry={geometry}>
        <meshStandardMaterial color="#c4a08a" side={BackSide} />
      </mesh>
      <lineSegments>
        <edgesGeometry args={[geometry]} />
        <lineBasicMaterial color="#2a4060" />
      </lineSegments>
    </Center>
  );
}

export const MeshViewer = memo(function MeshViewer({
  src,
  height = 400,
  cameraDistance = 7,
  fov = 50,
}: {
  src: string;
  height?: number;
  cameraDistance?: number;
  fov?: number;
}) {
  const d = cameraDistance;
  return (
    <div
      className="my-4 overflow-hidden rounded-lg border border-fd-border bg-fd-card/30"
      style={{ height }}
    >
      <Suspense fallback={<LoadingFallback />}>
        <Canvas camera={{ position: [d, d, d], fov }}>
          <ambientLight intensity={0.6} />
          <directionalLight position={[5, 5, 5]} intensity={1.0} />
          <directionalLight position={[-3, -2, -4]} intensity={0.3} />
          <Mesh url={src} />
          <OrbitControls />
        </Canvas>
      </Suspense>
    </div>
  );
});
