'use client';

import { Canvas } from '@react-three/fiber';
import { OrbitControls, Center } from '@react-three/drei';
import { Suspense, useEffect, useState } from 'react';
import * as THREE from 'three';
import { STLLoader } from 'three/examples/jsm/loaders/STLLoader.js';

function Mesh({ url }: { url: string }) {
  const [geometry, setGeometry] = useState<THREE.BufferGeometry | null>(null);

  useEffect(() => {
    const loader = new STLLoader();
    loader.load(url, (geo) => {
      geo.computeVertexNormals();
      setGeometry(geo);
    });
  }, [url]);

  if (!geometry) return null;

  return (
    <Center>
      <mesh geometry={geometry}>
        <meshStandardMaterial color="steelblue" />
      </mesh>
      <mesh geometry={geometry}>
        <meshBasicMaterial color="black" wireframe transparent opacity={0.1} />
      </mesh>
    </Center>
  );
}

export function MeshViewer({
  src,
  height = 400,
}: {
  src: string;
  height?: number;
}) {
  return (
    <div
      className="my-4 overflow-hidden rounded-lg border border-fd-border"
      style={{ height }}
    >
      <Canvas camera={{ position: [2, 2, 2], fov: 50 }}>
        <ambientLight intensity={0.4} />
        <directionalLight position={[5, 5, 5]} intensity={0.8} />
        <Suspense fallback={null}>
          <Mesh url={src} />
        </Suspense>
        <OrbitControls />
      </Canvas>
    </div>
  );
}
