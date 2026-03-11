'use client';

import { useCallback, useRef, useState } from 'react';
import Image from 'next/image';

interface ImageCompareProps {
  beforeSrc: string;
  afterSrc: string;
  beforeLabel?: string;
  afterLabel?: string;
  className?: string;
}

export function ImageCompare({
  beforeSrc,
  afterSrc,
  beforeLabel = 'Extracted',
  afterLabel = 'Smoothed',
  className = '',
}: ImageCompareProps) {
  const [position, setPosition] = useState(50);
  const containerRef = useRef<HTMLDivElement>(null);
  const dragging = useRef(false);

  const updatePosition = useCallback((clientX: number) => {
    const container = containerRef.current;
    if (!container) return;
    const rect = container.getBoundingClientRect();
    const x = clientX - rect.left;
    const pct = Math.min(95, Math.max(5, (x / rect.width) * 100));
    setPosition(pct);
  }, []);

  const onPointerDown = useCallback(
    (e: React.PointerEvent) => {
      dragging.current = true;
      (e.target as HTMLElement).setPointerCapture(e.pointerId);
      updatePosition(e.clientX);
    },
    [updatePosition],
  );

  const onPointerMove = useCallback(
    (e: React.PointerEvent) => {
      if (!dragging.current) return;
      updatePosition(e.clientX);
    },
    [updatePosition],
  );

  const onPointerUp = useCallback(() => {
    dragging.current = false;
  }, []);

  const onKeyDown = useCallback((e: React.KeyboardEvent) => {
    if (e.key === 'ArrowLeft') {
      setPosition((p) => Math.max(5, p - 2));
    } else if (e.key === 'ArrowRight') {
      setPosition((p) => Math.min(95, p + 2));
    }
  }, []);

  return (
    <div
      ref={containerRef}
      className={`relative select-none overflow-hidden focus-visible:ring-2 focus-visible:ring-brand-500 focus-visible:ring-offset-2 ${className}`}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
      role="slider"
      aria-label="Image comparison slider"
      aria-valuemin={5}
      aria-valuemax={95}
      aria-valuenow={Math.round(position)}
      tabIndex={0}
      onKeyDown={onKeyDown}
    >
      {/* Before image (full) */}
      <Image
        src={beforeSrc}
        alt="Before processing"
        width={800}
        height={800}
        className="block h-auto w-full"
        draggable={false}
        priority
      />

      {/* After image (clipped from left) */}
      <div
        className="absolute inset-0"
        style={{ clipPath: `inset(0 0 0 ${position}%)` }}
      >
        <Image
          src={afterSrc}
          alt="After processing"
          width={800}
          height={800}
          className="block h-auto w-full"
          draggable={false}
          priority
        />
      </div>

      {/* Labels */}
      <span
        className="absolute top-3 left-3 rounded bg-brand-900/60 px-1.5 py-0.5 font-mono text-[10px] uppercase tracking-widest text-white transition-opacity duration-200"
        style={{ opacity: position < 20 ? 0 : 1 }}
      >
        {beforeLabel}
      </span>
      <span
        className="absolute top-3 right-3 rounded bg-brand-900/60 px-1.5 py-0.5 font-mono text-[10px] uppercase tracking-widest text-white transition-opacity duration-200"
        style={{ opacity: position > 80 ? 0 : 1 }}
      >
        {afterLabel}
      </span>

      {/* Drag handle — wide hit area, narrow visible line */}
      <div
        className="absolute top-0 bottom-0 z-10 flex w-10 -translate-x-1/2 cursor-ew-resize items-center justify-center"
        style={{ left: `${position}%` }}
        onPointerDown={onPointerDown}
      >
        <div className="h-full w-0.5 bg-brand-500" />
        {/* Grip circle */}
        <div className="group absolute top-1/2 flex h-9 w-9 -translate-y-1/2 items-center justify-center rounded-full border-2 border-white bg-brand-600 shadow-md transition-transform duration-150 hover:scale-110">
          <svg
            width="16"
            height="16"
            viewBox="0 0 16 16"
            fill="none"
            className="text-white"
          >
            <path
              d="M4.5 4L1.5 8L4.5 12M11.5 4L14.5 8L11.5 12"
              stroke="currentColor"
              strokeWidth="1.5"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
        </div>
      </div>
    </div>
  );
}
