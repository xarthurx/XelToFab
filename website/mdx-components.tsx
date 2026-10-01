import defaultMdxComponents from 'fumadocs-ui/mdx';
import type { MDXComponents } from 'mdx/types';
import { MeshViewer } from '@/components/MeshViewer';

export function getMDXComponents(components?: MDXComponents): MDXComponents {
  // Object.assign keeps MDXComponents as an intersection; an object spread would
  // flatten it and trip over the `never`-props JSX elements that
  // @react-three/fiber adds to React.JSX.IntrinsicElements.
  return Object.assign({}, defaultMdxComponents, { MeshViewer }, components);
}
