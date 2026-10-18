"use client";

import { OrbitControls } from "@react-three/drei";
import { Canvas } from "@react-three/fiber";
import { Component, useEffect, useLayoutEffect, useMemo, useRef, type ReactNode } from "react";
import * as THREE from "three";
import {
  PROVIDER_COLOR,
  providerBucket,
  stableJitter,
  type GenomeView,
  type LayoutModel,
} from "@/lib/project";

const SCALE = 0.045;

interface Props {
  views: GenomeView[];
  layout: LayoutModel;
  onSelect: (id: string) => void;
  reduced: boolean;
}

function Cloud({
  views,
  layout,
  opacity,
  onSelect,
}: {
  views: GenomeView[];
  layout: LayoutModel;
  opacity: number;
  onSelect: (id: string) => void;
}) {
  const ref = useRef<THREE.InstancedMesh>(null);
  const ids = useMemo(() => views.map((view) => view.genome.id), [views]);
  useLayoutEffect(() => {
    const mesh = ref.current;
    if (!mesh) return;
    const dummy = new THREE.Object3D();
    const color = new THREE.Color();
    views.forEach((view, index) => {
      const pos = layout.nodes.get(view.genome.id);
      if (!pos) return;
      dummy.position.set(
        (pos.x - layout.width / 2) * SCALE,
        -(pos.y - layout.height / 2) * SCALE,
        (stableJitter(view.genome.id) - 0.5) * 8,
      );
      dummy.scale.setScalar(Math.max(0.35, pos.r * 0.16));
      dummy.updateMatrix();
      mesh.setMatrixAt(index, dummy.matrix);
      color.set(PROVIDER_COLOR[providerBucket(view.genome.provider)]);
      mesh.setColorAt(index, color);
    });
    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
  }, [views, layout]);

  if (views.length === 0) return null;
  return (
    <instancedMesh
      ref={ref}
      args={[undefined, undefined, views.length]}
      onClick={(event) => {
        event.stopPropagation();
        const index = event.instanceId;
        if (index != null && ids[index]) onSelect(ids[index]);
      }}
    >
      <sphereGeometry args={[1, 16, 16]} />
      <meshStandardMaterial transparent opacity={opacity} roughness={0.42} metalness={0.08} />
    </instancedMesh>
  );
}

function Edges({ views, layout }: { views: GenomeView[]; layout: LayoutModel }) {
  const geometry = useMemo(() => {
    const positions: number[] = [];
    const born = new Set(views.map((view) => view.genome.id));
    for (const view of views) {
      const child = layout.nodes.get(view.genome.id);
      if (!child) continue;
      const zChild = (stableJitter(view.genome.id) - 0.5) * 8;
      for (const parentId of view.genome.parent_ids) {
        if (!born.has(parentId)) continue;
        const parent = layout.nodes.get(parentId);
        if (!parent) continue;
        positions.push(
          (parent.x - layout.width / 2) * SCALE,
          -(parent.y - layout.height / 2) * SCALE,
          (stableJitter(parentId) - 0.5) * 8,
          (child.x - layout.width / 2) * SCALE,
          -(child.y - layout.height / 2) * SCALE,
          zChild,
        );
      }
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
    return geo;
  }, [views, layout]);

  useEffect(() => () => geometry.dispose(), [geometry]);

  return (
    <lineSegments geometry={geometry}>
      <lineBasicMaterial color="#c9ddd6" transparent opacity={0.38} />
    </lineSegments>
  );
}

function Scene({ views, layout, onSelect, reduced }: Props) {
  const living = views.filter((view) => view.status !== "pruned");
  const pruned = views.filter((view) => view.status === "pruned");
  return (
    <>
      <color attach="background" args={["#070908"]} />
      <ambientLight intensity={0.7} />
      <directionalLight position={[12, 18, 20]} intensity={1.1} />
      <Edges views={views} layout={layout} />
      <Cloud key={`live-${living.length}`} views={living} layout={layout} opacity={1} onSelect={onSelect} />
      <Cloud key={`pruned-${pruned.length}`} views={pruned} layout={layout} opacity={0.3} onSelect={onSelect} />
      <OrbitControls enableDamping={!reduced} makeDefault />
    </>
  );
}

class WebglBoundary extends Component<{ children: ReactNode }, { failed: boolean }> {
  state = { failed: false };
  static getDerivedStateFromError() {
    return { failed: true };
  }
  render() {
    if (this.state.failed) {
      return <p className="p-6 text-sm text-dim">3D needs WebGL in this browser. The 2D lineage is the same export.</p>;
    }
    return this.props.children;
  }
}

export default function Lineage3D(props: Props) {
  return (
    <WebglBoundary>
      <div className="h-[68vh] min-h-[520px] overflow-hidden rounded-md border border-line" data-testid="lineage-3d">
        <Canvas camera={{ position: [0, 8, 52], fov: 42, near: 0.1, far: 400 }} dpr={[1, 1.5]} gl={{ antialias: true, powerPreference: "high-performance" }}>
          <Scene {...props} />
        </Canvas>
      </div>
      <p className="mt-2 font-mono text-[10px] text-faint">
        Depth is a stable jitter so nodes do not stack. It is not a fitness distance or a biological measure.
      </p>
    </WebglBoundary>
  );
}
