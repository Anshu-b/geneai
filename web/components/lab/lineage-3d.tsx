"use client";

import { OrbitControls } from "@react-three/drei";
import { Canvas } from "@react-three/fiber";
import { Component, useEffect, useLayoutEffect, useMemo, useRef, useState, type ReactNode } from "react";
import * as THREE from "three";
import {
  providerBucket,
  stableJitter,
  type GenomeView,
  type LayoutModel,
} from "@/lib/project";
import { readThemePaint, type ThemePaint } from "@/lib/theme";

const SCALE = 0.045;

function useThemePaint() {
  const [paint, setPaint] = useState<ThemePaint>(() => readThemePaint());
  useLayoutEffect(() => {
    const sync = () => setPaint(readThemePaint());
    sync();
    const observer = new MutationObserver(sync);
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
    return () => observer.disconnect();
  }, []);
  return paint;
}

function place(id: string, layout: LayoutModel, dummy: THREE.Object3D) {
  const pos = layout.nodes.get(id);
  if (!pos) return null;
  dummy.position.set(
    (pos.x - layout.width / 2) * SCALE,
    -(pos.y - layout.height / 2) * SCALE,
    (stableJitter(id) - 0.5) * 8,
  );
  return pos;
}

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
  paint,
  onSelect,
}: {
  views: GenomeView[];
  layout: LayoutModel;
  opacity: number;
  paint: ThemePaint;
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
      place(view.genome.id, layout, dummy);
      dummy.scale.setScalar(Math.max(0.35, pos.r * 0.16));
      dummy.updateMatrix();
      mesh.setMatrixAt(index, dummy.matrix);
      const bucket = providerBucket(view.genome.provider);
      color.set(view.status === "champion" ? paint.champ : paint.providers[bucket]);
      mesh.setColorAt(index, color);
    });
    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
  }, [views, layout, paint]);

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

function Edges({ views, layout, paint }: { views: GenomeView[]; layout: LayoutModel; paint: ThemePaint }) {
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
      <lineBasicMaterial color={paint.edge} transparent opacity={paint.edgeOpacity} />
    </lineSegments>
  );
}

function MarkerRings({ views, layout, paint }: { views: GenomeView[]; layout: LayoutModel; paint: ThemePaint }) {
  const marked = views.filter((view) => view.status === "elite" || view.status === "champion");
  const dummy = useMemo(() => new THREE.Object3D(), []);
  return (
    <>
      {marked.map((view) => {
        const pos = place(view.genome.id, layout, dummy);
        if (!pos) return null;
        const radius = Math.max(0.35, pos.r * 0.16) + (view.status === "champion" ? 0.28 : 0.18);
        return (
          <mesh key={view.genome.id} position={dummy.position.clone()} rotation={[Math.PI / 2, 0, 0]}>
            <ringGeometry args={[radius, radius + 0.07, 40]} />
            <meshBasicMaterial color={view.status === "champion" ? paint.champ : paint.elite} side={THREE.DoubleSide} />
          </mesh>
        );
      })}
    </>
  );
}

function Scene({ views, layout, onSelect, reduced }: Props) {
  const paint = useThemePaint();
  const living = views.filter((view) => view.status !== "pruned");
  const pruned = views.filter((view) => view.status === "pruned");
  return (
    <>
      <color attach="background" args={[paint.stage]} />
      <ambientLight intensity={paint.light ? 1.05 : 0.7} />
      <directionalLight position={[12, 18, 20]} intensity={paint.light ? 0.85 : 1.1} />
      <Edges views={views} layout={layout} paint={paint} />
      <Cloud key={`live-${living.length}`} views={living} layout={layout} opacity={1} paint={paint} onSelect={onSelect} />
      <Cloud key={`pruned-${pruned.length}`} views={pruned} layout={layout} opacity={paint.pruned} paint={paint} onSelect={onSelect} />
      <MarkerRings views={living} layout={layout} paint={paint} />
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
