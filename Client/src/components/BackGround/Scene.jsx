import React, { useRef, useMemo, useState } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import * as THREE from "three";
import CameraFeed from "../CameraFeed/CameraFeed";

const Dot = ({ position, mouse, color }) => {
  const ref = useRef();
  const [originalPos] = useState(() => new THREE.Vector3(...position));

  useFrame(() => {
    if (!ref.current) return;
    const dot = ref.current.position;
    const distance = mouse.current.distanceTo(dot);

    if (distance < 3) {
      const strength = 1 / (distance * 8);
      const direction = new THREE.Vector3()
        .subVectors(mouse.current, dot)
        .normalize();
      const pull = direction.multiplyScalar(strength);
      const target = originalPos.clone().add(pull);
      dot.lerp(target, 0.04);
    } else {
      dot.lerp(originalPos, 0.01);
    }
  });

  return (
    <mesh ref={ref} position={position}>
      <sphereGeometry args={[0.07, 8, 8]} />
      <meshBasicMaterial color={color} />
    </mesh>
  );
};

const DotCloud = ({ mouse, count = 1000, space = 20 }) => {
  const dots = useMemo(() => {
    return Array.from({ length: count }, () => ({
      position: [
        (Math.random() - 0.5) * space,
        (Math.random() - 0.5) * space,
        (Math.random() - 0.5) * space,
      ],
      color: new THREE.Color(
        Math.random(),
        Math.random(),
        Math.random()
      ).getStyle(),
    }));
  }, [count, space]);

  return (
    <>
      {dots.map((dot, i) => (
        <Dot key={i} position={dot.position} color={dot.color} mouse={mouse} />
      ))}
    </>
  );
};

const Scene = () => {
  const mouse = useRef(new THREE.Vector3());
  const controlsRef = useRef();
  const [stream, setStream] = useState(null);

  const prevMouse = useRef({ x: null, y: null });

  const handlePointerMove = (e) => {
    if (
      prevMouse.current.x !== null &&
      prevMouse.current.y !== null &&
      controlsRef.current
    ) {
      const deltaX = e.clientX - prevMouse.current.x;
      const deltaY = e.clientY - prevMouse.current.y;
      const panSpeed = 0.009;

      const controls = controlsRef.current;
      const camera = controls.object;
      const target = controls.target;

      // Vector from target to camera
      const offset = new THREE.Vector3().subVectors(camera.position, target);

      // Calculate pan offsets in camera local space
      const cameraDirection = new THREE.Vector3();
      camera.getWorldDirection(cameraDirection);

      const panRight = new THREE.Vector3();
      panRight.crossVectors(camera.up, cameraDirection).normalize();

      const panUp = new THREE.Vector3();
      panUp.copy(camera.up).normalize();

      // Calculate movement vector relative to target
      const moveRight = panRight.multiplyScalar(-deltaX * panSpeed);
      const moveUp = panUp.multiplyScalar(deltaY * panSpeed);

      // Total movement
      const pan = new THREE.Vector3();
      pan.add(moveRight);
      pan.add(moveUp);

      // Move the camera position *around* the target (add pan to offset)
      offset.add(pan);

      // Update camera position
      camera.position.copy(target).add(offset);

      // Keep camera looking at the target
      camera.lookAt(target);

      controls.update();
    }

    prevMouse.current = { x: e.clientX, y: e.clientY };

    if (controlsRef.current) {
      const x = (e.clientX / window.innerWidth) * 2 - 1;
      const y = -(e.clientY / window.innerHeight) * 2 + 1;
      const camera = controlsRef.current.object;
      const vector = new THREE.Vector3(x, y, 0.5).unproject(camera);
      mouse.current.set(vector.x, vector.y, vector.z);
    }
  };

  const handlePointerLeave = () => {
    prevMouse.current = { x: null, y: null };
  };

  return (
    <div className="relative w-screen h-screen bg-black">
      <div className="absolute top-10 left-1/2 -translate-x-1/2 z-10 text-white text-4xl font-bold pointer-events-none">
        Face Recognition System
      </div>
      {stream && (
        <div className="absolute top-30 left-1/2 -translate-x-1/2 z-10">
          <CameraFeed />
          {/* <div className="text-white text-2xl font-bold mt-4">
            Camera Feed Active
          </div> */}
        </div>
      )}

      <div className="absolute top-160 left-1/2 -translate-x-1/2 z-10 flex flex-col space-x-4 text-black">
        <button
          className="bg-white px-4 py-2 rounded shadow hover:bg-gray-200 transition"
          onClick={() => setStream(!stream)}
        >
          Toggle Camera Feed
        </button>
        {/* <button
          className="bg-white px-4 mt-1 py-2 rounded shadow hover:bg-gray-200 transition"
          onClick={() => {
            setAddFace(!addFace);
            setStream(false);
          }}
        >
          Add Face
        </button> */}
        <button
          hidden
          className="bg-white px-4 py-2 rounded shadow hover:bg-gray-200 transition"
        >
          Single Mode
        </button>
        {/*<button className="bg-white px-4 py-2 rounded shadow hover:bg-gray-200 transition">
          Multi Mode
        </button> */}
      </div>

      <Canvas
        camera={{ position: [0, 0, 15], fov: 75 }}
        onPointerMove={handlePointerMove}
        onPointerLeave={handlePointerLeave}
      >
        <ambientLight />
        <DotCloud mouse={mouse} count={1000} space={40} />
        <OrbitControls
          ref={controlsRef}
          enableZoom={false}
          enableRotate
          enablePan
        />
      </Canvas>
    </div>
  );
};

export default Scene;
