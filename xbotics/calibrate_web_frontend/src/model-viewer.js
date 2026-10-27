import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { STLLoader } from 'three/addons/loaders/STLLoader.js';
import URDFLoader from 'urdf-loader';
import { JOINT_ORDER } from './joints.js';

// 从臂用星禾套件的仿真模型（Xbotics-SO101-Sim 的 kit_v1_so101）：连底板、型材架、顶视与腕部相机支架一起显示；
// 套件的主臂就是官方 SO101 主臂，沿用官方模型。
const MODEL_URLS = {
  follower: new URL('model/kit/kit_v1_so101.urdf', document.baseURI).href,
  leader: new URL('model/so101_leader_new_calib.urdf', document.baseURI).href,
};

const STATUS_COLORS = {
  pending: 0x3f4b5f,
  active: 0x38bdf8,
  pass: 0x22c55e,
  error: 0xef4444,
};

const TARGET_COLOR = new THREE.Color(0x38bdf8);

function cloneMaterials(root) {
  root.traverse((object) => {
    if (!object.isMesh) return;
    const source = Array.isArray(object.material) ? object.material : [object.material];
    const cloned = source.map((material) => material.clone());
    object.material = Array.isArray(object.material) ? cloned : cloned[0];
    object.castShadow = true;
    object.receiveShadow = true;
    object.userData.baseColor = cloned[0]?.color?.clone?.() ?? new THREE.Color(0xb8c0cc);
  });
}

function disposeObject(root) {
  if (!root) return;
  root.traverse((object) => {
    if (!object.isMesh) return;
    object.geometry?.dispose?.();
    const materials = Array.isArray(object.material) ? object.material : [object.material];
    materials.forEach((material) => material?.dispose?.());
  });
  root.parent?.remove(root);
}

function isMotorMesh(mesh) {
  const materials = Array.isArray(mesh.material) ? mesh.material : [mesh.material];
  return materials.some((material) => material?.name === 'sts3215') || mesh.name.includes('sts3215');
}

// 第一个关节以下的部分会动；其余（底座、底板、型材架、顶视相机支架）是台子
function movingMeshes(robot) {
  const meshes = new Set();
  robot.joints?.[JOINT_ORDER[0]]?.traverse((object) => {
    if (object.isMesh) meshes.add(object);
  });
  return meshes;
}

function collectLinkMeshes(link, motors, assigned) {
  const meshes = [];
  const visit = (node) => {
    if (node !== link && node.isURDFLink) return;
    if (node.isMesh && isMotorMesh(node) === motors && !assigned.has(node)) {
      meshes.push(node);
      assigned.add(node);
    }
    for (const child of node.children) visit(child);
  };
  visit(link);
  return meshes;
}

export class CalibrationViewer {
  constructor(container, { onStatus } = {}) {
    this.container = container;
    this.onStatus = onStatus || (() => {});
    this.snapshot = null;
    this.sessionId = null;
    this.actualRobot = null;
    this.targetRobot = null;
    this.childLinks = {};
    this.actualMeshes = {};
    this.actualPose = {};
    this.targetTime = 0;
    this.playing = true;
    this.ready = false;
    this.loadToken = 0;

    this._cameraGoal = null;
    this._controlsGoal = null;
    this._defaultCamera = null;
    this._defaultTarget = null;

    this._initScene();
    this._bindResize();
    this._startLoop();
  }

  _initScene() {
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100);
    this.camera.up.set(0, 0, 1);
    this.camera.position.set(0.75, -0.7, 0.62);

    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.1;
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.renderer.domElement.className = 'viewer-canvas';
    this.container.prepend(this.renderer.domElement);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.minDistance = 0.18;
    this.controls.maxDistance = 3;
    this.controls.target.set(0, 0, 0.22);
    this.controls.update();

    const ambient = new THREE.AmbientLight(0xffffff, 1.35);
    this.scene.add(ambient);

    const key = new THREE.DirectionalLight(0xfff4e0, 2.6);
    key.position.set(0.9, -1.1, 1.6);
    key.castShadow = true;
    key.shadow.mapSize.set(1024, 1024);
    key.shadow.camera.near = 0.1;
    key.shadow.camera.far = 6;
    key.shadow.camera.left = -1;
    key.shadow.camera.right = 1;
    key.shadow.camera.top = 1;
    key.shadow.camera.bottom = -1;
    this.scene.add(key);

    const fill = new THREE.DirectionalLight(0x8ab4ff, 1.1);
    fill.position.set(-1.2, 0.7, 0.9);
    this.scene.add(fill);

    const rim = new THREE.PointLight(0x38bdf8, 1.4, 4);
    rim.position.set(0, 1, 1.1);
    this.scene.add(rim);

    const grid = new THREE.GridHelper(1.6, 32, 0x223047, 0x172033);
    grid.rotation.x = Math.PI / 2;
    grid.position.z = -0.002;
    grid.material.transparent = true;
    grid.material.opacity = 0.65;
    this.scene.add(grid);

    this.clock = new THREE.Clock();
  }

  _bindResize() {
    this.resizeObserver = new ResizeObserver(() => this._resize());
    this.resizeObserver.observe(this.container);
    this._resize();
  }

  _resize() {
    const width = this.container.clientWidth || 320;
    const height = this.container.clientHeight || 320;
    this.renderer.setSize(width, height, false);
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
  }

  _startLoop() {
    const tick = () => {
      this._raf = requestAnimationFrame(tick);
      const dt = Math.min(this.clock.getDelta(), 0.05);
      this._stepCamera(dt);
      this._stepRobots(dt);
      this.controls.update();
      this.renderer.render(this.scene, this.camera);
    };
    this._raf = requestAnimationFrame(tick);
  }

  _stepCamera(dt) {
    if (this._cameraGoal) {
      const t = 1 - Math.exp(-7 * dt);
      this.camera.position.lerp(this._cameraGoal, t);
      if (this.camera.position.distanceToSquared(this._cameraGoal) < 1e-8) this._cameraGoal = null;
    }
    if (this._controlsGoal) {
      const t = 1 - Math.exp(-7 * dt);
      this.controls.target.lerp(this._controlsGoal, t);
      if (this.controls.target.distanceToSquared(this._controlsGoal) < 1e-8) this._controlsGoal = null;
    }
  }

  _stepRobots(dt) {
    if (this.actualRobot) {
      const joints = this.snapshot?.joints || {};
      for (const name of JOINT_ORDER) {
        const joint = this.actualRobot.joints?.[name];
        const state = joints[name];
        if (!joint || !state || state.angle === null || state.angle === undefined) continue;
        const desired = this._actualJointValue(name, Number(state.angle));
        if (!Number.isFinite(desired)) continue;
        const current = Number.isFinite(this.actualPose[name]) ? this.actualPose[name] : 0;
        const next = current + (desired - current) * (1 - Math.exp(-12 * dt));
        this.actualPose[name] = next;
        joint.setJointValue(next);
      }
    }

    if (!this.targetRobot) return;
    const phase = this.snapshot?.phase;
    const visible = phase === 'middle' || phase === 'ranges';
    this.targetRobot.visible = visible;
    if (!visible) return;

    if (this.playing) this.targetTime += dt;
    const activeJoint = this.snapshot?.active_joint;

    for (const name of JOINT_ORDER) {
      const joint = this.targetRobot.joints?.[name];
      if (!joint) continue;
      let value = 0;
      if (phase === 'ranges' && name === activeJoint) {
        const lower = Number(joint.limit?.lower ?? -Math.PI);
        const upper = Number(joint.limit?.upper ?? Math.PI);
        const span = upper - lower;
        const wave = (1 - Math.cos(this.targetTime * (Math.PI * 2) / 2.8)) / 2;
        value = lower + span * wave;
      }
      joint.setJointValue(value);
    }
  }

  _actualJointValue(name, value) {
    const joint = this.actualRobot?.joints?.[name];
    if (!joint || !Number.isFinite(value)) return null;
    if (name === 'gripper') {
      const lower = Number(joint.limit?.lower ?? 0);
      const upper = Number(joint.limit?.upper ?? 1);
      const ratio = Math.min(Math.max(value, 0), 100) / 100;
      return lower + (upper - lower) * ratio;
    }
    return (value * Math.PI) / 180;
  }

  async load(target = this.modelTarget || 'follower') {
    const modelTarget = target === 'leader' ? 'leader' : 'follower';
    this.modelTarget = modelTarget;
    const modelUrl = MODEL_URLS[modelTarget];
    const token = ++this.loadToken;
    this._setStatus('loading', `正在加载${modelTarget === 'leader' ? ' SO101 主臂' : '星禾套件从臂'}模型与网格…`);
    this._disposeRobots();
    this.ready = false;
    this.sessionId = null;

    try {
      const [actual, target] = await Promise.all([this._loadRobot(modelUrl), this._loadRobot(modelUrl)]);
      if (token !== this.loadToken) {
        disposeObject(actual);
        disposeObject(target);
        return;
      }

      cloneMaterials(actual);
      cloneMaterials(target);
      this._configureActual(actual);
      this._configureTarget(target);

      this.actualRobot = actual;
      this.targetRobot = target;
      this.scene.add(actual);
      this.scene.add(target);
      this.targetRobot.visible = false;

      this._mapJoints(actual);
      this._frameRobot();
      this.ready = true;
      this._setStatus('ready', '模型已加载');
      this._applyHighlights();
      if (this.snapshot) this.setSnapshot(this.snapshot, { force: true });
    } catch (error) {
      if (token !== this.loadToken) return;
      console.error('SO101 模型加载失败', error);
      this._setStatus('error', '模型加载失败，表单操作不受影响。');
    }
  }

  _loadRobot(modelUrl) {
    return new Promise((resolve, reject) => {
      let pending = 0;
      let parsed = false;
      let robot = null;
      let settled = false;

      const settle = () => {
        if (!settled && parsed && pending === 0) {
          settled = true;
          resolve(robot);
        }
      };

      const loader = new URDFLoader();
      loader.parseCollision = false;
      loader.loadMeshCb = (path, manager, material, done) => {
        pending += 1;
        const stlLoader = new STLLoader(manager);
        stlLoader.load(
          path,
          (geometry) => {
            geometry.computeVertexNormals();
            const mesh = new THREE.Mesh(
              geometry,
              material || new THREE.MeshStandardMaterial({ color: 0xb8c0cc, roughness: 0.55 }),
            );
            pending -= 1;
            done(mesh);
            settle();
          },
          undefined,
          (error) => {
            pending -= 1;
            done(null, error);
            if (!settled) {
              settled = true;
              reject(error);
            }
          },
        );
      };

      loader.load(
        modelUrl,
        (model) => {
          robot = model;
          parsed = true;
          settle();
        },
        undefined,
        (error) => {
          if (!settled) {
            settled = true;
            reject(error);
          }
        },
      );
    });
  }

  _configureActual(robot) {
    robot.name = 'so101_actual';
  }

  _configureTarget(robot) {
    robot.name = 'so101_target';
    // 半透明的目标姿态只画会动的部分，台子不重复叠一层
    const moving = movingMeshes(robot);
    robot.traverse((object) => {
      if (!object.isMesh) return;
      if (!moving.has(object)) {
        object.visible = false;
        return;
      }
      const materials = Array.isArray(object.material) ? object.material : [object.material];
      materials.forEach((material) => {
        material.transparent = true;
        material.opacity = 0.18;
        material.depthWrite = false;
        material.side = THREE.DoubleSide;
        if (material.color) material.color.lerp(TARGET_COLOR, 0.45);
        if (material.emissive) {
          material.emissive.copy(TARGET_COLOR);
          material.emissiveIntensity = 0.22;
        }
      });
      object.castShadow = false;
      object.receiveShadow = false;
      object.renderOrder = 2;
    });
  }

  _mapJoints(robot) {
    this.childLinks = {};
    this.actualMeshes = {};
    const assigned = new Set();
    for (const name of JOINT_ORDER) {
      const joint = robot.joints?.[name];
      if (!joint) continue;
      const childLink = joint.children.find((child) => child.isURDFLink);
      if (!childLink) continue;
      this.childLinks[name] = childLink;
      const parentLink = joint.parent?.isURDFLink ? joint.parent : null;
      const structureLink = name === 'gripper' && parentLink ? parentLink : childLink;
      const structureMeshes = name === 'wrist_roll' ? [] : collectLinkMeshes(structureLink, false, assigned);
      // 夹爪会动的那一片（从臂的动颚、主臂的扳机）也归夹爪，跟着它的状态变色，不当台子
      const jawMeshes = name === 'gripper' ? collectLinkMeshes(childLink, false, assigned) : [];
      this.actualMeshes[name] = [
        ...structureMeshes,
        ...jawMeshes,
        ...(parentLink ? collectLinkMeshes(parentLink, true, assigned) : []),
      ];
    }

    this.staticMeshes = [];
    robot.traverse((object) => {
      if (object.isMesh && !assigned.has(object)) this.staticMeshes.push(object);
    });
  }

  _frameRobot() {
    if (!this.actualRobot) return;
    const box = new THREE.Box3().setFromObject(this.actualRobot);
    if (box.isEmpty()) return;
    const center = box.getCenter(new THREE.Vector3());
    const radius = Math.max(box.getSize(new THREE.Vector3()).length() * 0.5, 0.18);
    const direction = new THREE.Vector3(0.9, -1, 0.68).normalize();
    const verticalFov = THREE.MathUtils.degToRad(this.camera.fov);
    const horizontalFov = 2 * Math.atan(Math.tan(verticalFov / 2) * this.camera.aspect);
    const fitFov = Math.min(verticalFov, horizontalFov);
    const distance = (radius / Math.sin(fitFov / 2)) * 0.95;
    const position = center.clone().addScaledVector(direction, distance);
    this._defaultCamera = position;
    this._defaultTarget = center.clone();
    this.camera.position.copy(position);
    this.controls.target.copy(center);
    this.controls.update();
  }

  setSnapshot(snapshot, { force = false } = {}) {
    if (!snapshot) {
      this.snapshot = null;
      this.targetRobot && (this.targetRobot.visible = false);
      return;
    }

    const modelTarget = snapshot.target === 'leader' ? 'leader' : 'follower';
    if (modelTarget !== this.modelTarget) this.load(modelTarget);

    if (snapshot.id !== this.sessionId) {
      this.sessionId = snapshot.id;
      this.targetTime = 0;
      for (const name of JOINT_ORDER) {
        this.actualPose[name] = 0;
        this.actualRobot?.joints?.[name]?.setJointValue(0);
      }
    } else if (!force && this.snapshot && snapshot.seq < this.snapshot.seq) {
      return;
    }

    this.snapshot = snapshot;
    this._applyHighlights();
  }

  _applyHighlights() {
    if (!this.actualRobot) return;
    const joints = this.snapshot?.joints || {};
    for (const name of JOINT_ORDER) {
      const status = joints[name]?.status || 'pending';
      const color = new THREE.Color(STATUS_COLORS[status] || STATUS_COLORS.pending);
      const strong = status === 'active' || status === 'error';
      for (const mesh of this.actualMeshes[name] || []) {
        const material = Array.isArray(mesh.material) ? mesh.material[0] : mesh.material;
        if (!material) continue;
        const base = mesh.userData.baseColor || new THREE.Color(0xb8c0cc);
        if (material.color) {
          const blend = status === 'pending' ? 0.94 : 0.96;
          material.color.copy(base).lerp(color, blend);
        }
        if (material.emissive) {
          material.emissive.copy(color);
          material.emissiveIntensity = strong ? 0.34 : status === 'pass' ? 0.18 : 0.05;
        }
      }
    }

    // 台子、相机支架这些不参与标定的部分保留本色，只稍微压暗，让关节的状态色更显眼
    const neutral = new THREE.Color(0x64748b);
    for (const mesh of this.staticMeshes || []) {
      const material = Array.isArray(mesh.material) ? mesh.material[0] : mesh.material;
      if (!material) continue;
      const base = mesh.userData.baseColor || new THREE.Color(0xb8c0cc);
      if (material.color) material.color.copy(base).lerp(neutral, 0.3);
      if (material.emissive) {
        material.emissive.setHex(0x000000);
        material.emissiveIntensity = 0;
      }
    }
  }

  focusJoint(name) {
    const link = this.childLinks[name];
    if (!link) return false;
    const box = new THREE.Box3().setFromObject(link);
    if (box.isEmpty()) return false;
    const center = box.getCenter(new THREE.Vector3());
    const radius = Math.max(box.getSize(new THREE.Vector3()).length() * 0.5, 0.05);
    const direction = new THREE.Vector3(0.55, -1, 0.42).normalize();
    this._cameraGoal = center.clone().addScaledVector(direction, Math.min(Math.max(radius * 3.4, 0.26), 0.75));
    this._controlsGoal = center.clone();
    return true;
  }

  resetView() {
    if (!this._defaultCamera || !this._defaultTarget) {
      this._frameRobot();
      return;
    }
    this._cameraGoal = this._defaultCamera.clone();
    this._controlsGoal = this._defaultTarget.clone();
  }

  togglePlaying() {
    this.playing = !this.playing;
    return this.playing;
  }

  setPlaying(value) {
    this.playing = Boolean(value);
    return this.playing;
  }

  clear() {
    this.snapshot = null;
    this.sessionId = null;
    if (this.targetRobot) this.targetRobot.visible = false;
    for (const name of JOINT_ORDER) {
      this.actualPose[name] = 0;
      this.actualRobot?.joints?.[name]?.setJointValue(0);
    }
  }

  _setStatus(status, text) {
    this.onStatus?.({ status, text });
  }

  _disposeRobots() {
    disposeObject(this.actualRobot);
    disposeObject(this.targetRobot);
    this.actualRobot = null;
    this.targetRobot = null;
    this.actualPose = {};
  }

  dispose() {
    cancelAnimationFrame(this._raf);
    this.resizeObserver?.disconnect();
    this.controls?.dispose();
    this._disposeRobots();
    this.renderer?.dispose();
    this.renderer?.domElement?.remove();
  }
}

export { MODEL_URLS };
