import assert from 'node:assert/strict';
import {projectOrbit, projectOrtho, sortFarFirst} from './view.mjs';

const yaw = .65, pitch = .35, origin = [.5, .375, .375];
const forward = [Math.sin(yaw) * Math.cos(pitch), Math.sin(pitch), Math.cos(yaw) * Math.cos(pitch)];
const farPoint = origin.map((v, i) => v + .15 * forward[i]);
const nearPoint = origin.map((v, i) => v - .15 * forward[i]);
const far = projectOrbit(farPoint, yaw, pitch, origin, 100, [200, 150]);
const near = projectOrbit(nearPoint, yaw, pitch, origin, 100, [200, 150]);
assert(Math.hypot(far[0] - near[0], far[1] - near[1]) < 1e-6);
assert(far[2] > near[2]);
assert.deepEqual(sortFarFirst([{i: 'near', d: near[2]}, {i: 'far', d: far[2]}], item => item.d).map(item => item.i), ['far', 'near']);

function ortho(camera, nearP, farP) {
  const n = projectOrtho(nearP, camera, [0, 0, 0], 1, [0, 0]);
  const f = projectOrtho(farP, camera, [0, 0, 0], 1, [0, 0]);
  assert(Math.hypot(n[0] - f[0], n[1] - f[1]) < 1e-9);
  assert(f[2] > n[2]);
  return sortFarFirst([{i: 'near', d: n[2]}, {i: 'far', d: f[2]}], item => item.d).map(item => item.i);
}
assert.deepEqual(ortho('xy', [0, 0, 0], [0, 0, 1]), ['far', 'near']);
assert.deepEqual(ortho('xz', [0, 1, 0], [0, 0, 0]), ['far', 'near']);
assert.deepEqual(ortho('yz', [0, 0, 0], [1, 0, 0]), ['far', 'near']);
console.log('PASS: nearer ball paints after the farther ball');
