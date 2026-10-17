// Depth increases into the screen. Draw larger depth first so a nearer ball
// covers the one behind it.

export function projectOrbit(p, yaw, pitch, origin, scale, center) {
  const x = p[0] - origin[0], y = p[1] - origin[1], z = (p[2] ?? 0) - origin[2];
  const a = x * Math.cos(yaw) - z * Math.sin(yaw);
  const b = x * Math.sin(yaw) + z * Math.cos(yaw);
  const depth = y * Math.sin(pitch) + b * Math.cos(pitch);
  return [center[0] + a * scale, center[1] + (y * Math.cos(pitch) - b * Math.sin(pitch)) * scale, depth];
}

export function projectOrtho(p, camera, origin, scale, center) {
  const x = p[0] - origin[0], y = p[1] - origin[1], z = (p[2] ?? 0) - origin[2];
  if (camera === 'xy') return [center[0] + x * scale, center[1] + y * scale, z];
  if (camera === 'xz') return [center[0] + x * scale, center[1] + z * scale, -y];
  if (camera === 'yz') return [center[0] + y * scale, center[1] + z * scale, x];
  throw new Error('unknown camera ' + camera);
}

export function sortFarFirst(items, depthOf) {
  return items.slice().sort((a, b) => depthOf(b) - depthOf(a));
}
