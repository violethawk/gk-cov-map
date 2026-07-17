export const GOAL_PLANE_POINTS = [
  [-3.66, 0.0],
  [3.66, 0.0],
  [-3.66, 2.44],
  [3.66, 2.44],
];

function solveLinearSystem(matrix, vector) {
  const n = vector.length;
  const augmented = matrix.map((row, i) => [...row, vector[i]]);
  for (let col = 0; col < n; col += 1) {
    let pivot = col;
    for (let row = col + 1; row < n; row += 1) {
      if (Math.abs(augmented[row][col]) > Math.abs(augmented[pivot][col])) pivot = row;
    }
    if (Math.abs(augmented[pivot][col]) < 1e-12) throw new Error("Degenerate homography points");
    [augmented[col], augmented[pivot]] = [augmented[pivot], augmented[col]];
    const divisor = augmented[col][col];
    for (let j = col; j <= n; j += 1) augmented[col][j] /= divisor;
    for (let row = 0; row < n; row += 1) {
      if (row === col) continue;
      const factor = augmented[row][col];
      for (let j = col; j <= n; j += 1) augmented[row][j] -= factor * augmented[col][j];
    }
  }
  return augmented.map((row) => row[n]);
}

export function solveHomography(pixelPoints, planePoints = GOAL_PLANE_POINTS) {
  if (pixelPoints.length !== 4 || planePoints.length !== 4) {
    throw new Error("Exactly four point correspondences are required");
  }
  const a = [];
  const b = [];
  for (let i = 0; i < 4; i += 1) {
    const [u, v] = pixelPoints[i];
    const [x, y] = planePoints[i];
    a.push([u, v, 1, 0, 0, 0, -x * u, -x * v]);
    b.push(x);
    a.push([0, 0, 0, u, v, 1, -y * u, -y * v]);
    b.push(y);
  }
  const h = solveLinearSystem(a, b);
  return [h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7], 1];
}

export function projectPoint(h, point) {
  const [u, v] = point;
  const denominator = h[6] * u + h[7] * v + h[8];
  if (Math.abs(denominator) < 1e-12) throw new Error("Point projects to infinity");
  return [
    (h[0] * u + h[1] * v + h[2]) / denominator,
    (h[3] * u + h[4] * v + h[5]) / denominator,
  ];
}
