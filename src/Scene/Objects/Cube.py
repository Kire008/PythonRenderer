from Scene.Objects.Template import ObjectTemp, Cross
from Scene.Objects.Triangle import Triangle
import numpy as np

class Cube(ObjectTemp):
    FACES = [
        (0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4),
        (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7),
    ]

    def __init__(self, position, verts, color):
        super().__init__(color)
        self.position = np.array(position, dtype=np.float64)
        self.local = np.array(verts, dtype=np.float64)
        self._Build()

    def _Build(self):
        w = self.local + self.position
        tris = []
        for a, b, c, d in self.FACES:
            tris.append(Triangle((0, 0, 0), w[a], w[b], w[c], self.color))
            tris.append(Triangle((0, 0, 0), w[a], w[c], w[d], self.color))

        # Batched triangle data, shape (12, 3) each
        self.V0 = np.stack([t.v0 for t in tris])
        self.E1 = np.stack([t.e1 for t in tris])
        self.E2 = np.stack([t.e2 for t in tris])
        self.N = np.stack([t.normal for t in tris])
        self._devxp = None  # invalidate GPU cache

    def SetPosition(self, position):
        self.position = np.array(position, dtype=np.float64)
        self._Build()

    def _Dev(self, xp):
        """Upload constants once, not every frame."""
        if self._devxp is not xp:
            self._dev = [xp.asarray(a, dtype=xp.float32)
                         for a in (self.V0, self.E1, self.E2, self.N)]
            self._devxp = xp
        return self._dev

    def Intersect(self, origins, d, xp):
        H, W = d.shape[:2]
        V0, E1, E2, _ = self._Dev(xp)
        v0, e1, e2 = V0[:, None, :], E1[:, None, :], E2[:, None, :]  # (T, 1, 3)
        o = origins.reshape(1, -1, 3)
        dd = d.reshape(1, -1, 3)

        h = Cross(dd, e2, xp)  # (T, N, 3)
        a = xp.sum(e1 * h, axis=-1)  # (T, N)
        parallel = xp.abs(a) < 1e-8
        f = 1.0 / xp.where(parallel, 1.0, a)

        s = o - v0
        u = f * xp.sum(s * h, axis=-1)
        q = Cross(s, e1, xp)
        v = f * xp.sum(dd * q, axis=-1)
        t = f * xp.sum(e2 * q, axis=-1)

        valid = (~parallel) & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 1e-6)
        return xp.where(valid, t, xp.inf).min(axis=0).reshape(H, W)

    def Normal(self, points, xp):
        V0, _, _, N = self._Dev(xp)
        dist = xp.abs(xp.sum((points[None, :, :] - V0[:, None, :]) * N[:, None, :], axis=-1))
        return N[xp.argmin(dist, axis=0)]

    @staticmethod
    def BoxVerts(w=1.0, h=1.0, d=1.0):
        x, y, z = w / 2, h / 2, d / 2
        return [
            (-x, -y, -z), (x, -y, -z), (x, -y, z), (-x, -y, z),
            (-x, y, -z), (x, y, -z), (x, y, z), (-x, y, z),
        ]