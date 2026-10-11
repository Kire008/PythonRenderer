from Scene.Objects.Template import ObjectTemp
import numpy as np

class Triangle(ObjectTemp):
    def __init__(self, position, v0, v1, v2, color):
        super().__init__(color)
        self.position = np.array(position, dtype=np.float64)
        self.local = np.array([v0, v1, v2], dtype=np.float64)   # (3, 3)
        self._Build()

    def _Build(self):
        """Recompute world-space data. Call again after changing position."""
        w = self.local + self.position
        self.v0, self.v1, self.v2 = w
        self.e1 = self.v1 - self.v0
        self.e2 = self.v2 - self.v0
        n = np.cross(self.e1, self.e2)
        self.normal = n / np.linalg.norm(n)


    def Intersect(self, origins, d, xp):
        v0 = xp.asarray(self.v0)
        e1 = xp.asarray(self.e1)
        e2 = xp.asarray(self.e2)

        # h = d x e2
        h = xp.cross(d, e2)
        a = xp.sum(e1 * h, axis=-1)

        parallel = xp.abs(a) < 1e-8
        a_safe = xp.where(parallel, 1.0, a)
        f = 1.0 / a_safe

        s = origins - v0
        u = f * xp.sum(s * h, axis=-1)

        q = xp.cross(s, e1)
        v = f * xp.sum(d * q, axis=-1)

        t = f * xp.sum(e2 * q, axis=-1)

        valid = (~parallel) & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 1e-6)
        return xp.where(valid, t, xp.inf)

    def Normal(self, points, xp):
        n = xp.asarray(self.normal)
        return xp.broadcast_to(n, points.shape)