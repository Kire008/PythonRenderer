def Cross(a, b, xp):
    """Manual cross product, much faster than xp.cross."""
    return xp.stack([
        a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1],
        a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2],
        a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0],
    ], axis=-1)


class ObjectTemp:
    def __init__(self, color):
        self.color = color

    def Intersect(self, origins, d, xp):
        raise NotImplementedError

    def Normal(self, points, xp):
        raise NotImplementedError