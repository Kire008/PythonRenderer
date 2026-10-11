from Scene.Objects.Template import ObjectTemp

class Sphere(ObjectTemp):
    def __init__(self, center, radius, color):
        super().__init__(color)
        self.center = center
        self.radius = radius

    def Intersect(self, origins, d, xp):
        center = xp.asarray(self.center, dtype=xp.float32)
        oc = origins - center
        b = xp.sum(oc * d, axis=-1)
        c = xp.sum(oc * oc, axis=-1) - self.radius ** 2
        disc = b * b - c
        t = -b - xp.sqrt(xp.maximum(disc, 0))
        return xp.where((disc > 0) & (t > 0), t, xp.inf)

    def Normal(self, points, xp):
        center = xp.asarray(self.center, dtype=xp.float32)
        return (points - center) / self.radius