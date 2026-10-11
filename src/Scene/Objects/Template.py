class ObjectTemp:
    def __init__(self, color):
        self.color = color

    def Intersect(self, origins, d, xp):
        raise NotImplementedError

    def Normal(self, points, xp):
        raise NotImplementedError