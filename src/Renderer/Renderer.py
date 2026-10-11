import pygame as pg
pg.init()
import numpy as np
import cupy as cp


class Renderer:
    def __init__(self, WIN, width, height, x=0, y=0, hardwareAcceleration=True, resolution=1):
        self.WIN = WIN
        self.width = width
        self.height = height
        self.x = x
        self.y = y
        self.resolution = resolution

        self.scenes = []
        self.currentScene = None

        # Numpy / Cupy
        self.xp = self.ToggleHardwareAcceleration(hardwareAcceleration)


    def ToggleHardwareAcceleration(self, hardwareAcceleration):
        if not hardwareAcceleration: return np
        if hardwareAcceleration: return cp


    def UploadScene(self, Scene):
        self.scenes = Scene
        self.currentScene = Scene


    def Display(self):
        """
        Create pygame surface and display
        :return:
        """

        surface = self.CreateSurface()
        self.WIN.blit(surface, (self.x, self.y))


    def CreateSurface(self):
        """
        Create surface based on current scene
        :return:
        """

        width = int(self.width * self.resolution)
        height = int(self.height * self.resolution)

        xp = self.xp

        # Calculate pixels
        pixels = self.CalculatePixels(width,height)
        # Cuda fix
        if xp is cp: pixels = cp.asnumpy(pixels)


        # Fix roation
        pixels = pixels.swapaxes(0, 1)
        surface = pg.surfarray.make_surface(pixels)
        return pg.transform.scale(surface, (self.width, self.height))


    def CalculatePixels(self, width, height):
        xp = self.xp

        origins, directions = self.GenerateRays(width, height)

        closest_t = xp.full(directions.shape[:2], xp.inf)
        closest_idx = xp.full(directions.shape[:2], -1, dtype=xp.int32)

        for i, obj in enumerate(self.currentScene.objects):
            t = obj.Intersect(origins, directions, xp)
            nearer = t < closest_t
            closest_t = xp.where(nearer, t, closest_t)
            closest_idx = xp.where(nearer, i, closest_idx)


        # Hit points
        hit = closest_idx >= 0
        safe_t = xp.where(hit, closest_t, 0.0)
        points = origins + directions * safe_t[..., None]

        # Set background
        color = xp.empty_like(directions)
        color[:] = xp.asarray(self.currentScene.background, dtype=xp.float64)


        ambient = 0.1
        for i, obj in enumerate(self.currentScene.objects):
            mask = closest_idx == i
            if not xp.any(mask): continue

            normals = obj.Normal(points[mask], xp)

            facing = xp.sum(normals * directions[mask], axis=-1, keepdims=True)
            normals = xp.where(facing > 0, -normals, normals)

            light_total = xp.full(normals.shape[0], ambient)
            for light in self.currentScene.lightsources:
                L = -xp.asarray(light.direction, dtype=xp.float64)
                L = L / xp.linalg.norm(L)
                diffuse = xp.clip(xp.sum(normals * L, axis=-1), 0, 1)
                light_total += diffuse * light.intensity

            base = xp.asarray(obj.color, dtype=xp.float64)
            color[mask] = xp.clip(light_total, 0, 1)[:, None] * base


        pixels = xp.clip(color, 0, 255).astype(xp.uint8)
        return pixels



    def GenerateRays(self, width, height):

        xp = self.xp

        x = xp.arange(width)
        y = xp.arange(height)

        px,py = xp.meshgrid(x,y)

        # Normalize pixels
        px = (2 * (px +0.5) / width) - 1
        py = 1 - (2 * (py +0.5) / height)

        px *= width / height

        directions = xp.stack(
            [px,py,xp.ones_like(px)],
            axis=-1
        )

        # Normalize directions
        directions /= xp.linalg.norm(
            directions, axis=-1, keepdims=True
        )

        R = self.currentScene.currentCamera.GenerateMatrix(xp)
        directions = directions @ R.T
        origins = xp.broadcast_to(
            xp.asarray(self.currentScene.currentCamera.pos, dtype=xp.float64), directions.shape
        ).copy()

        return origins, directions


    def Update(self):

        self.Display()