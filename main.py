import time

import pygame as pg
from src.Renderer.Renderer import Renderer
from src.Scene.Scenes.TestScene1 import scene as TestScene1
class Application:
    def __init__(self):
        self.running = True
        self.WIDTH = 1600
        self.HEIGHT = 1000
        self.resolution = 0.2
        self.hardwareAcceleration = True

        self.WIN = pg.display.set_mode((self.WIDTH, self.HEIGHT))
        pg.display.set_caption("3D Renderer")

        self.clock = pg.time.Clock()
        self.FPS = 60

        self.Renderer = Renderer(self.WIN, width=self.WIDTH, height=self.HEIGHT, resolution=self.resolution, hardwareAcceleration=self.hardwareAcceleration)

        # Test a scene
        self.Renderer.UploadScene(TestScene1)

        self.clickDelay = time.time()

        self.UpDelay = time.time()
        self.DownDelay = time.time()

        pg.mouse.set_visible(False)
        pg.event.set_grab(True)


    def HandleEvents(self):
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.running = False

            if event.type == pg.KEYDOWN:
                if event.key == pg.K_ESCAPE:
                    self.running = False


    def HandleInput(self, dt):
        keys = pg.key.get_pressed()

        # Move camera
        if keys[pg.K_w]:
            self.Renderer.currentScene.currentCamera.moveForward(dt)
        if keys[pg.K_s]:
            self.Renderer.currentScene.currentCamera.moveBackward(dt)
        if keys[pg.K_a]:
            self.Renderer.currentScene.currentCamera.moveLeft(dt)
        if keys[pg.K_d]:
            self.Renderer.currentScene.currentCamera.moveRight(dt)
        if keys[pg.K_SPACE]:
            self.Renderer.currentScene.currentCamera.moveUp(dt)
        if keys[pg.K_LCTRL]:
            self.Renderer.currentScene.currentCamera.moveDown(dt)


        # Change hardwareacceleration
        if keys[pg.K_1] and time.time()-self.clickDelay > 1:
            self.clickDelay = time.time()
            self.hardwareAcceleration = (not self.hardwareAcceleration)
            self.Renderer.xp = self.Renderer.ToggleHardwareAcceleration(self.hardwareAcceleration)


        if keys[pg.K_UP] and time.time()-self.UpDelay > 0.2:
            self.UpDelay = time.time()
            self.resolution *= 2
            self.resolution = min(1, self.resolution)
            self.Renderer.resolution = self.resolution
        if keys[pg.K_DOWN] and time.time()-self.DownDelay > 0.2 :
            self.DownDelay = time.time()
            self.resolution /= 2
            self.Renderer.resolution = self.resolution



        # Mouse input
        dx,dy = pg.mouse.get_rel()
        self.Renderer.currentScene.currentCamera.changeYaw(dx)
        self.Renderer.currentScene.currentCamera.changePitch(dy)


    def Update(self):
        while self.running:
            dt = self.clock.tick(self.FPS) / 1000.0

            self.HandleEvents()

            self.HandleInput(dt)

            self.Renderer.Update()


            # Show FPS:
            font = pg.font.SysFont('Arial', 20)
            text = f"FPS: {self.clock.get_fps():.2f} | HardwareAcceleration: {self.hardwareAcceleration} | Resolution: {self.resolution}"
            label_surface = font.render(text, True, (255,255,255))
            self.WIN.blit(label_surface, (0,0))
            pg.display.flip()


        pg.quit()


if __name__ == "__main__":
    app = Application()
    app.Update()