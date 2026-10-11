import numpy as np
from pygame import Vector3 as Vec3


class Camera:
    def __init__(self, pos, pitch, yaw):
        self.pos = pos
        self.pitch = 0
        self.yaw = 0

        self.speed = 5
        self.sensitivity = 0.001

    def GenerateMatrix(self, xp):
        cy, sy = np.cos(self.yaw), np.sin(self.yaw)
        cp, sp = np.cos(self.pitch), np.sin(self.pitch)

        Ry = np.array([
            [cy,0,sy],
            [0,1,0],
            [-sy,0,cy]
                       ])

        Rx = np.array([
            [1,0,0],
            [0,cp,sp],
            [0,-sp,cp]
        ])

        R = xp.asarray(Ry @ Rx)
        return R

    def moveForward(self, dt):
        matrix = self.GenerateMatrix(np)
        self.pos += matrix @ Vec3(0,0,1)*dt*self.speed
    def moveBackward(self, dt):
        matrix = self.GenerateMatrix(np)
        self.pos += matrix @ Vec3(0,0,-1)*dt*self.speed

    def moveLeft(self, dt):
        matrix = self.GenerateMatrix(np)
        self.pos += matrix @ Vec3(-1, 0, 0) * dt * self.speed
    def moveRight(self, dt):
        matrix = self.GenerateMatrix(np)
        self.pos += matrix @ Vec3(1, 0, 0) * dt * self.speed

    def moveUp(self, dt):
        self.pos += Vec3(0, 1, 0) * dt * self.speed
    def moveDown(self, dt):
        self.pos += Vec3(0, -1, 0) * dt * self.speed

    def changeYaw(self, yaw):
        self.yaw += yaw * self.sensitivity
    def changePitch(self, pitch):
        self.pitch -= pitch * self.sensitivity