from Scene.Scene import Scene
from Camera import Camera
from LightSource import LightSource
from Objects.Triangle import Triangle
from Objects.Sphere import Sphere
from pygame import Vector3 as Vec3

scene = Scene("First Render")
scene.background = (100,100,100)
scene.AddObject(
    Sphere(
        Vec3(0,0,5),
        1,
        (255,0,0)
    )
)

scene.AddObject(
    Sphere(
        Vec3(0,0,10),
        1,
        (0,0,255)
    )
)

scene.AddObject(
    Triangle(
        Vec3(0,2,3),
        Vec3(0,0,0),
        Vec3(1,0,0),
        Vec3(1,1,0),
        (255,255,0)
    )
)

scene.AddCamera(Camera(Vec3(0,0,0), 0,0))
scene.AddLightSource(LightSource(Vec3(0,0,0), Vec3(1,-1,1)))