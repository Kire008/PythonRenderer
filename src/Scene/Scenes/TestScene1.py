from Objects.Cube import Cube
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

cube = Cube(Vec3(5,2,5), Cube.BoxVerts(2,1,1), (200,0,200))
scene.AddObject(cube)
cube = Cube(Vec3(5,3,5), Cube.BoxVerts(2,1,1), (0,200,20))
scene.AddObject(cube)
cube = Cube(Vec3(5,4,5), Cube.BoxVerts(2,1,1), (200,200,0))
scene.AddObject(cube)
cube = Cube(Vec3(5,5,5), Cube.BoxVerts(2,1,1), (200,200,0))
scene.AddObject(cube)
cube = Cube(Vec3(5,6,5), Cube.BoxVerts(2,1,1), (200,200,0))
scene.AddObject(cube)
cube = Cube(Vec3(5,7,5), Cube.BoxVerts(2,1,1), (200,200,0))
scene.AddObject(cube)
cube = Cube(Vec3(5,8,5), Cube.BoxVerts(2,1,1), (200,200,0))
scene.AddObject(cube)
cube = Cube(Vec3(5,9,5), Cube.BoxVerts(2,1,1), (200,200,0))
scene.AddObject(cube)
cube = Cube(Vec3(5,10,5), Cube.BoxVerts(2,1,1), (200,200,0))
scene.AddObject(cube)
cube = Cube(Vec3(5,11,5), Cube.BoxVerts(2,1,1), (200,200,0))
scene.AddObject(cube)
cube = Cube(Vec3(5,12,5), Cube.BoxVerts(2,1,1), (200,200,0))
scene.AddObject(cube)
cube = Cube(Vec3(5,13,5), Cube.BoxVerts(2,1,1), (200,200,0))
scene.AddObject(cube)


scene.AddCamera(Camera(Vec3(0,0,0), 0,0))
scene.AddLightSource(LightSource(Vec3(0,0,0), Vec3(1,-1,1)))