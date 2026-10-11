class Scene:
    def __init__(self, sceneName="Scene", resolution=(100,100)):
        self.background = (0.3,0.3,0.3)
        self.sceneName = sceneName

        self.cameras = []
        self.currentCamera = None

        self.objects = []
        self.lightsources = []

        self.settings = {
            "resolution": resolution,
            "maxBounces": 2,
            "samplesPerPixel": 1
        }

    def AddObject(self, obj):
        self.objects.append(obj)
    def AddLightSource(self, lightSource):
        self.lightsources.append(lightSource)
    def AddCamera(self, camera):
        self.cameras.append(camera)
        self.currentCamera = self.cameras[-1]
