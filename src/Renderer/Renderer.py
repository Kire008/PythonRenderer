import pygame as pg
pg.init()
import numpy as np
import cupy as cp


KERNEL = r'''
extern "C" __global__
void trace(unsigned char* out, int W, int H,
           const float* cam,               // pos(3), R row-major(9)
           const float* spheres, int nS,   // cx,cy,cz,r, color(3)      = 7
           const float* tris, int nT,      // v0,e1,e2,n (12), color(3) = 15
           const float* lights, int nL,    // dir(3), intensity         = 4
           const float* bg)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;

    // Ray
    float px = ((2.f * (x + 0.5f) / W) - 1.f) * ((float)W / H);
    float py = 1.f - 2.f * (y + 0.5f) / H;
    float inv = rsqrtf(px * px + py * py + 1.f);
    float dx = px * inv, dy = py * inv, dz = inv;

    const float* R = cam + 3;
    float rx = R[0]*dx + R[1]*dy + R[2]*dz;
    float ry = R[3]*dx + R[4]*dy + R[5]*dz;
    float rz = R[6]*dx + R[7]*dy + R[8]*dz;
    float ox = cam[0], oy = cam[1], oz = cam[2];

    float best = 1e30f;
    float nx = 0, ny = 0, nz = 0;
    float cr = bg[0], cg = bg[1], cb = bg[2];
    bool hit = false;

    // Spheres
    for (int i = 0; i < nS; i++) {
        const float* s = spheres + i * 7;
        float ex = ox - s[0], ey = oy - s[1], ez = oz - s[2];
        float b = ex*rx + ey*ry + ez*rz;
        float c = ex*ex + ey*ey + ez*ez - s[3]*s[3];
        float disc = b*b - c;
        if (disc > 0.f) {
            float t = -b - sqrtf(disc);
            if (t > 1e-4f && t < best) {
                best = t; hit = true;
                nx = (ox + rx*t - s[0]) / s[3];
                ny = (oy + ry*t - s[1]) / s[3];
                nz = (oz + rz*t - s[2]) / s[3];
                cr = s[4]; cg = s[5]; cb = s[6];
            }
        }
    }

    // Triangles (Moller-Trumbore)
    for (int i = 0; i < nT; i++) {
        const float* T = tris + i * 15;
        float e1x = T[3], e1y = T[4], e1z = T[5];
        float e2x = T[6], e2y = T[7], e2z = T[8];
        float hx = ry*e2z - rz*e2y;
        float hy = rz*e2x - rx*e2z;
        float hz = rx*e2y - ry*e2x;
        float a = e1x*hx + e1y*hy + e1z*hz;
        if (fabsf(a) < 1e-8f) continue;
        float f = 1.f / a;
        float sx = ox - T[0], sy = oy - T[1], sz = oz - T[2];
        float u = f * (sx*hx + sy*hy + sz*hz);
        if (u < 0.f || u > 1.f) continue;
        float qx = sy*e1z - sz*e1y;
        float qy = sz*e1x - sx*e1z;
        float qz = sx*e1y - sy*e1x;
        float v = f * (rx*qx + ry*qy + rz*qz);
        if (v < 0.f || u + v > 1.f) continue;
        float t = f * (e2x*qx + e2y*qy + e2z*qz);
        if (t > 1e-4f && t < best) {
            best = t; hit = true;
            nx = T[9]; ny = T[10]; nz = T[11];
            cr = T[12]; cg = T[13]; cb = T[14];
        }
    }

    // Shading
    if (hit) {
        if (nx*rx + ny*ry + nz*rz > 0.f) { nx = -nx; ny = -ny; nz = -nz; }
        float total = 0.1f;
        for (int i = 0; i < nL; i++) {
            const float* l = lights + i * 4;
            float inv_l = rsqrtf(l[0]*l[0] + l[1]*l[1] + l[2]*l[2]);
            float d = -(nx*l[0] + ny*l[1] + nz*l[2]) * inv_l;
            total += fmaxf(d, 0.f) * l[3];
        }
        total = fminf(total, 1.f);
        cr *= total; cg *= total; cb *= total;
    }

    int o = (y * W + x) * 3;
    out[o]     = (unsigned char)fminf(fmaxf(cr, 0.f), 255.f);
    out[o + 1] = (unsigned char)fminf(fmaxf(cg, 0.f), 255.f);
    out[o + 2] = (unsigned char)fminf(fmaxf(cb, 0.f), 255.f);
}
'''


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

        # CPU/NumPy path cache
        self._rayKey = None

        # GPU kernel path
        self._kernel = cp.RawKernel(KERNEL, "trace") if hardwareAcceleration else None
        self._packed = None

    def ToggleHardwareAcceleration(self, hardwareAcceleration):
        if not hardwareAcceleration: return np
        if hardwareAcceleration: return cp

    def UploadScene(self, Scene):
        self.scenes.append(Scene)
        self.currentScene = Scene
        if self.xp is cp:
            self.PackScene()

    def PackScene(self):
        """Flatten scene objects into GPU arrays. Call again after moving/adding objects."""
        spheres, tris = [], []
        for o in self.currentScene.objects:
            if hasattr(o, "radius"):                      # Sphere
                spheres.append([*o.center, o.radius, *o.color])
            elif hasattr(o, "V0"):                        # Cube (batched triangles)
                for k in range(len(o.V0)):
                    tris.append([*o.V0[k], *o.E1[k], *o.E2[k], *o.N[k], *o.color])
            elif hasattr(o, "e1"):                        # Triangle
                tris.append([*o.v0, *o.e1, *o.e2, *o.normal, *o.color])

        lights = [[*l.direction, l.intensity] for l in self.currentScene.lightsources]

        def Up(data, width):
            arr = np.array(data, dtype=np.float32).reshape(-1, width)
            n = len(arr)
            if n == 0:
                arr = np.zeros((1, width), dtype=np.float32)   # avoid empty buffers
            return cp.asarray(arr), n

        sph, nS = Up(spheres, 7)
        tri, nT = Up(tris, 15)
        lig, nL = Up(lights, 4)
        bg = cp.asarray(self.currentScene.background, dtype=cp.float32)
        self._packed = (sph, nS, tri, nT, lig, nL, bg)

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

        pixels = self.CalculatePixels(width, height)
        if self.xp is cp:
            pixels = cp.asnumpy(pixels)

        # Fix rotation: surfarray expects (width, height, 3)
        pixels = np.ascontiguousarray(pixels.swapaxes(0, 1))
        surface = pg.surfarray.make_surface(pixels)
        return pg.transform.scale(surface, (self.width, self.height))

    # ------------------------------------------------------------------ GPU
    def CalculatePixels(self, width, height):
        if self.xp is not cp:
            return self.CalculatePixelsCPU(width, height)

        sph, nS, tri, nT, lig, nL, bg = self._packed
        camObj = self.currentScene.currentCamera
        R = camObj.GenerateMatrix(np).astype(np.float32).ravel()
        cam = cp.asarray(np.concatenate([np.asarray(camObj.pos, dtype=np.float32), R]))

        out = cp.empty((height, width, 3), dtype=cp.uint8)
        block = (16, 16)
        grid = ((width + 15) // 16, (height + 15) // 16)
        self._kernel(grid, block, (
            out, np.int32(width), np.int32(height), cam,
            sph, np.int32(nS), tri, np.int32(nT), lig, np.int32(nL), bg))
        return out

    # ------------------------------------------------------------------ CPU
    def CalculatePixelsCPU(self, width, height):
        xp = self.xp
        scene = self.currentScene
        objects = scene.objects

        origins, directions = self.GenerateRays(width, height)
        H, W = directions.shape[:2]

        # 1. Nearest hit per pixel
        closest_t = xp.full((H, W), xp.inf, dtype=xp.float32)
        closest_idx = xp.full((H, W), -1, dtype=xp.int32)
        for i, obj in enumerate(objects):
            t = obj.Intersect(origins, directions, xp)
            nearer = t < closest_t
            closest_t = xp.where(nearer, t, closest_t)
            closest_idx = xp.where(nearer, i, closest_idx)

        # 2. Flatten to (N, ...)
        hit = (closest_idx >= 0).reshape(-1, 1)
        idx = closest_idx.reshape(-1, 1)
        points = (origins + directions * xp.where(closest_idx >= 0, closest_t, 0)[..., None]).reshape(-1, 3)
        dirs = directions.reshape(-1, 3)

        # 3. Normals
        normals = xp.zeros_like(points)
        for i, obj in enumerate(objects):
            normals = xp.where(idx == i, obj.Normal(points, xp), normals)

        facing = xp.sum(normals * dirs, axis=-1, keepdims=True)
        normals = xp.where(facing > 0, -normals, normals)

        # 4. Lighting
        total = xp.full(points.shape[0], 0.1, dtype=xp.float32)  # ambient
        for light in scene.lightsources:
            L = -xp.asarray(light.direction, dtype=xp.float32)
            L = L / xp.linalg.norm(L)
            total += xp.clip(xp.sum(normals * L, axis=-1), 0, 1) * light.intensity

        # 5. Color
        colors = xp.asarray([o.color for o in objects], dtype=xp.float32)  # (K, 3)
        base = colors[xp.maximum(idx[:, 0], 0)]
        bg = xp.asarray(scene.background, dtype=xp.float32)

        out = xp.where(hit, xp.clip(total, 0, 1)[:, None] * base, bg)
        return xp.clip(out, 0, 255).astype(xp.uint8).reshape(H, W, 3)

    def GenerateRays(self, width, height):
        xp = self.xp
        cam = self.currentScene.currentCamera

        # Base rays only change when resolution or backend changes
        key = (width, height, xp)
        if self._rayKey != key:
            px, py = xp.meshgrid(xp.arange(width, dtype=xp.float32),
                                 xp.arange(height, dtype=xp.float32))
            px = ((2 * (px + 0.5) / width) - 1) * (width / height)
            py = 1 - (2 * (py + 0.5) / height)
            d = xp.stack([px, py, xp.ones_like(px)], axis=-1)
            self._baseDirs = d / xp.linalg.norm(d, axis=-1, keepdims=True)
            self._rayKey = key

        R = cam.GenerateMatrix(xp).astype(xp.float32)
        directions = self._baseDirs @ R.T
        origins = xp.broadcast_to(xp.asarray(cam.pos, dtype=xp.float32), directions.shape)
        return origins, directions

    def Update(self):
        self.Display()