import imageio, numpy as np, pybullet as p
from datetime import datetime

class CameraRecorder:
    """
    Writes the PyBullet debug-visualizer view to an mp4, one frame per grab().
    Pass it to SimController(recorder=...) to grab after every physics step. Needs the GUI.
    """
    def __init__(self, fname=None, fps=240, res=(1024, 768), physicsClientId=0):
        if fname is None:
            ts   = datetime.now().strftime("%Y%m%d_%H%M%S") # e.g. 20250721_134859
            fname = f"sim_{ts}.mp4"
        self.writer = imageio.get_writer(fname, fps=fps, codec="libx264")
        self.W, self.H = res
        self.physicsClientId = physicsClientId

    def grab(self):
        # Use the current debug‑visualiser camera (same view you see on screen).
        _, _, viewM, projM, _, _, _, _, _, _, _, _ = p.getDebugVisualizerCamera(physicsClientId=self.physicsClientId)
        _, _, rgb, _, _ = p.getCameraImage(
            self.W, self.H, viewMatrix=viewM, projectionMatrix=projM,
            renderer=p.ER_BULLET_HARDWARE_OPENGL,  # use HW renderer when available
            physicsClientId=self.physicsClientId
        )
        frame = np.reshape(rgb, (self.H, self.W, 4))[:, :, :3]  # drop alpha
        self.writer.append_data(frame)

    def close(self):
        self.writer.close()
