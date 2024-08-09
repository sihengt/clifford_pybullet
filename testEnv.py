import torch
import numpy as np
from .Terrain import Terrain

class TestEnv(Terrain):
    def __init__(self, terrainParams, physicsClientId=0):
        super().__init__(terrainParams, physicsClientId)
        self.generate()

    # generate new terrain.
    def generate(self,stepStart=2,stepWidth=10,numSteps=7,stepLength=0.5,stepHeight=0.085,rampWidth=10,rampLength=5):
        gridZ = np.zeros_like(self.gridX)
        # add steps
        nextStepStart = stepStart
        nextStepHeight = stepHeight
        for _ in range(numSteps):
            gridZ = self.addStep(nextStepStart,
                                nextStepStart+stepLength,
                                -stepWidth/2.0,
                                stepWidth/2.0,
                                nextStepHeight,
                                gridZ)
            nextStepHeight += stepHeight
            nextStepStart += stepLength

        # add goal platform
        gridZ = self.addStep(nextStepStart,torch.inf,
                            -stepWidth/2.0,
                            stepWidth/2.0,
                            nextStepHeight,
                            gridZ)

        # add ramps
        gridZ = self.addRamp(stepStart,stepStart+rampLength,
                            -stepWidth/2.0-rampWidth,
                            -stepWidth/2.0,
                            nextStepHeight,
                            gridZ)
        gridZ = self.addRamp(stepStart,stepStart+rampLength,
                            stepWidth/2.0,
                            stepWidth/2.0+rampWidth,
                            nextStepHeight,
                            gridZ)

        self.updateTerrain(gridZ)

    def addStep(self,minX,maxX,minY,maxY,height,gridZ):
        indices = self.gridX <= maxX
        indices = np.logical_and(indices,self.gridX >= minX)
        indices = np.logical_and(indices,self.gridY <= maxY)
        indices = np.logical_and(indices,self.gridY >= minY)
        gridZ[indices] = height
        return gridZ

    def addRamp(self,minX,maxX,minY,maxY,height,gridZ):
        indices = self.gridX <= maxX
        indices = np.logical_and(indices,self.gridX >= minX)
        indices = np.logical_and(indices,self.gridY <= maxY)
        indices = np.logical_and(indices,self.gridY >= minY)
        rampHeight = (self.gridX-minX)*height/(maxX-minX)
        gridZ[indices] = rampHeight[indices]
        gridZ = self.addStep(maxX,torch.inf,minY,maxY,height,gridZ)
        return gridZ
        
