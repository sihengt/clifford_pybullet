import numpy as np
from .Terrain import Terrain
from scipy.ndimage import gaussian_filter
from scipy.interpolate import griddata
from noise import pnoise2

class RandomRockyTerrain(Terrain):
    """
    This class handles the generation of random rocky terrain
    """
    # initialize terrain object
    def __init__(self, terrainParams, physicsClientId=0):
        super().__init__(terrainParams, physicsClientId)
        if self.seed:
            np.random.seed(self.seed)
        self.generate()

    # generate new terrain.
    def generate(self):
        gridZ = np.zeros_like(self.gridX)

        # Generate random blocks using Perlin Noise
        numCells = int(self.mapArea / self.terrainParams['averageAreaPerCell'])
        blockHeights = self.randomSteps(numCells)
        blockHeights = gaussian_filter(blockHeights, sigma=self.terrainParams["smoothing"])
        gridZ += blockHeights

        # add more small noise
        for i in range(len(self.terrainParams['perlinScale'])):
            smallNoise = self.perlinNoise(
                self.gridX.reshape(-1),
                self.gridY.reshape(-1),
                self.terrainParams["perlinScale"][i],
                self.terrainParams["perlinHeightScale"][i]
            )
            gridZ += smallNoise.reshape(gridZ.shape)

        # add sharp point noise
        numSharpPoints = int(self.mapArea*self.terrainParams['sharpDensity'])
        if numSharpPoints > 0:
            gridZ += self.sharpNoise(numSharpPoints,self.terrainParams['maxSharpHeight'],self.gridX,self.gridY)

        # make center flat for initial robot start position
        gridZ = self.flatten(gridZ, [0, 0])
        self.updateTerrain(gridZ - (np.max(gridZ) + np.min(gridZ)) / 2.0)

    def sharpNoise(self,N,maxHeight,gridX,gridY):
        groundPointsX = np.random.rand(N) * (gridX.max() - gridX.min()) + gridX.min()
        groundPointsY = np.random.rand(N) * (gridY.max() - gridY.min()) + gridY.min()
        groundPointsZ = np.random.rand(N) * maxHeight
        
        return griddata(
            (groundPointsX, groundPointsY),
            groundPointsZ,
            (gridX, gridY),
            method='linear',
            fill_value=0
        )

    def flatten(self, gridZ, loc):
        distFromLoc = np.sqrt((self.gridX-loc[0]) ** 2 + (self.gridY - loc[1]) **2)
        flatIndices = distFromLoc < self.terrainParams['flatRadius']
        if flatIndices.any():
            flatHeight = np.mean(gridZ[flatIndices])
            gridZ[flatIndices] = flatHeight
            distFromFlat = distFromLoc - self.terrainParams['flatRadius']
            blendIndices = distFromFlat < self.terrainParams['blendRadius']
            gridZ[blendIndices] = flatHeight + (gridZ[blendIndices] - flatHeight) \
                * distFromFlat[blendIndices] / self.terrainParams['blendRadius']
        return gridZ

    def randomSteps(self, numCells):
        centersX = np.random.uniform(size=numCells, low=self.gridX.min(), high=self.gridX.max())
        centersY = np.random.uniform(size=numCells, low=self.gridY.min(), high=self.gridY.max())
        centersZ = self.perlinNoise(
            centersX,
            centersY,
            self.terrainParams["cellPerlinScale"],
            self.terrainParams["cellHeightScale"]
        )
        
        return griddata(
            (centersX, centersY),       # points
            centersZ,                   # values
            (self.gridX, self.gridY),   # points 
            method='nearest'
        )

    def perlinNoise(self, xPoints, yPoints, cell_perlin_scale, cell_height_scale):
        random_seed = np.random.rand(2)*1000
        noise_values = [
            pnoise2(
                random_seed[0] + x * cell_perlin_scale,
                random_seed[1] + y * cell_perlin_scale
            ) for x, y in zip(xPoints, yPoints)
        ]
        noise_array = np.array(noise_values) * cell_height_scale
        return noise_array
