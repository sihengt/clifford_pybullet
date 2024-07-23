import pybullet as p
import numpy as np
from scipy.ndimage import gaussian_filter
from noise import pnoise1,pnoise2
from scipy.interpolate import interp2d
from scipy.interpolate import griddata
import torch

class terrain(object):
    """
    parent class for random terrain generation
    """
    def __init__(self,terrainParams,physicsClientId=0):
        self.physicsClientId = physicsClientId
        self.setParams(terrainParams)

    def setParams(self,terrainParams):
        self.terrainParams = terrainParams
        # define map grid
        self.gridX = np.arange(self.terrainParams['mapWidth'])*self.terrainParams['mapScale']
        self.gridX = self.gridX-self.gridX.mean()
        self.gridY = np.arange(self.terrainParams['mapLength'])*self.terrainParams['mapScale']
        self.gridY = self.gridY-self.gridY.mean()
        self.mapArea = (self.gridY[-1]-self.gridY[0])*(self.gridX[-1]-self.gridX[0])
        self.gridX,self.gridY = np.meshgrid(self.gridX,self.gridY,indexing='xy')
        self.mapBounds = torch.tensor([[self.gridX[0,0],self.gridY[0,0]],
                                    [self.gridX[0,-1],self.gridY[0,-1]],
                                    [self.gridX[-1,0],self.gridY[-1,0]],
                                    [self.gridX[-1,-1],self.gridY[-1,-1]]])

    def updateTerrain(self,gridZIn):
        self.gridZ = np.copy(gridZIn)
        shapeArgs = {'shapeType':p.GEOM_HEIGHTFIELD,
                    'meshScale':[self.terrainParams['mapScale'],self.terrainParams['mapScale'],1],
                    'heightfieldData':self.gridZ.reshape(-1),
                    'numHeightfieldRows':self.terrainParams['mapWidth'],
                    'numHeightfieldColumns':self.terrainParams['mapLength'],
                    'physicsClientId':self.physicsClientId}

        if hasattr(self,'terrainShape'):
            shapeArgs['replaceHeightfieldIndex']=self.terrainShape
        else:
            self.terrainOffset = (np.max(self.gridZ)+np.min(self.gridZ))/2.
        self.terrainShape = p.createCollisionShape(**shapeArgs)

        if not hasattr(self,'terrainBody'):
            self.terrainBody  = p.createMultiBody(0, self.terrainShape,physicsClientId=self.physicsClientId)
            p.changeVisualShape(self.terrainBody, -1, textureUniqueId=-1,rgbaColor=self.terrainParams['color'],
                                physicsClientId=self.physicsClientId)

        # position terrain correctly
        p.resetBasePositionAndOrientation(self.terrainBody,[0,0,self.terrainOffset],[0,0,0,1],physicsClientId=self.physicsClientId)
    
    # find maximum terrain height within a circle around a position
    def maxLocalHeight(self,position,radius):
        vecX = self.gridX.reshape(-1)-position[0]
        vecY = self.gridY.reshape(-1)-position[1]
        indices = vecX*vecX+vecY*vecY<radius
        vecZ = self.gridZ.reshape(-1)[indices]
        return np.expand_dims(np.max(vecZ), axis=0)

class randomRockyTerrain(terrain):
    """
    This class handles the generation of random rocky terrain
    """
    # initialize terrain object
    def __init__(self,terrainParams,physicsClientId=0):
        super().__init__(terrainParams,physicsClientId)
        self.generate()

    # generate new terrain.
    def generate(self):
        gridZ = np.zeros_like(self.gridX)

        # generate random blocks
        numCells = int(self.mapArea/self.terrainParams['averageAreaPerCell'])
        blockHeights = self.randomSteps(self.gridX,self.gridY,numCells,self.terrainParams["cellPerlinScale"],self.terrainParams["cellHeightScale"])
        blockHeights = gaussian_filter(blockHeights, sigma=self.terrainParams["smoothing"])
        gridZ += blockHeights

        # add more small noise
        for i in range(len(self.terrainParams['perlinScale'])):
            smallNoise = self.perlinNoise(self.gridX.reshape(-1),self.gridY.reshape(-1),
                                        self.terrainParams["perlinScale"][i],
                                        self.terrainParams["perlinHeightScale"][i])
            gridZ += smallNoise.reshape(gridZ.shape)

        # add sharp point noise
        numSharpPoints = int(self.mapArea*self.terrainParams['sharpDensity'])
        if numSharpPoints > 0:
            gridZ += self.sharpNoise(numSharpPoints,self.terrainParams['maxSharpHeight'],self.gridX,self.gridY)

        # make center flat for initial robot start position
        gridZ = self.flatten(gridZ,[0,0])
        #originHeight = interp2d(self.gridX[0,:],self.gridY[:,0],gridZ, kind='linear')(0,0)
        self.updateTerrain(gridZ-(np.max(gridZ)+np.min(gridZ))/2.0)

    def sharpNoise(self,N,maxHeight,gridX,gridY):
        groundPointsX = np.random.rand(N)*(gridX.max()-gridX.min())+gridX.min()
        groundPointsY = np.random.rand(N)*(gridY.max()-gridY.min())+gridY.min()
        groundPointsZ = np.random.rand(N)*maxHeight
        z = griddata((groundPointsX,groundPointsY), groundPointsZ, (gridX, gridY), method='linear', fill_value=0)
        return z

    def flatten(self,gridZ,loc):
            distFromLoc = np.sqrt((self.gridX-loc[0])**2+(self.gridY-loc[1])**2)
            flatIndices = distFromLoc<self.terrainParams['flatRadius']
            if flatIndices.any():
                flatHeight = np.mean(gridZ[flatIndices])
                gridZ[flatIndices] = flatHeight
                distFromFlat = distFromLoc - self.terrainParams['flatRadius']
                blendIndices = distFromFlat < self.terrainParams['blendRadius']
                gridZ[blendIndices] = flatHeight + (gridZ[blendIndices]-flatHeight)*distFromFlat[blendIndices]/self.terrainParams['blendRadius']
            return gridZ

    def randomSteps(self,gridX,gridY,numCells,cellPerlinScale,cellHeightScale):
        centersX = np.random.uniform(size=numCells,low=gridX.min(),high=gridX.max())
        centersY = np.random.uniform(size=numCells,low=gridY.min(),high=gridY.max())
        centersZ = self.perlinNoise(centersX,centersY,cellPerlinScale,cellHeightScale)
        return griddata((centersX,centersY),centersZ,(gridX,gridY),method='nearest')

    def perlinNoise(self,xPoints,yPoints,perlinScale,heightScale):
        randomSeed = np.random.rand(2)*1000
        return np.array([pnoise2(randomSeed[0]+xPoints[i]*perlinScale,randomSeed[1]+yPoints[i]*perlinScale) for i in range(len(xPoints))])*heightScale
