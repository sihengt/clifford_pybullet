import pybullet as p
import numpy as np
import torch

class Terrain:
    """
    parent class for random terrain generation
    """
    def __init__(self, terrainParams, physicsClientId=0):
        self.physicsClientId = physicsClientId
        self.setParams(terrainParams)

    def createGrid(self, dim, scale):
        grid = np.arange(dim) * scale
        return grid - grid.mean()

    def setParams(self, terrainParams):
        """
        Creates the map grid via mapWidth / Length, scales the coordinates according to scale, and centers coordinates 
        around zero.

        Populates self.gridX, self.gridY, self.mapArea and self.mapBounds.
        """
        self.terrainParams = terrainParams
        # define map grid
        x = self.createGrid(self.terrainParams['mapWidth'], self.terrainParams['mapScale'])
        y = self.createGrid(self.terrainParams['mapLength'], self.terrainParams['mapScale'])
        
        max_x, min_x = x[-1], x[0] 
        max_y, min_y = x[-1], x[0] 

        self.mapArea = (max_y - min_y) * (max_x - min_x)
        self.mapBounds = torch.tensor([
            [min_x, min_y],
            [max_x, min_y],
            [min_x, max_y],
            [max_x, max_y]
        ])

        self.gridX, self.gridY = np.meshgrid(x, y, indexing='xy')

        # TODO: kind of ugly way of getting the seed
        if "seed" in terrainParams:
            self.seed = terrainParams['seed']
        else:
            self.seed = None

    def updateTerrain(self, gridZIn):
        self.gridZ = np.copy(gridZIn)
        shapeArgs = {
            'shapeType'             : p.GEOM_HEIGHTFIELD,
            'meshScale'             : [self.terrainParams['mapScale'], self.terrainParams['mapScale'], 1],
            'heightfieldData'       : self.gridZ.reshape(-1),
            'numHeightfieldRows'    : self.terrainParams['mapWidth'],
            'numHeightfieldColumns' : self.terrainParams['mapLength'],
            'physicsClientId'       : self.physicsClientId
        }

        # Replace current terrain IF it already exists
        if hasattr(self,'terrainShape'):
            shapeArgs['replaceHeightfieldIndex'] = self.terrainShape
        else:
            self.terrainOffset = (np.max(self.gridZ) + np.min(self.gridZ)) / 2.
        self.terrainShape = p.createCollisionShape(**shapeArgs)

        if not hasattr(self,'terrainBody'):
            self.terrainBody  = p.createMultiBody(0, self.terrainShape, physicsClientId=self.physicsClientId)
            p.changeVisualShape(
                self.terrainBody,                       # objectUniqueID
                -1,                                     # linkIndex
                textureUniqueId=-1,                     # textureUniqueId
                rgbaColor=self.terrainParams['color'],  # rgba color
                physicsClientId=self.physicsClientId
            )

        # PyBullet has the origin (0, 0, 0) as ground level, we center the terrain around terrain offset.
        p.resetBasePositionAndOrientation(
            self.terrainBody,
            [0, 0, self.terrainOffset],
            [0, 0, 0, 1],
            physicsClientId=self.physicsClientId
        )
    
    # find maximum terrain height within a circle around a position
    def maxLocalHeight(self, position, radius):
        vecX = self.gridX.reshape(-1) - position[0]
        vecY = self.gridY.reshape(-1) - position[1]
        indices = vecX*vecX+vecY*vecY<radius
        vecZ = self.gridZ.reshape(-1)[indices]
        return np.expand_dims(np.max(vecZ), axis=0)
