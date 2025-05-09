import pybullet as p
import numpy as np
import torch

class Terrain:
    """
    Parent class for Terrain Generation
    """
    def __init__(self, terrainParams, physicsClientId=0):
        """
        Initializes PyBullet Terrain. Calls two functions:
            setParams
            updateTerrain
        """
        self.seed = None
        self.physicsClientId = physicsClientId
        self.setParams(terrainParams)
        self.updateTerrain(np.zeros_like(self.gridX))

    def createGrid(self, map_width, map_length, scale):
        """
        Creates a map grid based on the width / length and the scale of the map.
        Populates self.gridX, self.gridY.

        Params:
            map_width: 
            map_length:
            map_scale:
        """
        
        # Creating range of actual map values according to scale, and zero centering it so (0,0) is at the center.
        grid_x = np.arange(map_width) * scale
        grid_x = grid_x - grid_x.mean()

        grid_y = np.arange(map_length) * scale
        grid_x = grid_y - grid_y.mean()

        self.gridX, self.gridY = np.meshgrid(grid_x, grid_y, indexing='xy')

        return self.gridX, self.gridY

    def setParams(self, terrainParams):
        """
        Creates the map grid via mapWidth / Length, scales the coordinates according to scale, and centers coordinates 
        around zero.

        Populates self.gridX, self.gridY, self.mapArea and self.mapBounds.
        """
        self.terrainParams = terrainParams
        
        # Create map grid (function automatically populates self.gridX and self.gridY)
        self.createGrid(self.terrainParams['mapWidth'], self.terrainParams['mapLength'], self.terrainParams['mapScale'])
        
        # Setting map bounds and calculating area of map
        max_x, min_x = np.max(self.gridX), np.min(self.gridX) 
        max_y, min_y = np.max(self.gridY), np.min(self.gridY)

        self.mapArea = (max_y - min_y) * (max_x - min_x)
        self.mapBounds = torch.tensor([
            [min_x, min_y],
            [max_x, min_y],
            [min_x, max_y],
            [max_x, max_y]
        ])

        # If there's a seed option, we include a seed as well. This is for random generation.
        if "seed" in terrainParams:
            self.seed = terrainParams['seed']

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

        # Replace current terrain if it already exists
        if hasattr(self, 'terrainShape'):
            shapeArgs['replaceHeightfieldIndex'] = self.terrainShape
        
        # For centering terrain.
        self.terrainOffset = (np.max(self.gridZ) + np.min(self.gridZ)) / 2.
        
        # Creates shape and updates ID of terrainShape
        self.terrainShape = p.createCollisionShape(**shapeArgs)

        # If terrain hasn't been instantiated, create it:
        if not hasattr(self, 'terrainBody'):
            self.terrainBody = p.createMultiBody(0, self.terrainShape, physicsClientId=self.physicsClientId)
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
