import pybullet as p
import pdb
import numpy as np
import time
import torch.nn.functional as F
import torch

class simController:
    # this class controls the simulation. It controls the terrain and robot, and returns data
    def __init__(self,robot,terrain,simParams,camFollowBot=False,realtime=False,physicsClientId=0,stateProcessor=None):
        self.camFollowBot = camFollowBot
        self.camPitch = -30
        self.camDist = 0.8
        self.camHeading = -90
        self.physicsClientId=physicsClientId
        self.terrain = terrain
        self.robot = robot
        self.setParams(simParams)
        self.termTracking = {}
        self.realtime = realtime
        self.stateProcessor = stateProcessor

    # reset the robot
    def resetRobot(self,doFall=True,pose=((0,0),(0,0,0,1))):
        realtime = self.realtime
        self.realtime = False
        self.controlLoopStep([0,0])
        if len(pose[0])>2:
            safeFallHeight = pose[0][2]
        else:
            safeFallHeight = self.terrain.maxLocalHeight(pose[0],1)+0.4
        self.robot.reset(((pose[0][0],pose[0][1],safeFallHeight),pose[1]))
        if doFall:
            fallSteps = int(np.ceil(self.simParams['resetFallTime']/self.simParams['simTimeStep']))
            for i in range(fallSteps):
                self.stepSim()
        self.termTracking = {}
        self.realtime = realtime

    def controlLoopStep(self,driveCommand,useEulerState=False,useBodyVel=False):
        throttle = driveCommand[0]
        steering = driveCommand[1:]
        #if len(steering) == 1:
        #    steering = steering[0]
        
        # getRobotState returns: position, orientation, velocity, jointstate.
        prevState = list(self.getRobotState(useBodyVel))
        self.robot.drive(throttle)
        self.robot.steer(steering)
        for i in range(self.simParams['numStepsPerControl']):
            self.stepSim()
        nextState = list(self.getRobotState(useBodyVel))
        termCheck = self.simTerminateCheck(nextState)
        if useEulerState:
            prevState[1] = p.getEulerFromQuaternion(prevState[1])
            nextState[1] = p.getEulerFromQuaternion(nextState[1])
        # Converting every element in list to lists (some are tuples)
        prevState = [list(item) for item in prevState]
        nextState = [list(item) for item in nextState]
        # Flattening list
        prevState = [item for sublist in prevState for item in sublist]
        nextState = [item for sublist in nextState for item in sublist]
        if self.stateProcessor is None:
            return prevState,driveCommand,nextState,termCheck
        else:
            return self.stateProcessor(prevState),driveCommand,self.stateProcessor(nextState),termCheck

    def getWeightDistribution(self):
        normalForces = []
        for wheel in ['fr_wheel','fl_wheel','br_wheel','bl_wheel']:
            contactPoint = p.getContactPoints(bodyA=self.robot.robotID,bodyB=self.terrain.terrainBody,
                                linkIndexA = self.robot.linkNameToID[wheel],
                                physicsClientId=self.physicsClientId)
            normalForce = 0
            for i in range(len(contactPoint)):
                normalForce+=contactPoint[i][9]
            normalForces.append(normalForce)
        print(normalForces)


    def getRobotState(self,useBodyVel=False):
        pos,orien = self.robot.getBasePositionOrientation()
        if useBodyVel:
            vel = self.robot.getBaseVelocity_body()
        else:
            vel = self.robot.getBaseVelocity_world()
        jointState = self.robot.measureJoints()
        return pos,orien,vel,jointState
    
    def stepSim(self):
        p.stepSimulation(physicsClientId=self.physicsClientId)
        if self.camFollowBot:
            pos,_,heading,tilt = self.robot.getBasePositionOrientation(calcHeadingTilt=True)
            p.resetDebugVisualizerCamera(self.camDist,heading*180.0/np.pi+self.camHeading,self.camPitch,pos,physicsClientId=self.physicsClientId)
        currTime = time.time()
        if not hasattr(self,'lastTimeStep'):
            self.lastTimeStep = currTime
        self.lastTimeStep = max(self.lastTimeStep,currTime-self.simParams['simTimeStep'])
        if self.realtime:
            time.sleep(max(0,self.lastTimeStep+self.simParams['simTimeStep']-currTime))
        self.lastTimeStep += self.simParams['simTimeStep']
        if hasattr(self,'screenRecorder'):
            self.screenRecorder.simStep()

    # check if simulation should be terminated
    def simTerminateCheck(self,robotState):
        pos,orien = robotState[0:2]
        # check how long robot has been stuck
        if not 'stopMoveCount' in self.termTracking or\
                (pos[0]-self.termTracking['lastX'])*(pos[0]-self.termTracking['lastX']) +\
                (pos[1]-self.termTracking['lastY'])*(pos[1]-self.termTracking['lastY'])\
                > self.moveThreshold:
            self.termTracking['lastX'] = pos[0]
            self.termTracking['lastY'] = pos[1]
            self.termTracking['stopMoveCount'] = 0
        else:
            self.termTracking['stopMoveCount'] += 1
        if self.simParams['maxStopMoveLength'] > 0 and\
                self.termTracking['stopMoveCount'] >= self.simParams["maxStopMoveLength"]:
            return 1

        # flipped robot termination criteria
        if not 'flippedCount' in self.termTracking:
            self.termTracking['flippedCount'] = 0
        upDir = p.multiplyTransforms(pos,orien,[0,0,1],[0,0,0,1])[0]
        if upDir[2] < 0:
            self.termTracking['flippedCount'] += 1
        else:
            self.termTracking['flippedCount'] = 0
        if self.simParams['maxFlippedCount'] > 0 and\
                self.termTracking['flippedCount'] >= self.simParams['maxFlippedCount']:
            return 2
        # boundary criteria
        minZ = np.min(self.terrain.gridZ)
        maxZ = np.max(self.terrain.gridZ) + self.simParams["terminationHeight"]
        minX = np.min(self.terrain.gridX) + 1.
        maxX = np.max(self.terrain.gridX) - 1.
        minY = np.min(self.terrain.gridY) + 1.
        maxY = np.max(self.terrain.gridY) - 1.
        if pos[0] > maxX or pos[0] < minX or \
        pos[1] > maxY or pos[1] < minY or \
        pos[2] > maxZ or pos[2] < minZ:
             return 3
        return False

    def setParams(self,simParams):
        self.simParams = simParams
        p.setPhysicsEngineParameter(fixedTimeStep=self.simParams['simTimeStep'],
                    numSubSteps=self.simParams['numSubSteps'],
                    erp=self.simParams['erp'],
                    numSolverIterations=self.simParams['numSolverIterations'],
                    physicsClientId=self.physicsClientId)
        p.setGravity(0,0,self.simParams["gravity"],physicsClientId=self.physicsClientId)
        self.moveThreshold = self.simParams["moveThreshold"]**2 # store square distance for easier computation later

    def plotTerrainLines(self,lineStart,lineEnd,lineWidth = 1,color = (0,0,0), replaceLines = True):
        if not hasattr(self,'plottedLines'):
            self.plottedLines = []
        
        if type(lineStart) != torch.Tensor:
            lineStart = torch.tensor(lineStart)

        if type(lineEnd) != torch.Tensor:
            lineEnd = torch.tensor(lineEnd)

        if len(lineStart.shape) < 2:
            lineStart = lineStart.unsqueeze(0)
            lineEnd = lineEnd.unsqueeze(0)

        if not hasattr(lineWidth,'__len__'):
            lineWidth = [lineWidth]*lineStart.shape[0]

        if not hasattr(color[0],'__len__'):
            color = [color]*lineStart.shape[0]

        linesXY = torch.cat((lineStart[:,:2],lineEnd[:,:2]),dim=0)

        # make xy relative to terrain map
        mapOrigin = self.terrain.mapBounds[0]
        mapDirs = self.terrain.mapBounds[1:3] - mapOrigin
        scaledMapDirs = mapDirs/torch.sum(mapDirs**2,dim=-1)
        relXY = (linesXY - mapOrigin).matmul(scaledMapDirs.transpose(0,1))*2-1

        # get ground height using grid sample
        groundHeight = F.grid_sample(torch.tensor(self.terrain.gridZ).unsqueeze(0).unsqueeze(0),
                                    relXY.unsqueeze(0).unsqueeze(0),
                                    align_corners=True).squeeze()

        adjustedLineStart = lineStart.clone()
        adjustedLineEnd = lineEnd.clone()
        adjustedLineStart[:,2] = adjustedLineStart[:,2] + groundHeight[:lineStart.shape[0]]
        adjustedLineEnd[:,2] = adjustedLineEnd[:,2] + groundHeight[lineStart.shape[0]:]

        for i in range(lineStart.shape[0]):
            try:
                lineParams = {'lineFromXYZ': adjustedLineStart[i,:],
                            'lineToXYZ': adjustedLineEnd[i,:],
                            'lineWidth': lineWidth[i],
                            'lineColorRGB': color[i],
                            'physicsClientId': self.physicsClientId}
            except:
                pdb.set_trace()
            if i < len(self.plottedLines):
                lineParams['replaceItemUniqueId'] = self.plottedLines[i]
                p.addUserDebugLine(**lineParams)
            else:
                self.plottedLines.append(p.addUserDebugLine(**lineParams))
        for i in range(lineStart.shape[0],len(self.plottedLines)):
            p.addUserDebugLine([0,0,0],[0,0,0],replaceItemUniqueId=self.plottedLines[i])

    def bufferTerrainLine(self,lineStart=None,lineEnd=None,lineWidth=1,color=(0,0,0),resetBuffer=False,flush=False):
        if resetBuffer or not hasattr(self,'lineStartBuffer'):
            self.lineStartBuffer = torch.tensor([])
            self.lineEndBuffer = torch.tensor([])
            self.lineWidthBuffer = []
            self.lineColorBuffer = []

        if not lineStart is None:
            if type(lineStart) != torch.Tensor:
                lineStart = torch.tensor(lineStart)

            if type(lineEnd) != torch.Tensor:
                lineEnd = torch.tensor(lineEnd)

            if len(lineStart.shape) < 2:
                lineStart = lineStart.unsqueeze(0)
                lineEnd = lineEnd.unsqueeze(0)

            if not hasattr(lineWidth,'__len__'):
                lineWidth = [lineWidth]*lineStart.shape[0]

            if not hasattr(color[0],'__len__'):
                color = [color]*lineStart.shape[0]

            self.lineStartBuffer = torch.cat((self.lineStartBuffer,lineStart),dim=0)
            self.lineEndBuffer = torch.cat((self.lineEndBuffer,lineEnd),dim=0)
            self.lineWidthBuffer += lineWidth
            self.lineColorBuffer += color

        if flush:
            self.plotTerrainLines(self.lineStartBuffer,self.lineEndBuffer,self.lineWidthBuffer,self.lineColorBuffer,replaceLines = True)
            self.lineStartBuffer = torch.tensor([])
            self.lineEndBuffer = torch.tensor([])
            self.lineWidthBuffer.clear()
            self.lineColorBuffer.clear()

    def bufferConstHeightLine(self,lineXY,height,alpha=0,lineWidth=1,color=(0,0,0),resetBuffer=False,flush=False):
        # make xy relative to terrain size
        mapOrigin = self.terrain.mapBounds[0]
        mapDirs = self.terrain.mapBounds[1:3] - mapOrigin
        scaledMapDirs = mapDirs/torch.sum(mapDirs**2,dim=-1)
        relXY = (lineXY - mapOrigin).matmul(scaledMapDirs.transpose(0,1))*2-1
        

        # get ground height using grid sample
        groundHeight = F.grid_sample(torch.tensor(self.terrain.gridZ).unsqueeze(0).unsqueeze(0),
                                    relXY.unsqueeze(0).unsqueeze(0),
                                    align_corners=True).squeeze()

        # calculate absolute height of points
        absHeight = groundHeight + height
        if alpha == 0:
            absHeight[:] = absHeight[0]
        elif alpha < 1:
            for i in range(1,absHeight.shape[0]):
                absHeight[i] = absHeight[i]*alpha + (1-alpha)*absHeight[i-1]

        relHeight = absHeight - groundHeight

        # plot lines
        xyz = torch.cat((lineXY,relHeight.unsqueeze(-1)),dim=-1)
        self.bufferTerrainLine(xyz[:-1],xyz[1:],lineWidth,color,resetBuffer,flush)
