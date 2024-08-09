import time
import numpy as np
import pybullet as p
import torch.nn.functional as F
import torch

POSITION = 0
ORIENTATION = 1

ROBOT_OK = 0
ROBOT_STUCK = 1
ROBOT_FLIPPED = 2
ROBOT_OUT_OF_BOUNDS = 3

cam_params = {"pitch": -30, "dist": 0.8, "heading": -90}

class SimController:
    # this class controls the simulation. It controls the terrain and robot, and returns data
    def __init__(self, robot, terrain, simParams, camFollowBot=False, realtime=False, physicsClientId=0, stateProcessor=None):
        self.camFollowBot = camFollowBot
        self.physicsClientId=physicsClientId
        self.terrain = terrain
        self.robot = robot
        self.setParams(simParams)
        self.termTracking = {'stopMoveCount': 0, 'flipped': 0}
        self.realtime = realtime
        self.stateProcessor = stateProcessor
        self.plottedLines = []

    def resetRobot(self, doFall=True, pose=((0,0),(0,0,0,1))):
        """
        Resets robot to pose defined by input.
        If no pose is provided, sets x, y to (0,0) and z to the height + 0.4m
        Also resets parameters used in simulation. 
        """
        realtime = self.realtime
        self.realtime = False
        self.controlLoopStep([0,0])

        # pose[0] \in R^3 = height provided. Use height.
        if len(pose[0])>2:
            safeFallHeight = pose[0][2]
        else:
            safeFallHeight = self.terrain.maxLocalHeight(pose[POSITION], 1) + 0.4
        
        reset_pose = (
            (pose[POSITION][0], pose[POSITION][1], safeFallHeight),
            pose[ORIENTATION]
        )
        self.robot.reset(reset_pose)
        
        # Let the robot fall
        if doFall:
            fallSteps = int(np.ceil(self.simParams['resetFallTime']/self.simParams['simTimeStep']))
            for _ in range(fallSteps):
                self.stepSim()
        
        self.termTracking = {'stopMoveCount': 0, 'flipped': 0}
        self.realtime = realtime

    def controlLoopStep(self, driveCommand, useEulerState=False, useBodyVel=False):
        """
        For each step within the control loop, drives the robot by numStepsPerControl according to commands.
        This function stores prevState BEFORE controls, and nextState AFTER controls, with an optional stateProcessor 
        decorator if needed.

        Returns:
            [prevState, driveCommand, nextState, termCheck]
        """
        throttle = driveCommand[0]
        steering = driveCommand[1:] # generally front_steer / rear_steer
        # getRobotState returns: position_b, orientation_b, velocity_s, jointstate.
        prevState = list(self.getRobotState(useBodyVel))

        # Drive / steer for numStepsPerControl steps within the simulator.
        self.robot.drive(throttle)
        self.robot.steer(steering)
        for _ in range(self.simParams['numStepsPerControl']):
            self.stepSim()

        # Gets robot state after driving for number of steps.
        nextState = list(self.getRobotState(useBodyVel))
        termCheck = self.simTerminateCheck(nextState)
        
        # Converts from quaternion to euler angles if useEulerState
        if useEulerState:
            prevState[1] = p.getEulerFromQuaternion(prevState[1])
            nextState[1] = p.getEulerFromQuaternion(nextState[1])
        
        # Converting every element in list to lists (some are tuples)
        # TODO: why don't we just get them to lists of lists to begin with?
        prevState = [list(item) for item in prevState]
        nextState = [list(item) for item in nextState]
        
        # Flattening list
        prevState = [item for sublist in prevState for item in sublist]
        nextState = [item for sublist in nextState for item in sublist]
        
        if self.stateProcessor is None:
            return prevState, driveCommand, nextState, termCheck
        else:
            return self.stateProcessor(prevState), driveCommand, self.stateProcessor(nextState), termCheck

    def getWeightDistribution(self):
        NORMAL_FORCE = 9
        normalForces = []
        # wheelNames = ['fr_wheel', 'fl_wheel', 'br_wheel', 'bl_wheel']
        for wheel in self.robot.wheelNames:
            contactPoint = p.getContactPoints(
                bodyA=self.robot.robotID,
                bodyB=self.terrain.terrainBody,
                linkIndexA = self.robot.linkNameToID[wheel],
                physicsClientId=self.physicsClientId)
            
            normalForce = 0
            for i_contact in range(len(contactPoint)):
                normalForce += contactPoint[i_contact][NORMAL_FORCE]
            normalForces.append(normalForce)

        print(normalForces)

    def getRobotState(self, useBodyVel=False):
        """
        Returns robot state in following format:
            [world2body_pos, world2body_orien, vel(body / world), jointState(joint positions, joint velocities)]
        """
        world2body_pos, world2body_orien = self.robot.getBasePositionOrientation()
        if useBodyVel:
            vel = self.robot.getBaseVelocity_body()
        else:
            vel = self.robot.getBaseVelocity_world()
        jointState = self.robot.measureJoints()
        return world2body_pos, world2body_orien, vel, jointState
    
    def stepSim(self):
        """
        Takes one step in the simulator.
        Sets camera parameters if camera is following the car.
        
        Updates self.lastTimeStep to prevent simulation from attempting rapid catch-up.
        """
        p.stepSimulation(physicsClientId=self.physicsClientId)
        if self.camFollowBot:
            pos, _, heading, tilt = self.robot.getBasePositionOrientation(calcHeadingTilt=True)
            
            p.resetDebugVisualizerCamera(
                cam_params["dist"],
                heading * 180.0/np.pi + cam_params["heading"],
                cam_params["pitch"],
                pos,
                physicsClientId=self.physicsClientId
            )
        
        currTime = time.time()
        if not hasattr(self,'lastTimeStep'):
            self.lastTimeStep = currTime
        self.lastTimeStep = max(self.lastTimeStep, currTime - self.simParams['simTimeStep'])
        
        if self.realtime:
            time.sleep(max(0, self.lastTimeStep + self.simParams['simTimeStep'] - currTime))
        self.lastTimeStep += self.simParams['simTimeStep']
        if hasattr(self,'screenRecorder'):
            self.screenRecorder.simStep()

    def simTerminateCheck(self, robotState):
        """
            robotState: position_b, orientation_b, velocity, joint state
            Returns:
                ROBOT_OK (0): all good.
                ROBOT_STUCK (1): if robot is stuck
                ROBOT_FLIPPED (2): if robot is flipped
                ROBOT_OOB (3): if robot OOB
        """
        p_sb, R_sb = robotState[0:2]

        # Check if robot has been stuck before OR if checks squared distance > than squared moveThreshold.
        if not 'lastX' in self.termTracking or\
            ((p_sb[0] - self.termTracking['lastX']) ** 2 + (p_sb[1] - self.termTracking['lastY']) ** 2) \
        > self.moveThreshold ** 2:
            self.termTracking['lastX'] = p_sb[0]
            self.termTracking['lastY'] = p_sb[1]
            self.termTracking['stopMoveCount'] = 0
        else:
            self.termTracking['stopMoveCount'] += 1
        if self.simParams['maxStopMoveLength'] > 0 and\
                self.termTracking['stopMoveCount'] >= self.simParams["maxStopMoveLength"]:
            return ROBOT_STUCK

        # Check if robot has flipped        
        upDir = p.multiplyTransforms(p_sb, R_sb, [0,0,1], [0,0,0,1])[POSITION]
        
        # Check if z-value is below 0
        if upDir[2] < 0:
            self.termTracking['flippedCount'] += 1
        else:
            self.termTracking['flippedCount'] = 0
        
        if self.simParams['maxFlippedCount'] > 0 and \
                self.termTracking['flippedCount'] >= self.simParams['maxFlippedCount']:
            return ROBOT_FLIPPED
        
        # Check if robot is OOB
        minZ = np.min(self.terrain.gridZ)
        maxZ = np.max(self.terrain.gridZ) + self.simParams["terminationHeight"]
        minX = np.min(self.terrain.gridX) + 1.
        maxX = np.max(self.terrain.gridX) - 1.
        minY = np.min(self.terrain.gridY) + 1.
        maxY = np.max(self.terrain.gridY) - 1.
        
        if p_sb[0] > maxX or p_sb[0] < minX \
        or p_sb[1] > maxY or p_sb[1] < minY \
        or p_sb[2] > maxZ or p_sb[2] < minZ:
            return ROBOT_OUT_OF_BOUNDS
        
        return ROBOT_OK

    def setParams(self, simParams):
        self.simParams = simParams
        p.setPhysicsEngineParameter(
            fixedTimeStep=self.simParams['simTimeStep'],
            numSubSteps=self.simParams['numSubSteps'],
            erp=self.simParams['erp'],
            numSolverIterations=self.simParams['numSolverIterations'],
            physicsClientId=self.physicsClientId
        )
        p.setGravity(0, 0, self.simParams["gravity"], physicsClientId=self.physicsClientId)
        self.moveThreshold = self.simParams["moveThreshold"]

    def plotTerrainLines(self, lineStart, lineEnd, lineWidth=1, color=(0,0,0)):        
        if type(lineStart) != torch.Tensor:
            lineStart = torch.tensor(lineStart)
        if type(lineEnd) != torch.Tensor:
            lineEnd = torch.tensor(lineEnd)
        
        # Assumes if lineStart's shape is < 2 that it is squeezed.
        if len(lineStart.shape) < 2:
            lineStart = lineStart.unsqueeze(0)
            lineEnd = lineEnd.unsqueeze(0)
        
        # Handle line width and color to match number of lines.
        if not hasattr(lineWidth, '__len__'):
            lineWidth = [lineWidth]*lineStart.shape[0]
        if not hasattr(color[0], '__len__'):
            color = [color]*lineStart.shape[0]

        # Calculate relative XY coordinates
        linesXY = torch.cat((lineStart[:,:2], lineEnd[:,:2]), dim=0)
        mapOrigin = self.terrain.mapBounds[0]
        mapDirs = self.terrain.mapBounds[1:3] - mapOrigin
        scaledMapDirs = mapDirs/torch.sum(mapDirs ** 2, dim=-1)
        relXY = (linesXY - mapOrigin).matmul(scaledMapDirs.transpose(0, 1)) * 2 - 1

        # get ground height using grid sample
        groundHeight = F.grid_sample(
            torch.tensor(self.terrain.gridZ).unsqueeze(0).unsqueeze(0),
            relXY.unsqueeze(0).unsqueeze(0),
            align_corners=True
        ).squeeze()

        # Adjust line start and end heights
        adjustedLineStart = lineStart.clone()
        adjustedLineEnd = lineEnd.clone()
        adjustedLineStart[:,2] += groundHeight[:lineStart.shape[0]]
        adjustedLineEnd[:,2] += groundHeight[lineStart.shape[0]:]

        for i_line in range(lineStart.shape[0]):
            lineParams = {
                'lineFromXYZ': adjustedLineStart[i_line, :],
                'lineToXYZ': adjustedLineEnd[i_line, :],
                'lineWidth': lineWidth[i_line],
                'lineColorRGB': color[i_line],
                'physicsClientId': self.physicsClientId}
            if i_line < len(self.plottedLines):
                lineParams['replaceItemUniqueId'] = self.plottedLines[i]
                p.addUserDebugLine(**lineParams)
            else:
                self.plottedLines.append(p.addUserDebugLine(**lineParams))
        
        # Hide lines used previously but not required now.
        for i in range(lineStart.shape[0],len(self.plottedLines)):
            p.addUserDebugLine([0,0,0], [0,0,0], replaceItemUniqueId=self.plottedLines[i])

    def bufferTerrainLine(self, lineStart=None, lineEnd=None, lineWidth=1, color=(0,0,0), resetBuffer=False, flush=False):
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
                lineWidth = [lineWidth] * lineStart.shape[0]

            if not hasattr(color[0],'__len__'):
                color = [color] * lineStart.shape[0]

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

    def bufferConstHeightLine(self, lineXY, height, alpha=0, lineWidth=1, color=(0,0,0), resetBuffer=False, flush=False):
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
