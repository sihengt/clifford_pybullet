import pybullet as p
import os
from SimRobot import SimRobot
from utils.checkRobotExists import checkRobotExists

cliford_dir = os.path.join(os.path.dirname(__file__), "clifford")

class CliffordRobot(SimRobot):
    def __init__(self, physicsClientId=0):
        super().__init__(physicsClientId)
        self.sdfPath = os.path.join(cliford_dir,'clifford.sdf')
        self.tireNames = ['frtire','fltire','brtire','bltire']
        self.wheelNames = ['fr_wheel', 'fl_wheel', 'br_wheel', 'bl_wheel']
        self.wheel2TireJoints = ['frwheel2tire','flwheel2tire','brwheel2tire','blwheel2tire']
        self.axle2WheelJoints = ['axle2frwheel','axle2brwheel','axle2flwheel','axle2blwheel']
    @checkRobotExists
    def reset(self, pose=((0,0,0.25),(0,0,0,1))):
        p.resetBasePositionAndOrientation(
            self.robotID,
            pose[0],
            pose[1],
            physicsClientId=self.physicsClientId)
        p.resetJointStatesMultiDof(
            self.robotID,
            range(self.nJoints),
            self.initialJointPositions,
            self.initialJointVelocities,
            physicsClientId=self.physicsClientId)

    # Required after initialization to set parameters used by other functions
    def setParams(self, params, importHeight=10, **kwargs):
        self.driveParams = params['drive']
        self.steerParams = params['steer']

        # self.robotID is set when sdf file is imported
        if self.robotID is None:
            self.importClifford(importHeight)
            self.changeColor()
            self.loosenModel()

        self.setSuspensionParam(params['suspension'])
        self.setInertialParam(params['inertia'])
        self.setTireContactParam(params['tireContact'])
    
    def importClifford(self, importHeight=10):
        # SDF file = open chain clifford. We'll add constraints to make closed chain.
        self.robotID = p.loadSDF(self.sdfPath, physicsClientId=self.physicsClientId)[0]

        BASE_POSITION = (0, 0, importHeight) # (x, y, z)
        BASE_ORIENTATION = (0, 0, 0, 1) # (quarternions)
        p.resetBasePositionAndOrientation(
            self.robotID,
            BASE_POSITION,
            BASE_ORIENTATION,
            physicsClientId=self.physicsClientId)

        # define number of joints of clifford robot
        self.nJoints = p.getNumJoints(self.robotID, physicsClientId=self.physicsClientId)
        initialJointStates = p.getJointStatesMultiDof(self.robotID,range(self.nJoints), physicsClientId=self.physicsClientId)
        
        # 0th index = joint position (quaternion); 1st index = joint velocity (cartesian)
        self.initialJointPositions = [initialJointStates[i][0] for i in range(self.nJoints)]
        self.initialJointVelocities = [initialJointStates[i][1] for i in range(self.nJoints)]
        
        self.buildModelDict()
        self.linkNameToID['base'] = -1
        self.recordInitialInertia()

        # make closed chain
        linkFrame2Joint ={}
        linkFrame2Joint['upperSpring']  = [0, 0, 0]
        linkFrame2Joint['outer']        = [0.23, 0, 0]
        linkFrame2Joint['inner']        = [0.195, 0, 0]

        self.addClosedChainConstraint('brsupper',linkFrame2Joint['upperSpring'])
        self.addClosedChainConstraint('blsupper',linkFrame2Joint['upperSpring'])
        self.addClosedChainConstraint('frsupper',linkFrame2Joint['upperSpring'])
        self.addClosedChainConstraint('flsupper',linkFrame2Joint['upperSpring'])
        self.addClosedChainConstraint('bri',linkFrame2Joint['inner'])
        self.addClosedChainConstraint('bli',linkFrame2Joint['inner'])
        self.addClosedChainConstraint('fri',linkFrame2Joint['inner'])
        self.addClosedChainConstraint('fli',linkFrame2Joint['inner'])
        self.addClosedChainConstraint('blo',linkFrame2Joint['outer'])
        self.addClosedChainConstraint('flo',linkFrame2Joint['outer'])

    def isSingleDOFJoint(self, jointIndex):
        JOINT_POSITION = 0
        joint_state = p.getJointStateMultiDof(
            bodyUniqueId=self.robotID,
            jointIndex=jointIndex,
            physicsClientId=self.physicsClientId)
        if len(joint_state[JOINT_POSITION]) == 4:
            return False
        else:
            return True

    @checkRobotExists
    def loosenModel(self):
        for i_joint in range(self.nJoints):
                # Disable control of all joints that have more than 1 DOF
                if not self.isSingleDOFJoint(i_joint):
                    p.setJointMotorControlMultiDof(
                        bodyUniqueId=self.robotID,
                        jointIndex=i_joint,
                        controlMode=p.POSITION_CONTROL,
                        targetPosition=[0,0,0,1],
                        positionGain=0,
                        velocityGain=0,
                        force=[0,0,0],
                        physicsClientId=self.physicsClientId)
        
        # Constraining back wheels
        c = p.createConstraint(self.robotID,
                               self.linkNameToID['blwheel'],
                               self.robotID,
                               self.linkNameToID['brwheel'],
                               jointType=p.JOINT_GEAR,
                               jointAxis=[0, 1, 0],
                               parentFramePosition=[0, 0, 0],
                               childFramePosition=[0, 0, 0],
                               physicsClientId=self.physicsClientId)
        p.changeConstraint(c,
                           gearRatio=-1,
                           maxForce=10000,
                           erp=0.2,
                           physicsClientId=self.physicsClientId)
        
        # Constraining front wheels
        c = p.createConstraint(self.robotID,
                               self.linkNameToID['flwheel'],
                               self.robotID,
                               self.linkNameToID['frwheel'],
                               jointType=p.JOINT_GEAR,
                               jointAxis=[0, 1, 0],
                               parentFramePosition=[0, 0, 0],
                               childFramePosition=[0, 0, 0],
                               physicsClientId=self.physicsClientId)
        p.changeConstraint(c,
                           gearRatio=-1,
                           maxForce=10000,
                           erp=0.2,
                           physicsClientId=self.physicsClientId)

    @checkRobotExists
    def recordInitialInertia(self):
        self.linkInertias = {}
        self.tireMass = 0
        self.baseMass = 0
        self.otherMass = 0
        
        MASS = 0
        LOCAL_INERTIA_DIAGONAL = 2 # vec3

        for linkName, linkID in self.linkNameToID.items():
            dynamicsInfo = p.getDynamicsInfo(
                self.robotID,
                linkID,
                physicsClientId=self.physicsClientId)
            
            self.linkInertias[linkName] = (
                dynamicsInfo[MASS],
                dynamicsInfo[LOCAL_INERTIA_DIAGONAL][0],
                dynamicsInfo[LOCAL_INERTIA_DIAGONAL][1],
                dynamicsInfo[LOCAL_INERTIA_DIAGONAL][2])
            
            if 'tire' in linkName:
                self.tireMass += dynamicsInfo[MASS]
            elif linkName == 'base':
                self.baseMass = dynamicsInfo[MASS]
            else:
                self.otherMass += dynamicsInfo[MASS]

    @checkRobotExists
    def changeColor(self, color=None):
        if color is None:
            color = [0.6,0.1,0.1,1]
        for i in range(-1, self.nJoints):
            p.changeVisualShape(self.robotID, i, rgbaColor=color, specularColor=color, physicsClientId=self.physicsClientId)
        for tire in self.tireNames:
            p.changeVisualShape(self.robotID,self.linkNameToID[tire],rgbaColor=[0.15,0.15,0.15,1],specularColor=[0.15,0.15,0.15,1],physicsClientId=self.physicsClientId)
    
    @checkRobotExists
    def addClosedChainConstraint(self, linkName, linkFrame2Joint):
        # Indices for getLinkState
        LINK_WORLD_POSITION = 0
        LINK_WORLD_ORIENTATION = 1
        WORLD_LINK_FRAME_POSITION = 4       # vec3
        WORLD_LINK_FRAME_ORIENTATION = 5    # vec4
        
        # Indices for transformations
        POSITION = 0
        ORIENTATION = 1

        linkState = p.getLinkState(
            self.robotID,
            self.linkNameToID[linkName],
            physicsClientId=self.physicsClientId)
        
        bodyState = p.getBasePositionAndOrientation(
            self.robotID,
            physicsClientId=self.physicsClientId)
        
        world2joint = p.multiplyTransforms(
            linkState[WORLD_LINK_FRAME_POSITION], linkState[WORLD_LINK_FRAME_ORIENTATION],
            linkFrame2Joint, [0,0,0,1])
        
        body2world = p.invertTransform(bodyState[POSITION], bodyState[ORIENTATION])
        body2joint = p.multiplyTransforms(body2world[POSITION], body2world[ORIENTATION], 
                                          world2joint[POSITION], world2joint[ORIENTATION])
        
        linkcm2world = p.invertTransform(linkState[LINK_WORLD_POSITION], 
                                         linkState[LINK_WORLD_ORIENTATION])
        linkcm2joint = p.multiplyTransforms(linkcm2world[POSITION], linkcm2world[ORIENTATION],
                                            world2joint[POSITION], world2joint[ORIENTATION])

        p.createConstraint(
            self.robotID,                   # parent body unique id
            -1,                             # constraint to base
            self.robotID,                   # child body unique id
            self.linkNameToID[linkName],    # child link index
            p.JOINT_POINT2POINT,            # joint type
            [0,0,0],                        # joint axis
            body2joint[POSITION],           # parent frame position (relative to parent CoM)
            linkcm2joint[POSITION],         # child frame position (relative to child CoM)
            physicsClientId=self.physicsClientId)

    @checkRobotExists
    def setInertialParam(self,inertialParam):
        totalWeightedMass = self.otherMass \
            + self.tireMass * inertialParam['tireRelDensity'] \
            + self.baseMass * inertialParam['baseRelDensity']
        
        for linkName in self.linkNameToID:
            linkMassScale = 1
            if 'tire' in linkName:
                linkMassScale = inertialParam['tireRelDensity']
            elif linkName == 'base':
                linkMassScale = inertialParam['baseRelDensity']
            
            # Distribute totalMass among links according to their weighted contributions.
            massScale = linkMassScale * inertialParam['totalMass'] / totalWeightedMass

            linkMass = self.linkInertias[linkName][0]*massScale
            linkInertia = [self.linkInertias[linkName][1]*massScale,
                           self.linkInertias[linkName][2]*massScale,
                           self.linkInertias[linkName][3]*massScale]
            p.changeDynamics(self.robotID,self.linkNameToID[linkName],physicsClientId=self.physicsClientId,
                            mass=linkMass,localInertiaDiagonal=linkInertia)

        totalMass = 0
        MASS = 0
        for linkName in self.linkNameToID:
            linkMass = p.getDynamicsInfo(
                self.robotID,
                self.linkNameToID[linkName],
                physicsClientId=self.physicsClientId)[MASS]
            totalMass += linkMass

    @checkRobotExists
    def setSuspensionParam(self,suspensionParams):
        springJointNames = ['brslower2upper','blslower2upper','frslower2upper','flslower2upper']
        
        # If only one value provided, assume same values for all suspension.
        if not 'members' in suspensionParams or len(suspensionParams['members']) == 1:
            for key in suspensionParams:
                suspensionParams[key] = [suspensionParams[key]] * len(springJointNames)
            suspensionParams['members'] = springJointNames
        
        for member, preload, springConstant, dampingconstant, lowerLimit, upperLimit in zip(
            suspensionParams['members'],
            suspensionParams['preload'],
            suspensionParams['springConstant'],
            suspensionParams['dampingconstant'],
            suspensionParams['lowerLimit'],
            suspensionParams['upperLimit']):
            p.setJointMotorControl2(bodyUniqueId=self.robotID,
                                    jointIndex=self.jointNameToID[member],
                                    controlMode=p.POSITION_CONTROL,
                                    targetPosition=lowerLimit - preload,
                                    positionGain=springConstant,
                                    velocityGain=dampingconstant,
                                    physicsClientId=self.physicsClientId)
            p.changeDynamics(self.robotID,
                             self.jointNameToID[member],
                             jointLowerLimit=lowerLimit,
                             jointUpperLimit=upperLimit,
                             physicsClientId=self.physicsClientId)

    @checkRobotExists
    def setTireContactParam(self,tireContactParam):
        tireNames = ['frtire','fltire','brtire','bltire']
        
        # If only one value provided, assume same values for all tires.
        if len(tireContactParam['members']) == 1:
            for key in tireContactParam:
                tireContactParam[key] = [tireContactParam[key]] * len(tireNames)
            tireContactParam['members'] = tireNames
        
        for i_tire in range(len(tireContactParam['members'])):
            p.changeDynamics(
                self.robotID,             
                self.linkNameToID[tireContactParam['members'][i_tire]],             
                lateralFriction=tireContactParam['lateralFriction'][i_tire],             
                restitution=tireContactParam['restitution'][i_tire],             
                physicsClientId=self.physicsClientId)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
    
    @checkRobotExists
    def drive(self,driveSpeed):
        driveSpeed = max(min(driveSpeed,1),-1)
        driveJoints = [self.jointNameToID[name] for name in self.wheel2TireJoints]
        
        scale = self.driveParams['scale']
        velocityGain = self.driveParams['velocityGain']
        maxForce = self.driveParams['maxForce']

        targetVelocities = [driveSpeed * scale] * len(driveJoints)
        velocityGains = [velocityGain] * len(driveJoints)
        forces = [maxForce]*len(driveJoints),
        
        p.setJointMotorControlArray(
            self.robotID,
            driveJoints,
            p.VELOCITY_CONTROL,
            targetVelocities=targetVelocities,
            velocityGains=velocityGains,
            forces=forces,
            physicsClientId=self.physicsClientId)

    @checkRobotExists
    def steer(self,angle):
        # Convert angle into a list
        if not isinstance(angle, list):
            angle = [angle]

        # Getting front / rear angles, and clamping
        frontAngle = angle[0]
        rearAngle = angle[1] if len(angle) > 1 else 0
        frontAngle = max(min(frontAngle,1),-1)
        rearAngle = max(min(rearAngle,1),-1)

        steerJoints = [self.jointNameToID[name] for name in self.axle2WheelJoints]
        steerAngles = [-frontAngle * self.steerParams['scale'] + self.steerParams['frontTrim'],
                       rearAngle * self.steerParams['scale'] + self.steerParams['backTrim']
                        ] * 2

        n_steerAngles = len(steerAngles)
        positionGains = self.steerParams['positionGain'] * n_steerAngles
        velocityGains = self.steerParams['velocityGain'] * n_steerAngles
        maxForces = self.steerParams['maxForce'] * n_steerAngles

        p.setJointMotorControlArray(self.robotID,
                                    steerJoints,
                                    p.POSITION_CONTROL,
                                    targetPositions=steerAngles,
                                    positionGains=positionGains,
                                    velocityGains=velocityGains,
                                    forces=maxForces,
                                    physicsClientId=self.physicsClientId)
