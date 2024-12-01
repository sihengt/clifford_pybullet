import pybullet as p
import os

from .SimRobot import SimRobot
from .utils.checkRobotExists import checkRobotExists
from .utils.constants.pybulletIndices import LinkStateIndex, TransformsIndex
import numpy as np
import torch

cliford_dir = os.path.join(os.path.dirname(__file__), "clifford")

class CliffordRobot(SimRobot):
    TIRE_NAMES = ['frtire','fltire','brtire','bltire']
    WHEEL_NAMES = ['fr_wheel', 'fl_wheel', 'br_wheel', 'bl_wheel']
    WHEEL_TO_TIRE_JOINTS = ['frwheel2tire','flwheel2tire','brwheel2tire','blwheel2tire']
    AXLE_TO_WHEEL_JOINTS = ['axle2frwheel','axle2brwheel','axle2flwheel','axle2blwheel']
    SPRING_JOINTS = ['brslower2upper','blslower2upper','frslower2upper','flslower2upper']

    def __init__(self, physicsClientId=0):
        super().__init__(physicsClientId)
        self.sdfPath = os.path.join(cliford_dir,'clifford.sdf')

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
        """
        Takes params and sets sim robot with it. 

        Args:
            params(dict): dictionary of dictionary after YAML is parsed by getParams
        """
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

        # Adding constraints to make the robot closed chain.
        linkFrame2Joint = {
            'upperSpring': [0, 0, 0],
            'outer': [0.23, 0, 0],
            'inner': [0.195, 0, 0]
        }
        self.addClosedChainConstraint('brsupper', linkFrame2Joint['upperSpring'])
        self.addClosedChainConstraint('blsupper', linkFrame2Joint['upperSpring'])
        self.addClosedChainConstraint('frsupper', linkFrame2Joint['upperSpring'])
        self.addClosedChainConstraint('flsupper', linkFrame2Joint['upperSpring'])
        self.addClosedChainConstraint('bri', linkFrame2Joint['inner'])
        self.addClosedChainConstraint('bli', linkFrame2Joint['inner'])
        self.addClosedChainConstraint('fri', linkFrame2Joint['inner'])
        self.addClosedChainConstraint('fli', linkFrame2Joint['inner'])
        self.addClosedChainConstraint('blo', linkFrame2Joint['outer'])
        self.addClosedChainConstraint('flo', linkFrame2Joint['outer'])

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
                        bodyUniqueId    =   self.robotID,
                        jointIndex      =   i_joint,
                        controlMode     =   p.POSITION_CONTROL,
                        targetPosition  =   [0, 0, 0, 1],
                        positionGain    =   0,
                        velocityGain    =   0,
                        force           =   [0, 0, 0],
                        physicsClientId =   self.physicsClientId)
        
        # Constraining back wheels
        c = p.createConstraint(self.robotID,
                               self.linkNameToID['blwheel'],
                               self.robotID,
                               self.linkNameToID['brwheel'],
                               jointType            =   p.JOINT_GEAR,
                               jointAxis            =   [0, 1, 0],
                               parentFramePosition  =   [0, 0, 0],
                               childFramePosition   =   [0, 0, 0],
                               physicsClientId      =   self.physicsClientId)
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
                               jointType            =   p.JOINT_GEAR,
                               jointAxis            =   [0, 1, 0],
                               parentFramePosition  =   [0, 0, 0],
                               childFramePosition   =   [0, 0, 0],
                               physicsClientId      =   self.physicsClientId)
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
        for tire in self.TIRE_NAMES:
            p.changeVisualShape(self.robotID,self.linkNameToID[tire],rgbaColor=[0.15,0.15,0.15,1],specularColor=[0.15,0.15,0.15,1],physicsClientId=self.physicsClientId)
    
    @checkRobotExists
    def addClosedChainConstraint(self, linkName, linkFrame2Joint):
        """
        Adds constraint for link defined in linkName to its COM position, a cartesian offset 
        defined in linkFrame2Joint.

        Args:
            linkName (string): name of link as defined in SDF.
            linkFrame2Joint (list): R^3 cartesian offset
        """
        # Getting link state
        linkState = p.getLinkState(
            self.robotID,
            self.linkNameToID[linkName],
            physicsClientId=self.physicsClientId)
        
        # Getting transformation from world frame to base link
        world2body = self.getBasePositionOrientation()
        
        # Gets the transformation from origin {W} to the link's joint.
        world2joint = p.multiplyTransforms(
            linkState[LinkStateIndex.WORLD_LINK_FRAME_POSITION],
            linkState[LinkStateIndex.WORLD_LINK_FRAME_ORIENTATION],
            linkFrame2Joint, [0,0,0,1])
        
        # Transformation from base link to world
        body2world = p.invertTransform(
            world2body[TransformsIndex.POSITION],
            world2body[TransformsIndex.ORIENTATION]
        )
        
        # base link -> world -> joint
        body2joint = p.multiplyTransforms(
            body2world[TransformsIndex.POSITION],
            body2world[TransformsIndex.ORIENTATION], 
            world2joint[TransformsIndex.POSITION],
            world2joint[TransformsIndex.ORIENTATION]
        )
        
        # link COM -> world
        linkcm2world = p.invertTransform(
            linkState[LinkStateIndex.LINK_WORLD_POSITION], 
            linkState[LinkStateIndex.LINK_WORLD_ORIENTATION]
        )
        
        # link COM -> joint
        linkcm2joint = p.multiplyTransforms(
            linkcm2world[TransformsIndex.POSITION],
            linkcm2world[TransformsIndex.ORIENTATION],
            world2joint[TransformsIndex.POSITION],
            world2joint[TransformsIndex.ORIENTATION]
        )

        p.createConstraint(
            self.robotID,                           # parent body unique id
            -1,                                     # constraint to base
            self.robotID,                           # child body unique id
            self.linkNameToID[linkName],            # child link index
            p.JOINT_POINT2POINT,                    # joint type
            [0,0,0],                                # joint axis
            body2joint[TransformsIndex.POSITION],   # parent frame position [base link -> joint]
            linkcm2joint[TransformsIndex.POSITION], # child frame position [link COM -> joint]
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
    def setSuspensionParam(self, suspension_params):
        """
        Checks 'members' within the suspension_params. If members is not specified in the .yaml, 
        populate it with the default spring joints, and duplicates each value with the default 
        number of spring joints.

        Loads parameters into simulation through setJointMotorControl2 and changeDynamics.

        Args:
            suspension_params (dict) : dictionary containing the following parameters:
                1. 'preload'
                2. 'springConstant'
                3. 'dampingConstant'
                4. 'lowerLimit': joint lower limit
                5. 'upperLimit': joint upper limit
        """
        for key in suspension_params:
            suspension_params[key] = [suspension_params[key]] * len(self.SPRING_JOINTS)
            
        # All joints will be controlled using the same POSITION_CONTROL
        for member, preload, spring_constant, damping_constant, lower_limit, upper_limit in zip(
            self.SPRING_JOINTS,
            suspension_params['preload'],
            suspension_params['springConstant'],
            suspension_params['dampingconstant'],
            suspension_params['lowerLimit'],
            suspension_params['upperLimit']):

            # Sets joint control to POSITION_CONTROL, and updates joint with gains.
            p.setJointMotorControl2(bodyUniqueId    = self.robotID,
                                    jointIndex      = self.jointNameToID[member],
                                    controlMode     = p.POSITION_CONTROL,
                                    targetPosition  = lower_limit - preload,
                                    positionGain    = spring_constant,
                                    velocityGain    = damping_constant,
                                    physicsClientId = self.physicsClientId)
            
            # Changes the joint dynamics.
            p.changeDynamics(self.robotID,
                             self.jointNameToID[member],
                             jointLowerLimit    = lower_limit,
                             jointUpperLimit    = upper_limit,
                             physicsClientId    = self.physicsClientId)

    @checkRobotExists
    def setTireContactParam(self,tire_contact_params):
        for key in tire_contact_params:
            tire_contact_params[key] = [tire_contact_params[key]] * len(self.TIRE_NAMES)
            
        for i_tire in range(len(self.TIRE_NAMES)):
            p.changeDynamics(
                self.robotID,             
                self.linkNameToID[self.TIRE_NAMES[i_tire]],             
                lateralFriction =   tire_contact_params['lateralFriction'][i_tire],             
                restitution     =   tire_contact_params['restitution'][i_tire],             
                physicsClientId =   self.physicsClientId)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
    
    @checkRobotExists
    def drive(self, driveSpeed):
        driveSpeed = max(min(driveSpeed,1),-1)
        driveJoints = [self.jointNameToID[name] for name in self.WHEEL_TO_TIRE_JOINTS]
        
        scale = self.driveParams['scale']
        velocityGain = self.driveParams['velocityGain']
        maxForce = self.driveParams['maxForce']

        # rad/s
        targetVelocities = [driveSpeed * scale] * len(driveJoints)
        velocityGains = [velocityGain] * len(driveJoints)
        # N
        forces = [maxForce] * len(driveJoints)
                
        p.setJointMotorControlArray(
            self.robotID,
            driveJoints,
            p.VELOCITY_CONTROL,
            targetVelocities=targetVelocities,
            velocityGains=velocityGains,
            forces=forces,
            physicsClientId=self.physicsClientId)

    @checkRobotExists
    def steer(self, angle):
        # Convert angle into a list
        if not isinstance(angle, (list, torch.Tensor, np.ndarray)):
            angle = [angle]

        # Getting front / rear angles, and clamping
        frontAngle = angle[0]
        rearAngle = angle[1] if len(angle) > 1 else 0
        frontAngle = max(min(frontAngle,1),-1)
        rearAngle = max(min(rearAngle,1),-1)

        steerJoints = [self.jointNameToID[name] for name in self.AXLE_TO_WHEEL_JOINTS]
        steerAngles = [-frontAngle * self.steerParams['scale'] + self.steerParams['frontTrim'],
                       rearAngle * self.steerParams['scale'] + self.steerParams['backTrim']
                        ] * 2

        n_steerAngles = len(steerAngles)
        positionGains   = [self.steerParams['positionGain']] * n_steerAngles
        velocityGains   = [self.steerParams['velocityGain']] * n_steerAngles
        maxForces       = [self.steerParams['maxForce']] * n_steerAngles

        p.setJointMotorControlArray(self.robotID,
                                    steerJoints,
                                    p.POSITION_CONTROL,
                                    targetPositions=steerAngles,
                                    positionGains=positionGains,
                                    velocityGains=velocityGains,
                                    forces=maxForces,
                                    physicsClientId=self.physicsClientId)
