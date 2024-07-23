import pybullet as p
import numpy as np
import os,sys
cliford_dir = os.path.dirname(__file__)
sys.path.append(os.path.dirname(cliford_dir))
from simRobot import simRobot,checkRobotExists

class Clifford(simRobot):
    def __init__(self,physicsClientId=0):
        super().__init__(physicsClientId)
        self.sdfPath = os.path.join(cliford_dir,'clifford.sdf')
    
    def setParams(self,params,importHeight=10,**kwargs):
        self.driveParams = params['drive']
        self.steerParams = params['steer']
        if self.robotID is None:
            self.importClifford(importHeight)
            self.changeColor()
            self.loosenModel()
        self.setSuspensionParam(params['suspension'])
        self.setInertialParam(params['inertia'])
        self.setTireContactParam(params['tireContact'])
    
    def importClifford(self,importHeight=10):
        # load sdf file (this file defines open chain clifford. need to add constraints to make closed chain)
        self.robotID = p.loadSDF(self.sdfPath,physicsClientId=self.physicsClientId)[0]
        p.resetBasePositionAndOrientation(self.robotID,(0,0,importHeight),(0,0,0,1),physicsClientId=self.physicsClientId)

        # define number of joints of clifford robot
        nJoints = p.getNumJoints(self.robotID,physicsClientId=self.physicsClientId)
        initialJointStates = p.getJointStatesMultiDof(self.robotID,range(nJoints),physicsClientId=self.physicsClientId)
        self.initialJointPositions = [initialJointStates[i][0] for i in range(nJoints)]
        self.initialJointVelocities = [initialJointStates[i][1] for i in range(nJoints)]
        self.buildModelDict()
        self.linkNameToID['base'] = -1
        self.recordInitialInertia()

        # make closed chain
        linkFrame2Joint ={}
        linkFrame2Joint['upperSpring'] = [0,0,0]
        linkFrame2Joint['outer'] = [0.23,0,0]
        linkFrame2Joint['inner'] = [0.195,0,0]
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

    @checkRobotExists
    def reset(self,pose=((0,0,0.25),(0,0,0,1))):
        p.resetBasePositionAndOrientation(self.robotID, pose[0],pose[1],physicsClientId=self.physicsClientId)
        nJoints = p.getNumJoints(self.robotID,physicsClientId=self.physicsClientId)
        p.resetJointStatesMultiDof(self.robotID,range(nJoints),self.initialJointPositions,self.initialJointVelocities,physicsClientId=self.physicsClientId)
    
    @checkRobotExists
    def loosenModel(self):
        nJoints = p.getNumJoints(self.robotID,physicsClientId=self.physicsClientId)
        for i in range(nJoints):
            if len(p.getJointStateMultiDof(bodyUniqueId=self.robotID,jointIndex=i,physicsClientId=self.physicsClientId)[0]) == 4:
                p.setJointMotorControlMultiDof(bodyUniqueId=self.robotID,
                                                jointIndex=i,
                                                controlMode=p.POSITION_CONTROL,
                                                targetPosition=[0,0,0,1],
                                                positionGain=0,
                                                velocityGain=0,
                                                force=[0,0,0],physicsClientId=self.physicsClientId)
        c = p.createConstraint(self.robotID,
                               self.linkNameToID['blwheel'],
                               self.robotID,
                               self.linkNameToID['brwheel'],
                               jointType=p.JOINT_GEAR,
                               jointAxis=[0, 1, 0],
                               parentFramePosition=[0, 0, 0],
                               childFramePosition=[0, 0, 0],physicsClientId=self.physicsClientId)
        p.changeConstraint(c, gearRatio=-1, maxForce=10000,erp=0.2,physicsClientId=self.physicsClientId)
        c = p.createConstraint(self.robotID,
                               self.linkNameToID['flwheel'],
                               self.robotID,
                               self.linkNameToID['frwheel'],
                               jointType=p.JOINT_GEAR,
                               jointAxis=[0, 1, 0],
                               parentFramePosition=[0, 0, 0],
                               childFramePosition=[0, 0, 0],physicsClientId=self.physicsClientId)
        p.changeConstraint(c, gearRatio=-1, maxForce=10000,erp=0.2,physicsClientId=self.physicsClientId)

    @checkRobotExists
    def recordInitialInertia(self):
        self.linkInertias = {}
        self.tireMass = 0
        self.baseMass = 0
        self.otherMass = 0
        for linkName in self.linkNameToID:
            dynamicsInfo = p.getDynamicsInfo(self.robotID,self.linkNameToID[linkName],physicsClientId=self.physicsClientId)
            self.linkInertias[linkName] = (dynamicsInfo[0],dynamicsInfo[2][0],dynamicsInfo[2][1],dynamicsInfo[2][2])
            if 'tire' in linkName:
                self.tireMass += dynamicsInfo[0]
            elif linkName == 'base':
                self.baseMass = dynamicsInfo[0]
            else:
                self.otherMass += dynamicsInfo[0]
    
    @checkRobotExists
    def setInertialParam(self,inertialParam):
        totalWeightedMass = self.otherMass + self.tireMass*inertialParam['tireRelDensity'] + self.baseMass*inertialParam['baseRelDensity']
        for linkName in self.linkNameToID:
            linkMassScale = 1
            if 'tire' in linkName:
                linkMassScale = inertialParam['tireRelDensity']
            elif linkName == 'base':
                linkMassScale = inertialParam['baseRelDensity']
            massScale = linkMassScale*inertialParam['totalMass']/totalWeightedMass

            linkMass = self.linkInertias[linkName][0]*massScale
            linkInertia = [self.linkInertias[linkName][1]*massScale,
                            self.linkInertias[linkName][2]*massScale,
                            self.linkInertias[linkName][3]*massScale]
            p.changeDynamics(self.robotID,self.linkNameToID[linkName],physicsClientId=self.physicsClientId,
                            mass=linkMass,localInertiaDiagonal=linkInertia)

        totalMass = 0
        for linkName in self.linkNameToID:
            linkMass=p.getDynamicsInfo(self.robotID,self.linkNameToID[linkName],physicsClientId=self.physicsClientId)[0]
            totalMass += linkMass

    @checkRobotExists
    def setSuspensionParam(self,suspensionParams):
        springJointNames = ['brslower2upper','blslower2upper','frslower2upper','flslower2upper']
        if not 'members' in suspensionParams or len(suspensionParams['members']) == 1:
            for key in suspensionParams:
                suspensionParams[key] = [suspensionParams[key]]*len(springJointNames)
            suspensionParams['members'] = springJointNames
        for member,preload,springConstant,dampingconstant,lowerLimit,upperLimit in \
            zip(suspensionParams['members'],suspensionParams['preload'],suspensionParams['springConstant'],
                suspensionParams['dampingconstant'],suspensionParams['lowerLimit'],suspensionParams['upperLimit']):
            p.setJointMotorControl2(bodyUniqueId=self.robotID,
                                    jointIndex=self.jointNameToID[member],
                                    controlMode=p.POSITION_CONTROL,
                                    targetPosition=lowerLimit-preload,
                                    positionGain=springConstant,
                                    velocityGain=dampingconstant,
                                    physicsClientId=self.physicsClientId)
            p.changeDynamics(self.robotID,self.jointNameToID[member],
                            jointLowerLimit=lowerLimit,
                            jointUpperLimit=upperLimit,
                            physicsClientId=self.physicsClientId)

    @checkRobotExists
    def changeColor(self,color=None):
        nJoints = p.getNumJoints(self.robotID,physicsClientId=self.physicsClientId)
        if color is None:
            color = [0.6,0.1,0.1,1]
        for i in range(-1,nJoints):
            p.changeVisualShape(self.robotID,i,rgbaColor=color,specularColor=color,physicsClientId=self.physicsClientId)
        tires = ['frtire','fltire','brtire','bltire']
        for tire in tires:
            p.changeVisualShape(self.robotID,self.linkNameToID[tire],rgbaColor=[0.15,0.15,0.15,1],specularColor=[0.15,0.15,0.15,1],physicsClientId=self.physicsClientId)
    
    @checkRobotExists
    def addClosedChainConstraint(self,linkName,linkFrame2Joint):
        linkState = p.getLinkState(self.robotID,self.linkNameToID[linkName],physicsClientId=self.physicsClientId)
        bodyState = p.getBasePositionAndOrientation(self.robotID,physicsClientId=self.physicsClientId)
        world2joint = p.multiplyTransforms(linkState[4],linkState[5],linkFrame2Joint,[0,0,0,1])
        body2world = p.invertTransform(bodyState[0],bodyState[1])
        body2joint = p.multiplyTransforms(body2world[0],body2world[1],world2joint[0],world2joint[1])
        linkcm2world = p.invertTransform(linkState[0],linkState[1])
        linkcm2joint = p.multiplyTransforms(linkcm2world[0],linkcm2world[1],world2joint[0],world2joint[1])

        c = p.createConstraint(self.robotID,-1,self.robotID,self.linkNameToID[linkName],p.JOINT_POINT2POINT,[0,0,0],body2joint[0],linkcm2joint[0],physicsClientId=self.physicsClientId)
    
    @checkRobotExists
    def setTireContactParam(self,tireContactParam):
        tireNames = ['frtire','fltire','brtire','bltire']
        if len(tireContactParam['members']) == 1:
            for key in tireContactParam:
                tireContactParam[key] = [tireContactParam[key]]*len(tireNames)
            tireContactParam['members'] = tireNames
        for i in range(len(tireContactParam['members'])):
            p.changeDynamics(self.robotID,self.linkNameToID[tireContactParam['members'][i]],physicsClientId=self.physicsClientId,
                    lateralFriction=tireContactParam['lateralFriction'][i],
                    restitution=tireContactParam['restitution'][i])
                    #contactStiffness=tireContactParam['contactStiffness'][i],
                    #contactDamping=tireContactParam['contactDamping'][i])

    @checkRobotExists
    def drive(self,driveSpeed):
        driveSpeed = max(min(driveSpeed,1),-1)
        driveJoints = [self.jointNameToID[name] for name in ['frwheel2tire','flwheel2tire','brwheel2tire','blwheel2tire']]
        p.setJointMotorControlArray(self.robotID,
                                    driveJoints,
                                    p.VELOCITY_CONTROL,
                                    targetVelocities=[driveSpeed*self.driveParams['scale']]*len(driveJoints),
                                    velocityGains=[self.driveParams['velocityGain']]*len(driveJoints),
                                    forces=[self.driveParams['maxForce']]*len(driveJoints),
                                    physicsClientId=self.physicsClientId)

    @checkRobotExists
    def steer(self,angle):
        if not hasattr(angle,'__len__'):
            angle = [angle]
        frontAngle = angle[0]
        rearAngle = angle[1] if len(angle) > 1 else 0
        frontAngle = max(min(frontAngle,1),-1)
        rearAngle = max(min(rearAngle,1),-1)

        steerJoints = [self.jointNameToID[name] for name in ['axle2frwheel','axle2brwheel','axle2flwheel','axle2blwheel']]
        steerAngles = [-frontAngle*self.steerParams['scale']+self.steerParams['frontTrim'],
                (rearAngle*self.steerParams['scale']+self.steerParams['backTrim'])]*2 
        p.setJointMotorControlArray(self.robotID,
                                    steerJoints,
                                    p.POSITION_CONTROL,
                                    targetPositions=steerAngles,
                                    positionGains=[self.steerParams['positionGain']]*len(steerAngles),
                                    velocityGains=[self.steerParams['velocityGain']]*len(steerAngles),
                                    forces=[self.steerParams['maxForce']]*len(steerAngles),
                                    physicsClientId=self.physicsClientId)
