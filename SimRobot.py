import pybullet as p
import numpy as np
from utils.checkRobotExists import checkRobotExists

class SimRobot:
    def __init__(self,physicsClientId):
        self.physicsClientId=physicsClientId
        self.robotID = None
        self.nJoints = None
        self.measuredJointIDs = []

    @checkRobotExists
    def reset(self,pose=[[0,0,0.5],[0,0,0,1]]):
        p.resetBasePositionAndOrientation(self.robotID, pose[0],pose[1],physicsClientId=self.physicsClientId)
        initialJointPositions = [(0,) for i in range(self.nJoints)]
        initialJointVelocities = [(0,) for i in range(self.nJoints)]
        p.resetJointStatesMultiDof(self.robotID,range(self.nJoints),initialJointPositions,initialJointVelocities,physicsClientId=self.physicsClientId)

    @checkRobotExists
    def changeColor(self,color=None,tireKey='wheel',tireColor=None):
        if color is None:
            color = [0.6, 0.1, 0.1, 1]
        if tireColor is None:
            tireColor = [0.15, 0.15, 0.15, 1]
        for i in range(-1, self.nJoints):
            if i > -1 and tireKey in p.getJointInfo(self.robotID,i,physicsClientId=self.physicsClientId)[12].decode('UTF-8'):
                linkColor = tireColor
            else:
                linkColor = color
            p.changeVisualShape(self.robotID,i,rgbaColor=linkColor,specularColor=color,physicsClientId=self.physicsClientId)
    
    @checkRobotExists
    def buildModelDict(self):
        """ Populates self.jointNameToID, self.linkNameToID and self.jointNames """
        self.jointNameToID = {}
        self.linkNameToID = {}
        self.jointNames = []

        # Iterate through each joint
        JOINT_INDEX = 0
        JOINT_NAME = 1
        LINK_NAME = 12

        for i_joint in range(self.nJoints):
            jointInfo = p.getJointInfo(self.robotID, i_joint, physicsClientId=self.physicsClientId)
            self.jointNameToID[jointInfo[JOINT_NAME].decode('UTF-8')] = jointInfo[JOINT_INDEX]
            self.linkNameToID[jointInfo[LINK_NAME].decode('UTF-8')] = jointInfo[JOINT_INDEX]
            self.jointNames.append(jointInfo[JOINT_NAME].decode('UTF-8'))

    @checkRobotExists
    def measureJoints(self):
        if self.measuredJointIDs is None:
            self.measuredJointIDs = list(range(self.nJoints))
        jointStates = p.getJointStates(self.robotID,self.measuredJointIDs,physicsClientId=self.physicsClientId)
        if jointStates is None:
            return []
        return [jointState[0] for jointState in jointStates] + [jointState[1] for jointState in jointStates]

    @checkRobotExists
    def getBasePositionOrientation(self,calcHeadingTilt=False):
        output = p.getBasePositionAndOrientation(self.robotID,physicsClientId=self.physicsClientId)
        Rwb = output[1]
        if calcHeadingTilt:
            forwardDir = p.multiplyTransforms([0,0,0],Rwb,[1,0,0],[0,0,0,1])[0]
            headingAngle = np.arctan2(forwardDir[1],forwardDir[0])
            Rbw = p.invertTransform([0,0,0],Rwb)[1]
            upDir = p.multiplyTransforms([0,0,0],Rbw,[0,0,1],[0,0,0,1])[0]
            tiltAngles = (np.arccos(upDir[2]),
                        np.arctan2(upDir[1],upDir[0]))
            output+=(headingAngle,tiltAngles)
        return output
    
    @checkRobotExists
    def getBaseVelocity_body(self):
        gwb = p.getBasePositionAndOrientation(self.robotID,physicsClientId=self.physicsClientId)
        Rbw = p.invertTransform(gwb[0],gwb[1])[1]
        Vw = p.getBaseVelocity(self.robotID,physicsClientId=self.physicsClientId)
        v_b = p.multiplyTransforms([0,0,0],Rbw,Vw[0],[0,0,0,1])[0]
        w_b = p.multiplyTransforms([0,0,0],Rbw,Vw[1],[0,0,0,1])[0]
        return list(v_b)+list(w_b)
    
    @checkRobotExists
    def getBaseVelocity_world(self):
        Vw = p.getBaseVelocity(self.robotID,physicsClientId=self.physicsClientId)
        return list(Vw[0])+list(Vw[1])

    @checkRobotExists
    def plotLinks(self,lineLength=0.25,lineWidth=2):
        for i in range(-1,p.getNumJoints(self.robotID)):
            p.addUserDebugLine([0,0,0],[0,lineLength,0],lineWidth=lineWidth,lineColorRGB=[0,1,0],parentObjectUniqueId=self.robotID,parentLinkIndex=i,physicsClientId = self.physicsClientId)
            p.addUserDebugLine([0,0,0],[0,0,lineLength],lineWidth=lineWidth,lineColorRGB=[0,0,1],parentObjectUniqueId=self.robotID,parentLinkIndex=i,physicsClientId = self.physicsClientId)
            p.addUserDebugLine([0,0,0],[lineLength,0,0],lineWidth=lineWidth,lineColorRGB=[1,0,0],parentObjectUniqueId=self.robotID,parentLinkIndex=i,physicsClientId = self.physicsClientId)
