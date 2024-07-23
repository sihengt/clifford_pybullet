import pybullet as p
import numpy as np

def genParam(data,newParam=None,numSamples=1,genMean=False):
    if newParam is None:
        newParam = {}
    for key in data:
        if key == 'members':
            newParam[key] = data[key].copy()
        elif isinstance(data[key],dict):
            newParam[key]={}
            if 'members' in data[key]:
                genParam(data[key],newParam[key],len(data[key]['members']),genMean=genMean)
            else:
                genParam(data[key],newParam[key],numSamples,genMean=genMean)
        elif isinstance(data[key],list):
            if genMean:
                newParam[key] = (0.5*np.ones(numSamples)*(data[key][1]+data[key][0])).squeeze().tolist()
            else:
                newParam[key] = (np.random.rand(numSamples)*(data[key][1]-data[key][0])+data[key][0]).squeeze().tolist()
        else:
            newParam[key] = (np.ones(numSamples)*data[key]).squeeze().tolist()
    return newParam
    
def checkRobotExists(func):
    def decoratedFunc(*args, **kwargs):
        if args[0].robotID == None:
            return
        return func(*args, **kwargs)
    return decoratedFunc

class simRobot(object):
    def __init__(self,physicsClientId):
        self.physicsClientId=physicsClientId
        self.robotID = None
        self.measuredJointIDs = []

    @checkRobotExists
    def reset(self,pose=[[0,0,0.5],[0,0,0,1]]):
        p.resetBasePositionAndOrientation(self.robotID, pose[0],pose[1],physicsClientId=self.physicsClientId)
        nJoints = p.getNumJoints(self.robotID,physicsClientId=self.physicsClientId)
        initialJointPositions = [(0,) for i in range(nJoints)]
        initialJointVelocities = [(0,) for i in range(nJoints)]
        p.resetJointStatesMultiDof(self.robotID,range(nJoints),initialJointPositions,initialJointVelocities,physicsClientId=self.physicsClientId)

    @checkRobotExists
    def changeColor(self,color=None,tireKey='wheel',tireColor=None):
        nJoints = p.getNumJoints(self.robotID,physicsClientId=self.physicsClientId)
        if color is None:
            color = [0.6,0.1,0.1,1]
        if tireColor is None:
            tireColor = [0.15,0.15,0.15,1]
        for i in range(-1,nJoints):
            if i>-1 and tireKey in p.getJointInfo(self.robotID,i,physicsClientId=self.physicsClientId)[12].decode('UTF-8'):
                linkColor = tireColor
            else:
                linkColor = color
            p.changeVisualShape(self.robotID,i,rgbaColor=linkColor,specularColor=color,physicsClientId=self.physicsClientId)
    
    @checkRobotExists
    def buildModelDict(self):
        nJoints = p.getNumJoints(self.robotID,physicsClientId=self.physicsClientId)
        self.jointNameToID = {}
        self.linkNameToID = {}
        self.jointNames = []
        for i in range(nJoints):
            JointInfo = p.getJointInfo(self.robotID,i,physicsClientId=self.physicsClientId)
            self.jointNameToID[JointInfo[1].decode('UTF-8')] = JointInfo[0]
            self.linkNameToID[JointInfo[12].decode('UTF-8')] = JointInfo[0]
            self.jointNames.append(JointInfo[1].decode('UTF-8'))

    @checkRobotExists
    def measureJoints(self):
        if self.measuredJointIDs is None:
            nJoints = p.getNumJoints(self.robotID,physicsClientId=self.physicsClientId)
            self.measuredJointIDs = list(range(nJoints))
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
