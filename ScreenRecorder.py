import os
import argparse
import math

class ScreenRecorder(object):
    def __init__(self,captureDirectory,timeStep,frameRate):
        import pyautogui
        self.pyautogui = pyautogui
        self.timeStep = timeStep
        self.frameDuration = 1.0/frameRate
        self.dir = captureDirectory
        if not os.path.isdir(self.dir):
            os.mkdir(self.dir)
        self.simTime = 0
        self.lastFrame = -1
        self.on = False
    def simStep(self):
        if not self.on:
            return
        self.simTime += self.timeStep
        if math.floor(self.simTime/self.frameDuration) > self.lastFrame:
            self.lastFrame = math.floor(self.simTime/self.frameDuration)
            screenshot = self.pyautogui.screenshot()
            screenshot.save(os.path.join(self.dir,str(self.lastFrame)+'.png'))
            print('saving')

if __name__=="__main__":
    import cv2
    parser = argparse.ArgumentParser(description='convert screenshots to video')
    parser.add_argument('dir')
    parser.add_argument('--fps',type=str,nargs='?',default=30)
    args = parser.parse_args()
    imageNums = [int(img.split('.')[0]) for img in os.listdir(args.dir) if img.endswith(".png")]
    imageNums.sort()
    frame = cv2.imread(os.path.join(args.dir,str(imageNums[0])+'.png'))
    height,width,layers = frame.shape
    size=(width,height)
    video = cv2.VideoWriter('combined.avi', 
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         int(args.fps), size)
    for num in imageNums:
        video.write(cv2.imread(os.path.join(args.dir,str(num)+'.png')))
    video.release()
