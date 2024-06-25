import cv2
import asyncio
import gpiod
import time
 
LED_PIN = 17
chip = gpiod.Chip('gpiochip4')
led_line = chip.get_line(LED_PIN)
led_line.request(consumer="LED", type=gpiod.LINE_REQ_DIR_OUT)
 
#thres = 0.45 # Threshold to detect object
classNames = []
classFile = "Files/coco.names"
with open(classFile,"rt") as f:
    classNames = f.read().rstrip("\n").split("\n")
 
configPath = "Files/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = "Files/frozen_inference_graph.pb"
 
net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)
 
def getObjects(img, thres, nms, draw=True, objects=[]):
    classIds, confs, bbox = net.detect(img,confThreshold=thres,nmsThreshold=nms)
    if len(objects) == 0: objects = classNames
    objectInfo =[]
    if len(classIds) != 0:
        for classId, confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
            className = classNames[classId - 1]
            if className in objects:
                objectInfo.append([box,className])
                if (draw):
                    cv2.rectangle(img,box,color=(0,255,0),thickness=2)
                    cv2.putText(img,classNames[classId-1].upper(),(box[0]+10,box[1]+30),
                    cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
 
    return img,objectInfo
 
def set_led(value):
    global led_line
    led_line.set_value(value)
 
async def main():
    cap = cv2.VideoCapture(0)
    cap.set(3,640)
    cap.set(4,480)
 
    while True:
        success, img = cap.read()
        result, objectInfo = getObjects(img,0.45,0.2, objects=['scissors'])
        # print(objectInfo)
 
        found = any('scissors' in sublist for sublist in objectInfo)
        if found:
            set_led(1)
            time.sleep(2)
            set_led(0)
            time.sleep(1)
 
        cv2.imshow("Output",img)
        cv2.waitKey(1)
 
if __name__ == "__main__":
    asyncio.run(main())
