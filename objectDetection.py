import cv2
import numpy as np
import mimetypes
import sys

# Load Yolo
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

if(sys.argv[1] == 'webcam'):
    mimestart = 'webcam'
else:
    mimetypes.init()
    
    mimestart = mimetypes.guess_type(sys.argv[1])[0]
    
    if mimestart != None:
        mimestart = mimestart.split('/')[0]
           
if mimestart == 'video' or mimestart == 'webcam':

    counter = -1
    #read video file
    if mimestart == 'video': 
        cap = cv2.VideoCapture(sys.argv[1])
    else:
        cap = cv2.VideoCapture(0)
    while (True):
        #if ret is true than no error with cap.isOpened
        ret, frame = cap.read()
    
        if ret==True:
            
            frame = cv2.resize(frame, None, fx=2.2, fy=2.2)
            height, width, channels = frame.shape
            counter = counter + 1
            if (counter%3 == 0):
                # Detecting objects
                blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            
                net.setInput(blob)
                outs = net.forward(output_layers)
        
            # Showing informations on the screen
            class_ids = []
            confidences = []
            boxes = []
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.5:
                        # Object detected
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
            
                        # Rectangle coordinates
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
            
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)                   
            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
            #print(indexes)
            
            font = cv2.FONT_HERSHEY_PLAIN
            for i in range(len(boxes)):
                if i in indexes:
                    x, y, w, h = boxes[i]
                    label = str(classes[class_ids[i]] + "  " + str(round(confidences[i], 2)))
                    color = colors[i]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 1)
                    cv2.putText(frame, label, (x, y + 30), font, 2, color, 3)        
            cv2.imshow("frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
    cap.release()
    cv2.destroyAllWindows()
else:
    # Loading image
    img = cv2.imread(sys.argv[1])
    img = cv2.resize(img, None, fx=0.8, fy=0.8)
    height, width, channels = img.shape
    
    # Detecting objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    
    net.setInput(blob)
    outs = net.forward(output_layers)
    
    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
    
                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
    
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)    
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:       
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]] + "  " + str(round(confidences[i], 2)))
            color = colors[i]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 1)
            cv2.putText(img, label, (x, y + 30), font, 2, color, 3)
    
    
    cv2.imshow("Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()