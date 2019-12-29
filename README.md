# Object Detection with YOLOv3-416

## Notice
* yolov3.weights (416) must be on the same level with other files. 
* Frame skips of video and webcam can be changed from Line 43. It detects the object in every 3 frames as default.

### yolov3.weight can be downloaded from here
https://pjreddie.com/media/files/yolov3.weights

## Usage

* Webcam: objectDetection.py webcam
* Video: objectDetection.py 'Path of the video'
* Image: objectDetection.py 'Path of the image'

### Example

objectDetection.py test.jpg

![Object Detection](https://user-images.githubusercontent.com/42182119/71558653-8daa4b00-2a66-11ea-9151-d4007d5bbccf.jpg)
