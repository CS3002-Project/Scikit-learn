# CG3002 Dance Dance Project August 2019 Group 3

## Project Description
The wearable system can detect and send predefined dance moves to a server. Our system operates on a 5V supply.


## Comms
The Comms secction consist of the Arduino code file 
```
cg3002_comms.ino 
```
written in C and the Raspberry Pi code file 
```
rpi_process_new_final.py 
```
written in Python. The Arduino code is responsible for getting the sensor data and sending in serially to the Raspberry Pi. The Raspberry Pi code receives the sensor data from the Arduino, loads machine learning model and makes prediction of a dance move which is sent to the server.

## System design

![Prediction System Design](prediction_system.png)

## Quick start
1. Contact the authors for dataset
2. For training and benchmarking 4 models Random Forest, Multi-layer Perceptron, K-Nearest Neighbors, Support Vector Machine, run
```
python dance.py
```

Benchmarked results on accuracy, prediction time and training time are shown below
![Benchmarked Accuracy](accuracy.png)

![Benchmarked Prediction Time](prediction_time.png)

![Benchmarked Training Time](training_time.png)

## Authors
* Syed Ahmad Alkaff
* Aditya Agarwal
* Aw Wen Hao
* Tan Ken Hao Zachary
* Nguyen Van Hoang
* Lenald Ng Wai Jun
