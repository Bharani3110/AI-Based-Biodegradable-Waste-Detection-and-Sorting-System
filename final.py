import numpy as np
import os
import sys
import tensorflow as tf
from distutils.version import StrictVersion
from collections import defaultdict
from PIL import Image
from object_detection.utils import ops as utils_ops
import serial  # To communicate with hardware via serial port
import time

# Check TensorFlow version
if StrictVersion(tf.__version__) < StrictVersion('1.9.0'):
    raise ImportError('Please upgrade your TensorFlow installation to v1.9.* or later!')

from utils import label_map_util
from utils import visualization_utils as vis_util

MODEL_NAME = 'inference_graph'
PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'
PATH_TO_LABELS = 'training/labelmap.pbtxt'

# Setup serial communication (replace 'COM3' with your port)
ser = serial.Serial('COM5', 9600)  # Change 'COM3' to your device's serial port
time.sleep(2)  # Wait for the serial connection to initialize

# Load detection graph
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

def run_inference_for_single_image(image, graph):
    print("Running inference for single image...")  # Debugging line
    # TensorFlow inference logic
    image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
    tensor_dict = {
        'num_detections': tf.get_default_graph().get_tensor_by_name('num_detections:0'),
        'detection_boxes': tf.get_default_graph().get_tensor_by_name('detection_boxes:0'),
        'detection_scores': tf.get_default_graph().get_tensor_by_name('detection_scores:0'),
        'detection_classes': tf.get_default_graph().get_tensor_by_name('detection_classes:0')
    }

    # Check if 'detection_masks' exists in the graph and add to tensor_dict if it does
    if 'detection_masks' in [output.name for op in tf.get_default_graph().get_operations() for output in op.outputs]:
        tensor_dict['detection_masks'] = tf.get_default_graph().get_tensor_by_name('detection_masks:0')
    
    # Run inference
    output_dict = sess.run(tensor_dict, feed_dict={image_tensor: np.expand_dims(image, 0)})
    
    output_dict['num_detections'] = int(output_dict['num_detections'][0])
    output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
    output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
    output_dict['detection_scores'] = output_dict['detection_scores'][0]
    
    # If masks exist, add them to output_dict
    if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
        
    return output_dict

a1 = 0
a2 = 0
import cv2

cap = cv2.VideoCapture(0)

try:
    with detection_graph.as_default():
        with tf.Session() as sess:
            print("Session started!")  # Debugging line
            # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
                'num_detections', 'detection_boxes', 'detection_scores',
                'detection_classes', 'detection_masks'
            ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)

            while True:
                # Wait for data from hardware (e.g., 'A' means start detection)
                data = ser.readline().decode().strip()  # Read hardware data
                print(f"Received data: {data}")  # Debugging line

                if data == 'A':  # Check if data is 'A'
                    print("Received 'A' from hardware, starting object detection...")

                    # Keep reading frames from the camera until an object is detected
                    while True:
                        # Capture image from camera
                        (__, image_np) = cap.read()
                        if image_np is None:
                            print("Failed to capture image from the camera.")  # Debugging line
                            continue

                        cv2.imwrite('capture.jpg', image_np)
                        print("Image captured.")  # Debugging line

                        # Run object detection
                        output_dict = run_inference_for_single_image(image_np, detection_graph)
                        
                        # Visualization
                        vis_util.visualize_boxes_and_labels_on_image_array(
                            image_np,
                            output_dict['detection_boxes'],
                            output_dict['detection_classes'],
                            output_dict['detection_scores'],
                            category_index,
                            instance_masks=output_dict.get('detection_masks'),
                            use_normalized_coordinates=True,
                            line_thickness=8)

                        # Check if any object is detected
                        detected = False  # Flag to check if an object is detected
                        for i in range(output_dict['num_detections']):
                            if output_dict['detection_scores'][i] > 0.70:
                                class_id = output_dict['detection_classes'][i]
                                if class_id == 1:  # Apple
                                    print("Apple detected!")
                                    ser.write(b'1')  # Send '1' to hardware
                                    detected = True
                                    break
                               
                                elif class_id == 3:  # Pineapple
                                    print("Rotten Pineapple detected!")
                                    ser.write(b'2')  # Send '1' to hardware
                                    detected = True
                                    break

                        if detected:
                            print("Fruit detected and response sent to hardware.")
                            break  # Exit the detection loop

                        # Display the image
                        cv2.imshow('Object Detection', cv2.resize(image_np, (800, 600)))
                        
                        # Allow the loop to continue or break on 'q' key press
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            cap.release()
                            cv2.destroyAllWindows()
                            break

                else:
                    print("Waiting for 'A' from hardware...")

except Exception as e:
    print("Error:", e)
    cap.release()
    cv2.destroyAllWindows()
