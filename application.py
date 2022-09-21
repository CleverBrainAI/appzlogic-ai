import os
# import cv2
import random
import jsonpickle
import numpy as np
import tensorflow as tf


import sys
# sys.path.append("..")

# Import utilites
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
from flask import Flask, request, Response


application = Flask(__name__)


@application.route('/', methods=['GET','POST'])
def welcome():
    return 'Welcome'


# @application.route('/predict_image', methods=['GET','POST'])
# def predict_image():

#     nparr = np.fromstring(request.data, np.uint8)
#     image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

#     PATH_TO_CKPT = 'frozen_inference_graph.pb'
#     PATH_TO_LABELS = 'labelmap.pbtxt'

#     NUM_CLASSES = 1

#     label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
#     categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
#     category_index = label_map_util.create_category_index(categories)

#     detection_graph = tf.Graph()
#     with detection_graph.as_default():
#         od_graph_def = tf.GraphDef()
#         with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
#             serialized_graph = fid.read()
#             od_graph_def.ParseFromString(serialized_graph)
#             tf.import_graph_def(od_graph_def, name='')

#         sess = tf.Session(graph=detection_graph)

#     image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
#     detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
#     detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
#     detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
#     num_detections = detection_graph.get_tensor_by_name('num_detections:0')

#     #image = cv2.imread(PATH_TO_IMAGE)
#     image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     image_expanded = np.expand_dims(image_rgb, axis=0)

#     (boxes, scores, classes, num) = sess.run(
#         [detection_boxes, detection_scores, detection_classes, num_detections],
#         feed_dict={image_tensor: image_expanded})

#     vis_util.visualize_boxes_and_labels_on_image_array(
#         image,
#         np.squeeze(boxes),
#         np.squeeze(classes).astype(np.int32),
#         np.squeeze(scores),
#         category_index,
#         use_normalized_coordinates=True,
#         skip_scores=True,
#         line_thickness=8,
#         min_score_thresh=0.6)

#     predict_img_path = 'pred' 
#     if not os.path.exists(predict_img_path):
#         os.makedirs(predict_img_path) 
  
#     img_name = predict_img_path + '/output_' + str(random.randint(1000,9999)) + '.jpg'

#     cv2.imwrite(img_name, image)

#     response = {'message': 'image received. size={}x{}'.format(image.shape[1], image.shape[0])
#                 }
    
#     response_pickled = jsonpickle.encode(response)

#     return Response(response=response_pickled, status=200, mimetype="application/json")
    
