#
# MIT License
#
# Copyright (c) 2019 Taher Ahmadi 14taher@gmail.com
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

#!/usr/bin/env python

import tensorflow as tf
import sys
import os
import argparse
import time
import datetime
from utils import *
from trinet import *
import cvlib as cv
from cvlib.object_detection import draw_bbox
from shape_utils import Rectangle

# forces tensorflow to run on CPU
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

parser = argparse.ArgumentParser(description='Argument parser')

""" Arguments related to network architecture"""
parser.add_argument('--width', dest='width', type=int, default=512, help='width of input images')
parser.add_argument('--height', dest='height', type=int, default=256, help='height of input images')
parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', type=str, default='checkpoint/3DV18/3net', help='checkpoint directory')
parser.add_argument('--mode', dest='mode', type=int, default=0, help='Select the demo mode [0: depth-from-mono, 1:view synthesis, 2:stereo]')

# Norm. factors for visualization
DEPTH_FACTOR = 15
DISP_FACTOR = 6

args = parser.parse_args()

# function to get the output layer names 
# in the architecture
def get_output_layers(net):
    
    layer_names = net.getLayerNames()
    
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers

# function to draw bounding box on the detected object with class name
def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):

#    label = str(classes[class_id])

#    color = COLORS[class_id]
    color = (255,255,0)
    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 1)

    # cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def callback(data):
    img = bridge.imgmsg_to_cv2(data, "bgr8")
    img = cv2.resize(img, (width, height)).astype(np.float32) / 255.
    img_batch = np.expand_dims(img, 0)

    # Run 3net!!!
    start = time.time()
    disp_cr, disp_cl, synt_left, synt_right = sess.run([model.disparity_cr, model.disparity_cl, model.warp_left, model.warp_right], feed_dict={placeholders['im0']: img_batch})
    disp = build_disparity(disp_cr, disp_cl)
    print(disp.max(), disp.min())
    end = time.time()

    # Bring back images to uint8 and prepare visualization
    img = (img*255).astype(np.uint8)
    synt_left = (synt_left*255).astype(np.uint8)
    synt_right = (synt_right*255).astype(np.uint8)
    disp_color = (applyColorMap(disp*DEPTH_FACTOR, 'jet')*255).astype(np.uint8)
    if args.mode == 0:
        toShow_C = cv2.addWeighted(img,0.1,disp_color,0.8,0) # merge left and right into one
    else:
        toShow_C = np.concatenate((img, disp_color), 1)

    narrow_disp = np.zeros_like(synt_left)
    wide_disp = np.zeros_like(synt_left)
    # If synthetic views are active
    if args.mode==0:
      synt_left = np.zeros_like(synt_left)
      synt_right = np.zeros_like(synt_left)
    else:
      # If SGM is active
      if args.mode>1:
        sgm = cv2.StereoSGBM(0,128,1)
        narrow_disp = sgm.compute(synt_left[:,:,0:1], img[:,:,0:1])
        wide_disp = sgm.compute(synt_left[:,:,0:1], synt_right[:,:,0:1])
        cv2.filterSpeckles(narrow_disp, 0, 4000, 16)
        cv2.filterSpeckles(wide_disp, 0, 4000, 16)
        mask = np.expand_dims(np.uint8(narrow_disp > 0),-1)
        narrow_disp = cv2.applyColorMap(np.uint8(narrow_disp/DISP_FACTOR), 2)*mask
        mask = np.expand_dims(np.uint8(wide_disp > 0),-1)
        wide_disp = cv2.applyColorMap(np.uint8(wide_disp/DISP_FACTOR), 2)*mask

    # Build final output
    # toShow_L = (np.concatenate((synt_left, narrow_disp), 1)).astype(np.uint8)
    # toShow_R = (np.concatenate((synt_right, wide_disp), 1)).astype(np.uint8)
    # toShow = np.concatenate((toShow_L, toShow_C, toShow_R), 0)
    # toShow = cv2.resize(toShow, (width*2, height*2))

    # Show cool visualization
    cv2.imshow('3net', toShow_C)

    print("Time: " + str(end - start))


def main(_):
  with tf.Graph().as_default():
    height = args.height
    width = args.width
    center_box = Rectangle(int(width * 0.3), int(height * 0.25), int(width * 0.7), int(height * 0.8))

    placeholders = {'im0':tf.placeholder(tf.float32,[None, None, None, 3], name='im0')}
    model = trinet(placeholders,net='resnet50')

    init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    loader = tf.train.Saver()
    saver = tf.train.Saver()
    cam = cv2.VideoCapture('/home/taher/workspace/BSc/BSc/test/20190419_121630.mp4')
    net = cv2.dnn.readNet('/home/taher/workspace/BSc/BSc/darknet_conf/tiny-yolo-voc.weights', '/home/taher/workspace/BSc/BSc/darknet_conf/tiny-yolo-voc.cfg')
    with tf.Session() as sess:
        sess.run(init)
        loader.restore(sess, args.checkpoint_dir)
        while True:
          for i in range(2):
            cam.grab()
          ret_val, img = cam.read()

          # Prepare input to the network
          img = cv2.resize(img, (width, height)).astype(np.float32) / 255.
          img_batch = np.expand_dims(img, 0)

          # Run 3net!!!
          start = time.time()
          
          disp_cr, disp_cl, synt_left, synt_right = sess.run([model.disparity_cr, model.disparity_cl, model.warp_left, model.warp_right], feed_dict={placeholders['im0']: img_batch})
          disp = build_disparity(disp_cr, disp_cl)
          

          # Bring back images to uint8 and prepare visualization
          img = (img*255).astype(np.uint8)
          #run dnn yolo3
          if img is None:
                continue
          begin = time.time()
          rgb_show_img = img.copy()
          Width = img.shape[1]
          Height = img.shape[0]
          scale = 0.00392
          blob = cv2.dnn.blobFromImage(img, scale, (512,256), (0,0,0), True, crop=False)
          net.setInput(blob)
          outs = net.forward(get_output_layers(net))
          class_ids = []
          confidences = []
          boxes = []
          conf_threshold = 0.5
          nms_threshold = 0.4
          for out in outs:
              for detection in out:
                  scores = detection[5:]
                  class_id = np.argmax(scores)
                  confidence = scores[class_id] 
                  if confidence > 0.5:
                      center_x = int(detection[0] * Width)
                      center_y = int(detection[1] * Height)
                      w = int(detection[2] * Width)
                      h = int(detection[3] * Height)
                      x = center_x - w / 2
                      y = center_y - h / 2
                      class_ids.append(class_id)
                      confidences.append(float(confidence))
                      boxes.append([x, y, w, h])
          indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
          print('rgb time = ', str(time.time() - begin))

          #synt_left = (synt_left*255).astype(np.uint8)
          #synt_right = (synt_right*255).astype(np.uint8)
          if args.mode in [-1,3]:
            disp_color = (applyColorMap(disp*DEPTH_FACTOR, 'magma')*255).astype(np.uint8)
          elif args.mode in [1,2]:
            disp_color = (applyColorMap(disp*DEPTH_FACTOR, 'magma')*255).astype(np.uint8)
          
          if args.mode == -1 or args.mode == 3:
              toShow_C = cv2.addWeighted(img,0.1,disp_color,0.8,0) # merge disparity and img
              overlay = toShow_C.copy()
              # draw rectangle
              toShow_C = cv2.rectangle(toShow_C,center_box.upper_left(),center_box.bottom_right(),(100,100,100),1)
              for i in indices:
                  i = i[0]
                  box = boxes[i]
                  x = box[0]
                  y = box[1]
                  w = box[2]
                  h = box[3]

                  rect = Rectangle(x,y,x+w,y+h)
                  intersection = center_box&rect
                  if intersection is not None:
                      alpha = 0.4
                      print(intersection)
                      print(intersection.y1)
                      print('intersection..................................')
                      print('disp_shape: ',disp.shape)
                      crop_img = disp[int(intersection.y1):int(intersection.y2),
                                           int(intersection.x1):int(intersection.x2)].copy()
                      # print('crop: ',crop_img*DEPTH_FACTOR)
                      distance = np.mean(crop_img*DEPTH_FACTOR)
                      print('mean_distance: ', distance)
                      
                      
                      toShow_C = cv2.addWeighted(overlay, alpha, toShow_C, 1 - alpha,0, 0)
                      area = (intersection.x2-intersection.x1)*(intersection.y2-intersection.y1)
                      print('area: ', area)
                      if distance > 0.65:
                        cv2.rectangle(overlay, intersection.upper_left(), intersection.bottom_right(), (0, 100, 100), -1)
                        # setup text
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        text = "STOP"
                        # get boundary of this text
                        textsize = cv2.getTextSize(text, font, 1, 2)[0]
                        # get coords based on boundary
                        textX = int((overlay.shape[1] - textsize[0]) / 2)
                        textY = int((overlay.shape[0] + textsize[1]) / 2)
                        # add text centered on image
                        cv2.putText(toShow_C, text, (textX, textY ), font, 1, (0, 0, 255), 2)
                      if (distance > 0.5 and distance < 0.6) or (area > 7000):
                        cv2.rectangle(overlay, intersection.upper_left(), intersection.bottom_right(), (0, 150, 255), -1)
                                                # setup text
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        text = "warning"
                        # get boundary of this text
                        textsize = cv2.getTextSize(text, font, 1, 2)[0]
                        # get coords based on boundary
                        textX = int((overlay.shape[1] - textsize[0]) / 2)
                        textY = int((overlay.shape[0] + textsize[1]) / 2)
                        # add text centered on image
                        cv2.putText(toShow_C, text, (textX, textY ), font, 1, (200, 200, 255), 2)

                      # cv2.rectangle(overlay, intersection.upper_left(), intersection.bottom_right(), (0, 50, 50), -1)
                      print('..................................intersection')

                  draw_bounding_box(toShow_C, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))
                  # toShow_C = cv2.resize(toShow_C, (width*2, height*2))
              # rgb_show_img = cv2.resize(rgb_show_img, (width, height))
              # cv2.imshow("object detection", rgb_show_img)
              # cv2.waitKey(1)
              # continue
          if args.mode == 0 :
              toShow_C = img             
              toShow_C = cv2.rectangle(toShow_C,center_box.upper_left(),center_box.bottom_right(),(100,100,100),1)
              for i in indices:
                  i = i[0]
                  box = boxes[i]
                  x = box[0]
                  y = box[1]
                  w = box[2]
                  h = box[3]

                  rect = Rectangle(x,y,x+w,y+h)
                  intersection = center_box&rect
                  if intersection is not None:
                      print(intersection)
                      print(intersection.y1)
                      print('intersection..................................')
                      crop_img = toShow_C[int(intersection.y1):int(intersection.y2),
                                           int(intersection.x1):int(intersection.x2)].copy()
                      print('mean: ', np.mean(crop_img))
                      print('..................................intersection')
                      overlay = toShow_C.copy()
                      # output = image.copy()
                      
                      alpha = 0.5
                      cv2.rectangle(overlay, intersection.upper_left(), intersection.bottom_right(), (0, 200, 200), -1)
                      toShow_C = cv2.addWeighted(overlay, alpha, toShow_C, 1 - alpha,0, 0)
                  draw_bounding_box(toShow_C, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))
          if args.mode == 1:
              toShow_C = disp_color
              # draw rectangle
              # upper_left = (int(width * 0.3), int(height * 0.25))
              # bottom_right = (int(width * 0.7), int(height * 0.75))
              # toShow_C = cv2.rectangle(toShow_C,upper_left,bottom_right,(100,100,100),1)
          elif args.mode == 2:
              #output_image = draw_bbox(img, bbox, label, conf)
              #toShow_C = output_image
              for i in indices:
                  i = i[0]
                  box = boxes[i]
                  x = box[0]
                  y = box[1]
                  w = box[2]
                  h = box[3]
                  draw_bounding_box(img, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))
              # rgb_show_img = cv2.resize(rgb_show_img, (width, height))
              # cv2.imshow("object detection", rgb_show_img)
              # cv2.waitKey(1)
              # continue
              toShow_C = np.concatenate((img, disp_color), 1)

          cv2.imshow('3net', toShow_C)

          k = cv2.waitKey(1)
          print()
          print(k)
          print()
          if k in [1048685, 109]: # 'm' to change mode (shifting 0->1->2->3->0)
            args.mode = (args.mode+1) % 4
          if k in [1048603, 27]: # esc to quit
            break
          if k in [1048688, 112]: # 'p' to pause
            cv2.waitKey(0)
          end = time.time()
          print("Time: " + str((end - start)))
        cam.release()

def warning():
      # 1.
    a = Rectangle(0, 0, 100, 100)
    b = Rectangle(50, 50, 150, 150)
    c = a&b
    print(c)
    # Rectangle(0.5, 0.5, 1, 1)
    print(list(a-b))
    # [Rectangle(0, 0, 0.5, 0.5), Rectangle(0, 0.5, 0.5, 1), Rectangle(0.5, 0, 1, 0.5)]

    # white blank image
    # blank_image2 = 255 * np.ones(shape=[512, 512, 3], dtype=np.uint8)
    cv2.rectangle(blank_image2, (a.x1 , a.y1), (a.x2 , a.y2), (0, 255, 0), 5)
    cv2.rectangle(blank_image2, (b.x1 , b.y1), (b.x2 , b.y2), (0, 255, 0), 5)
    cv2.rectangle(blank_image2, (c.x1 , c.y1), (c.x2 , c.y2), (255, 255, 255), 5)
    crop_img = blank_image2[c.y1:c.y2, c.x1:c.x2]
    print(np.mean(crop_img))

if __name__ == '__main__':
    tf.app.run()
