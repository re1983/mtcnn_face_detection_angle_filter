# coding: utf-8
import mxnet as mx
from mtcnn_detector import MtcnnDetector
import cv2
import cv
import os
import time
import random
import shutil
import sys 
import pylab
import imageio

videopath = '/home/tony/video_center/'
picturepath = '/home/tony/picture_center/'
chipspath = '/home/tony/chip_center/'

detector = MtcnnDetector(model_folder='model',
                        ctx=mx.gpu(0),
                        num_worker = 8 , 
                        accurate_landmark = False,
                        #accurate_landmark = True,
                        minsize = 128,
                        threshold = [0.6, 0.7, 0.8],
                        factor = 0.709,)
    
def facedetector(dp,imgpath,image):

    global num    
    try:
        img = image
        #height, width, channels = img.shape
        #img = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
        imgpath = imgpath.replace('picture','face')
        if not os.path.exists(imgpath):
            results = detector.detect_face(img)

            if results is not None:            
                total_boxes = results[0]
                points = results[1]
                imgpath = imgpath.replace('.jpg','_noalign.jpg')
                '''
                #save img                
                if not os.path.exists(os.path.dirname(imgpath)):
                    os.makedirs(os.path.dirname(imgpath))
                cv2.imwrite(imgpath,img)
                
                #save_draw box & LM
                draw = img.copy()
                for b in total_boxes:
                    cv2.rectangle(draw, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (255, 255, 255))
                for p in points:
                    for i in range(5):
                        cv2.circle(draw, (p[i], p[i + 5]), 1, (0, 0, 255), 2)
                        #cv2.putText(draw, str(i), (p[i], p[i + 5]), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 0), 1,False)
                        #cv2.putText(draw, str(int(p[i])) + ',' + str(int(p[i + 5])), (p[i], p[i + 5]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1,False)
                imgpath = imgpath.replace('.jpg','')
                cv2.imwrite(imgpath+'.png', draw)
                '''
                #extract aligned face chips
                #chips = detector.extract_image_chips(img, points, 128, 0.37) #original function
                chips = detector.extract_image_chips_angle_limit(img, points, 128, 0.37, 0.1)  #new function for mtcnn by JJ
                chipspath = imgpath.replace('face','chip')
                for i, chips in enumerate(chips):                 
                    if not os.path.exists(os.path.dirname(chipspath)):
                        os.makedirs(os.path.dirname(chipspath))
                    cv2.imwrite(chipspath + '_' +str(i) + '.jpg',chips)
                    #break            
            #else:
                #print('no results:  ' + imgpath)
    except:
        e = sys.exc_info()[0]
        print(e)
    

if __name__ == "__main__":   
    
    start = time.time()    
    for (dp,dn,fnt) in os.walk(videopath):
        for vidname in fnt:          
            vidfile_abs = os.path.join(dp, vidname)
            vidcap = cv2.VideoCapture(vidfile_abs)
            totalFrames = int(vidcap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)) #
            print "***totalFrames = %d***" % (totalFrames)
            myFrameNumber = 3
            success,image = vidcap.read()
            #print(success)
            count =0
            try:
                while success:
                    imagepath_save = dp.replace('video','picture') + '/' + vidname.replace('.avi','') + '_' + str(count).zfill(len(str(totalFrames))+1) + '.jpg'
                    vidcap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,count)
                    success,image = vidcap.read()
                    facedetector(dp,imagepath_save,image)
                    #cv2.imwrite(imagepath_save, image)     # save frame as JPEG file at '/home/tony/picture_center/'
                    count += myFrameNumber
            except:
                e = sys.exc_info()[0]
                print(e)  
    print('down')
    end = time.time()
    elapsed = end - start
    print "Time taken: ", elapsed, "seconds."


