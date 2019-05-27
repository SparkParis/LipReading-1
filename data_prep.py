import cv2
import os
import argparse
from os.path import isdir 
from os import listdir
from os.path import isfile, join
from skimage.transform import resize   # for resizing images
import dlib # run "pip install dlib"
from imutils import face_utils
#This program extracts face data from the images and saves it in images size of 224,224,3
#this size was chosen as the VGG model needs 224 x 224 x 3 images to process 
RECTANGLE_LENGTH = 90

def get_faces( filename, face_cascade, mouth_cascade,detector,predictor):
	image = cv2.imread(filename)
	#image = imutils.resize(image, width=500)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	face_images = []
	# detect faces in the grayscale image
	rects = detector(gray, 1)
	if len(rects) > 1:
		print( "ERROR: more than one face detected")
		return
	if len(rects) < 1:
		print( "ERROR: no faces detected")
		return 

	rect = rects[0]
	shape = predictor(gray, rect)
	shape = face_utils.shape_to_np(shape)
	(x, y, w, h) = face_utils.rect_to_bb(rect)
	w = RECTANGLE_LENGTH
	h = RECTANGLE_LENGTH
	cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

	(x_r, y_r, w_r, h_r) = (x, y, w, h)
 
	cv2.putText(image, "Face #{}".format(0 + 1), (x - 10, y - 10),
		cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
 
	crop_img = image[y_r:y_r + h_r, x_r:x_r + w_r]
	face_images.append(crop_img)
	return face_images

# Walk into directories in filesystem
# Ripped from os module and slightly modified
# for alphabetical sorting
#
def sortedWalk(top, topdown=True, onerror=None):
    from os.path import join, isdir, islink

    names = os.listdir(top)
    names.sort()
    dirs, nondirs = [], []

    for name in names:
        if isdir(os.path.join(top, name)):
            dirs.append(name)
        else:
            nondirs.append(name)

    if topdown:
        yield top, dirs, nondirs
    for name in dirs:
        path = join(top, name)
        if not os.path.islink(path):
            for x in sortedWalk(path, topdown, onerror):
                yield x
    if not topdown:
        yield top, dirs, nondirs

def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]

def scan_images(root_dir):
    #image_extensions = ["jpg", "png"]
    image_extensions = ["jpg"]
    num_faces = 0
    file_num = 0
    num_images = 0
    current_dir = ""
    
    face_cascade = cv2.CascadeClassifier('../opencv/haarcascade_frontalface_default.xml')
    mouth_cascade = cv2.CascadeClassifier('../opencv/haarcascade_mouth.xml')
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    dir_created = 0
    for root, dirs, files in sortedWalk(root_dir):
        for dir in dirs:
            current_dir = os.path.join(root, dir)
            print("Current directory" + current_dir)
            dir_check = current_dir
            dir_check[-6:]
            if dir_check is "output":
                print("This is already an output directory" % current_dir)
                break
            dir_created = 0
            file_num = 0
            output_dir =  current_dir + "/output"
            
            file_list = [f for f in listdir(current_dir) if isfile(join(current_dir, f))]
            for filename in file_list:
                extension = os.path.splitext(filename)[1][1:]
                if extension in image_extensions:

                    faces = get_faces(os.path.join(current_dir, filename), face_cascade, mouth_cascade,detector,predictor)
                    if faces is not None:
                        if len(faces) is 1:
                            num_images += 1
                            if dir_created is 0:
                                try:
                                    os.mkdir(output_dir)
                                except OSError:
                                    print("Creation of the directory %s failed" % output_dir)
                                    break
                                else:
                                    print("Successfully created the directory %s " % output_dir)
                                    dir_created = 1

                            file_num = int(filename[6:-4])
                            for face in faces:
                                face_filename = os.path.join(output_dir, "face_{:03d}.png".format(file_num))
                                print(face.shape)
                                face = cv2.resize(face, (90, 90))
                                print(face.shape)
                                cv2.imwrite(face_filename, face)
                                print("\tWrote {} extracted from {}".format(face_filename, filename))
                                num_faces += 1 
								#file_num+=1
                    else:
                        print("no face written on file!")


    print("-" * 20)
    print("Total number of images: {}".format(num_images))
    print("Total number of faces: {}".format(num_faces))          


#scan_images("F01")
#scan_images("F02")
#scan_images("F04")
#scan_images("F05")
#scan_images("F06")
scan_images("F07")
scan_images("F08")
scan_images("F09")
scan_images("F10")
scan_images("F11")
scan_images("M01")
scan_images("M02")
scan_images("M04")
scan_images("M07")
scan_images("M08")
