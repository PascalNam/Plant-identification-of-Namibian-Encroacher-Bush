# Training of the random forest classifier is done in this script.
 # Outputs: .hpy5 files containing the feature vector (stored under
#  output).'cleaned' text files, where '[',']' and ',' have been 
# removed (stored under Textfiles). The text files' lines have the 
# format: destination, number of instances found, x1 y1 x2 y2 \n
# There is one text file per class.
##################### Folder Structure #################################
#	Machine_learning - dataset - train 		- Bush
#			*		 -		*  -	*  		- Tree
#			*		 -		*  - Testingpics 
#			*		 -		*  - output
#			*		 -		*  - Textfiles
#			*		 - Definitions.py
#			*		 - Train.py
#			*		 - Test.py	
#	
#######################################################################

# Authors: P Marggraff and Dr MP Venter

# Organise imports:
import numpy as np
import mahotas
import cv2
import os
import h5py
import glob
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.externals import joblib
from PIL import Image
import pickle
import imutils
import time
from sklearn.model_selection import RandomizedSearchCV
from skimage.exposure import rescale_intensity
import skimage.data as data
import skimage.segmentation as seg
import skimage.filters as filters
import skimage.draw as draw
import skimage.color as color
from Definitions import definitions as defs

################# General Variables ####################################
# fixed-sizes for image
fixed_size = tuple((500, 500))

# path to training data
train_path = 'dataset/train'

# number of images per class
images_per_class = 4000


# no of trees for Random Forests
num_trees = 100

# bins for histogram
binss = [3,4,6,7,8]

# train_test_split size
test_size = 0.10

# seed for reproducing same results
seed = 9

# get the training labels
train_labels = os.listdir(train_path)

# sort the training labels
train_labels.sort()
print(train_labels)

# empty lists to hold feature vectors and labels and planes
global_features = []
labels = []

# Lower and upper bound for colour segmentation
low = (0, 0, 100)
up = (255, 255, 255)

#gamma-correction
gamma = 0.95

# kernel matrecies for filter 4;
convMatrix = np.ones((3,3),np.float32)/9

# loop over histogram bins
for bins in binss:
	print('I am using BIN-size: ', bins)
	#Loop over folders
	
	if bins == 8:		# Use a combination of filter 6 and 4
		for training_name in train_labels:
			dir = os.path.join(train_path, training_name)
			current_label = training_name
			
			
			#Loop over images
			for x in range(1,images_per_class+1):
				dir = os.path.join(train_path, training_name)
				file = dir + "/" + str(x) + ".jpg"
				image = np.array(Image.open(file))
				print(file)
				if current_label == 'Tree': # Use filter 6
					result_planes = []
					# Split image channels
					rgb_planes = cv2.split(image)
					# Loop over channels
					for plane in rgb_planes:
						# Remove noise
					    dilated_img = cv2.dilate(plane, np.ones((600,600), np.uint8))
					    bg_img = cv2.medianBlur(dilated_img, 21)
					    diff_img = 255 - cv2.absdiff(plane, bg_img)
					    result_planes.append(diff_img)
				
					result = cv2.merge(result_planes)
					hsv_image = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
					#Segment colour range
					mask = cv2.inRange(hsv_image,low,up)
					# Mask for positive images
					npMask = np.array(mask)
					col = cv2.cvtColor(npMask,cv2.COLOR_GRAY2RGB)
					cond = col<1
					if np.array(cond).shape != np.array(image).shape:
						x=image.transpose(1,0,2)
						f = np.flip(x,1)
						pixels=np.where(cond,f,col)
							
					else:
						pixels=np.where(cond,image,col)
					
					#Save fetures to label		
					imagenoneg = cv2.resize(pixels, fixed_size)
					fv_histogram  = defs.fd_histogram(imagenoneg,bins)
					global_feature = fv_histogram
					labels.append(current_label)
					global_features.append(global_feature)
					
					#Negative
					npMask = np.array(mask)
					col = cv2.cvtColor(npMask,cv2.COLOR_GRAY2RGB)
					cond = col>1						####--------> Negative
					if np.array(cond).shape != np.array(image).shape:
						x=image.transpose(1,0,2)
						f = np.flip(x,1)
						pixels = np.where(image,f,col)
					else:
						pixels=np.where(cond,image,col)	
					# Save features to label
					imagenegative = cv2.resize(pixels, fixed_size)
					fv_histogram  = defs.fd_histogram(imagenegative,bins)
					global_feature = fv_histogram
					labels.append('Negative')
					global_features.append(global_feature)	
					
				else:	# Else can only be Bush - use filter 4
					lookUpTable = np.empty((1,256), np.uint8)
					# Change contrast using gamma correction
					for i in range(256):
						lookUpTable[0,i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
					res = cv2.LUT(image, lookUpTable)
					convolve = cv2.filter2D(res,-1,convMatrix)
					mask = cv2.inRange(convolve,low,up)
					# Positive mask
					npMask = np.array(mask)
					col = cv2.cvtColor(npMask,cv2.COLOR_GRAY2RGB)
					cond = col<1
					if np.array(cond).shape != np.array(image).shape:
						x=image.transpose(1,0,2)
						f = np.flip(x,1)
						pixels_of_bg=np.where(cond,f,col)
							
					else:
						pixels_of_bg=np.where(cond,image,col)	
					
					#Save fetures to label		
					imagenoneg = cv2.resize(pixels_of_bg, fixed_size)
					fv_histogram  = defs.fd_histogram(imagenoneg,bins)
					global_feature = fv_histogram
					labels.append(current_label)
					global_features.append(global_feature)
					#Negative
					npMask = np.array(mask)
					col = cv2.cvtColor(npMask,cv2.COLOR_GRAY2RGB)
					cond = col>1							####--------> Negative
					if np.array(cond).shape != np.array(image).shape:
						x=image.transpose(1,0,2)
						f = np.flip(x,1)
						pixels = np.where(image,f,col)
					else:
						pixels=np.where(cond,image,col)	
					#Save fetures to label
					imagenegative = cv2.resize(pixels, fixed_size)
					fv_histogram  = defs.fd_histogram(imagenegative,bins)
					global_feature = fv_histogram
					labels.append('Negative')
					global_features.append(global_feature)	
	else:		# All other bins are done on filter 6.
		for training_name in train_labels:
			dir = os.path.join(train_path, training_name)
			current_label = training_name
			
			
			#Loop over images
			for x in range(1,images_per_class+1):
				dir = os.path.join(train_path, training_name)
				file = dir + "/" + str(x) + ".jpg"
				image = np.array(Image.open(file))
				print(file)
				result_planes = []
				# Split image channels
				rgb_planes = cv2.split(image)
				# Loop over channels
				for plane in rgb_planes:
					# Remove noise
				    dilated_img = cv2.dilate(plane, np.ones((600,600), np.uint8))
				    bg_img = cv2.medianBlur(dilated_img, 21)
				    diff_img = 255 - cv2.absdiff(plane, bg_img)
				    result_planes.append(diff_img)
			
				result = cv2.merge(result_planes)
				hsv_image = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
				# Segmentation of specified colour range
				mask = cv2.inRange(hsv_image,low,up)
				
				# Positive mask
				npMask = np.array(mask)
				col = cv2.cvtColor(npMask,cv2.COLOR_GRAY2RGB)
				cond = col<1
				if np.array(cond).shape != np.array(image).shape:
					x=image.transpose(1,0,2)
					f = np.flip(x,1)
					pixels=np.where(cond,f,col)
						
				else:
					pixels=np.where(cond,image,col)
				
				#Save fetures to label		
				imagenoneg = cv2.resize(pixels, fixed_size)
				fv_histogram  = defs.fd_histogram(imagenoneg,bins)
				global_feature = fv_histogram
				labels.append(current_label)
				global_features.append(global_feature)
				
				#Negative
				npMask = np.array(mask)
				col = cv2.cvtColor(npMask,cv2.COLOR_GRAY2RGB)
				cond = col>1						####--------> Negative
				if np.array(cond).shape != np.array(image).shape:
					x=image.transpose(1,0,2)
					f = np.flip(x,1)
					pixels = np.where(image,f,col)
				else:
					pixels=np.where(cond,image,col)	
				#Save fetures to label
				imagenegative = cv2.resize(pixels, fixed_size)
				fv_histogram  = defs.fd_histogram(imagenegative,bins)
				global_feature = fv_histogram
				labels.append('Negative')
				global_features.append(global_feature)		
	
	
		print ('[STATUS] processed folder: {}'.format(current_label))

	print('[STATUS] Finished training')	
	print ("[FORMAT] Feature vector size {}".format(np.array(global_features).shape))
	# encode the target labels
	targetNames = np.unique(labels)
	le = LabelEncoder()
	target = le.fit_transform(labels)
	# normalize the feature vector in the range (0-1)
	scaler = MinMaxScaler(feature_range=(0, 1))
	rescaled_features = scaler.fit_transform(global_features)
	
	#Save rescaled features and encoded labels as h5py files
	dir = os.path.join('output',str(bins))
	file = dir +'_feature.h5'
	h5f_data = h5py.File(file, 'w')
	h5f_data.create_dataset('dataset_1', data=np.array(rescaled_features))
	
	file = dir +'_label.h5'
	h5f_label = h5py.File(file, 'w')
	h5f_label.create_dataset('dataset_1', data=np.array(target))
	
	h5f_data.close()
	h5f_label.close()
	global_features.clear()
	labels.clear()
	print ("[FORMAT] Feature vector size after clearing {}".format(np.array(global_features).shape))



