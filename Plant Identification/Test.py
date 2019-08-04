# Testing of the features found in the images. Use feature vectors and
# labels to predict the class within a window. At first, filter 5 is 
# applied to the image and a sliding window predicts the class. If filter
# 5 filters out more than half of the pixels, then filter 6 is applied 
# on the remaining pixels. After this process, a final decision is made,
# where each feature vector can decide on the outcome. For this all 
# cleaned text files with window coordinates is opened and according to
# the number of instances found by each vector, the final outcome is 
# decided upon.
# Outputs: Text files and 'cleaned' text files, where '[',']' and ',' 
# have been removed (stored under Textfiles). The text files' lines have  
# the format: destination, number of instances found, x1 y1 x2 y2 \n
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
import matplotlib.image as mpimg
import itertools
from Definitions import definitions as defs



# General Variables:
# Parameters used in RandomizedsearchCV for tuning of RF classifier
param_dist = {"n_estimators":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,23,40,50,60,70,80,90,100,120,130,140,150,160,180,190,200,225,250,275,300,310,320,330],
						"min_samples_split": [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,23,40,50,60,70,80,90,100,120,130,140,150,160,180,190,200,225,250,275,300,310,320,330],
						"min_samples_leaf":[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,23,40,50,60,70,80,90,100,120,130,140,150,160,180,190,200,225,250,275,300,310,320,330],
						"max_depth": [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,23,40,50,60,70,80,90,100,120,130,140,150,160,180,190,200,225,250,275,300,310,320,330],
						"bootstrap": [True, False],
						"criterion": ["gini", "entropy"]}	
						
# Window width and height						
(winW, winH) = (50,50)						

# Classifier
clf  = RandomForestClassifier()

# Images in training folder
images_in_testingfile = 201	

# Empty lists for coordinates and counts										
bushcoords = []
treecoords = []
bushcountarr = []
treecountarr = []
bushcount = []
treecount = []

# bins for histogram
binss = [2,3,4,6,8]

# Dilation kernel:
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))

# Filter elements: Convolution matrix, HSV bandwidth
convMatrix = np.ones((3,3),np.float32)/9
low = (0, 0, 100)
up = (255, 255, 255)
# gamma-correction
gamma = 0.95

# Colour for final decision, where red is woody mass and green is leafy 
# things.
RED = (255,0,0)
GREEN = (0,255,0)
# Loop over bins
for bins in binss:
	# Open features and labels
	dir = os.path.join('output/test',str(bins))
	file = dir +'_feature.h5'
	print('Using features stored in: ',file)
	h5f_data = h5py.File(file, 'r')
	global_features_string = h5f_data['dataset_1']
	global_features = np.array(global_features_string)
	h5f_data.close()
	
	file = dir +'_label.h5'
	h5f_label = h5py.File(file, 'r')
	global_labels_string = h5f_label['dataset_1']
	global_labels = np.array(global_labels_string)
	h5f_label.close()
	# Hyperparameter tuning with RandomizedSearchCV on RF
	random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
							n_iter=10,cv = 5, n_jobs = -1, iid=False)
	# Fit classifier and report results
	random_search.fit(global_features, global_labels)
	defs.report(random_search.cv_results_)
	# Loop over test images
	for p in range(1,images_in_testingfile+1):
	
		dir = 'dataset/Testingpics'
		file = dir + '/' +str(p) + '.jpg'
		dest = file
		
		image = np.array(Image.open(file))
		# Images have to be resized for SLIC
		resized = cv2.resize(image, ((1000, 1000)))	
		cpy = resized.copy()
		# Segmentation using SLIC
		image_slic = seg.slic(resized,n_segments=5, compactness=2,max_iter=11, convert2lab=True)
		slicmask = color.label2rgb(image_slic, colors = ['black','white','black','black'], kind='overlay')
		# Make a mask suitable for OpenCV
		threshd = color.rgb2gray(slicmask)
		threshd =  rescale_intensity(threshd, out_range=(0, 255)).astype("uint8")
		# Mask image
		npMask = threshd
		col = cv2.cvtColor(npMask,cv2.COLOR_GRAY2RGB)
		cond = col>1
		if np.array(cond).shape != np.array(resized).shape:
			x=image.transpose(1,0,2)
			print(x.shape)
			f = np.flip(x,1)
			print(f.shape)
			pixels=np.where(cond,f,col)
		else:
			pixels=np.where(cond,resized,col)
			pixelsbg = np.where(cond,col,resized)
			(thresh, inv) = cv2.threshold(pixelsbg,0,255,cv2.THRESH_BINARY)
			pixelsbg = np.where(cond,inv,resized)
		print('[STATUS] Searching in:',file)
		resized = pixels
		# Class counts - used in text files and final decision
		bushcount = 0
		treecount = 0
		# Sliding window that identifies class in the image
		for (x, y, window)in defs.slidingwindow(resized, stepSize=32,windowSize=(winW,winH)):
				if window.shape[0] != winH or window.shape[1] != winW:
					continue
				else:
					# If all pixels in the window are black, it should 
					# simply sjip these
					if np.all(window == 0):
						continue
					else:
						#Do feature extraction
						
						fv_histogram  = defs.fd_histogram(window,bins)
						global_feature = fv_histogram
						
						# Predict class
						prediction = random_search.predict(global_feature.reshape(1,-1))[0]
						clone = resized.copy()
						cv2.rectangle(clone, (x,y), (x+winW, y+winH), (0,225,0),2)
						cv2.namedWindow("Window",cv2.WINDOW_NORMAL)
						cv2.imshow('Window',clone)
				
						cv2.waitKey(1)
						time.sleep(0.025)
						# Save coordinates (x1,y1,x2,y2)
						if prediction == 0:
							x1 = x
							bushcoords.append(x1)
							y1 = y
							bushcoords.append(y1)
							x2 = x+winW
							bushcoords.append(x2)
							y2 = y+winW
							bushcoords.append(y2)
							
							bushcount += 1
							
						elif prediction == 1:
							x1 = x
							treecoords.append(x1)
							y1 = y
							treecoords.append(y1)
							x2 = x+winW
							treecoords.append(x2)
							y2 = y+winW
							treecoords.append(y2)
							
							treecount += 1
		# Sometimes SLIC leaves out important information that we want 
		# capture. Therefore, search in the remaining image. For this,
		# filter 6 is applied dirst to the image.
		if cv2.countNonZero(npMask) > 505000:
			result_planes = []
			# split channels
			rgb_planes = cv2.split(pixelsbg)
			for plane in rgb_planes:
				# remove noise (shadows)
			    dilated_img = cv2.dilate(plane, np.ones((500,500), np.uint8))
			    bg_img = cv2.medianBlur(dilated_img, 21)
			    diff_img = 255 - cv2.absdiff(plane, bg_img)
			    result_planes.append(diff_img)
		
			result = cv2.merge(result_planes)
			hsv_image = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
			# segment colour range
			mask = cv2.inRange(hsv_image,low,up)
			
			npMask = np.array(mask)
			col = cv2.cvtColor(npMask,cv2.COLOR_GRAY2RGB)
			cond = col<1
			if np.array(cond).shape != np.array(resized).shape:
				x=image.transpose(1,0,2)
				f = np.flip(x,1)
				pixels_of_bg=np.where(cond,f,col)
					
			else:
				pixels_of_bg=np.where(cond,cpy,col)	
				
			resized2 = pixels_of_bg
			
		
			# Use sliding window for remainder of image and predict class				
			for (x, y, window)in defs.slidingwindow(resized2, stepSize=32,windowSize=(winW,winH)):
					if window.shape[0] != winH or window.shape[1] != winW:
						continue
					else:
						
						if np.all(window == 255):
							continue
						else:
							
							fv_histogram  = defs.fd_histogram(window,bins)
							global_feature = fv_histogram
							
							
							prediction = random_search.predict(global_feature.reshape(1,-1))[0]
							clone = resized2.copy()
							cv2.rectangle(clone, (x,y), (x+winW, y+winH), (0,225,225),2)
							cv2.namedWindow("Window",cv2.WINDOW_NORMAL)
							cv2.imshow('Window',resized2)
					
							cv2.waitKey(1)
							time.sleep(0.025)
							if prediction == 0:
								x1 = x
								bushcoords.append(x1)
								y1 = y
								bushcoords.append(y1)
								x2 = x+winW
								bushcoords.append(x2)
								y2 = y+winW
								bushcoords.append(y2)
								bushcount += 1
								
								
								
							elif prediction == 1:
								x1 = x
								treecoords.append(x1)
								y1 = y
								treecoords.append(y1)
								x2 = x+winW
								treecoords.append(x2)
								y2 = y+winW
								treecoords.append(y2)
								treecount += 1
								
			# save destination, counts and coordinates to .txt files
			print('[STATUS] Writing to textfile')
			print(dest)
			if bushcount != 0:
				print('I found Bush')
				dir = os.path.join('textfiles',str(bins))
				file = dir + '_bush.txt'
				file1 = open(file,'a')
				print('writig')
				file1.write(dest)
				file1.write(' ')
				file1.write(str(bushcount))
				file1.write(' ')
				file1.write(str(bushcoords))
				file1.write('\n')
				print('wrote')
				file1.close
				print('writig_close')
				
			elif bushcount==0:
				print('I did not find bush')
				dir = os.path.join('textfiles',str(bins))
				file = dir + '_bush.txt'
				file1 = open(file,'a')
				file1.write(dest)
				file1.write(' ')
				file1.write(str(0))
				file1.write('\n')
				file1.close	
				
			if treecount !=0:
				print('I found tree')
				dir = os.path.join('textfiles',str(bins))
				file = dir + '_tree.txt'
				file2 = open(file,'a')
				file2.write(dest)
				file2.write(' ')
				file2.write(str(treecount))
				file2.write(' ')
				file2.write(str(treecoords))
				file2.write('\n')
				file2.close
			
			
			elif treecount==0:
				print(dest)
				print('I did not find tree')
				dir = os.path.join('textfiles',str(bins))
				file = dir + '_tree.txt'
				file2 = open(file,'a')
				file2.write(dest)
				file2.write(' ')
				file2.write(str(0))
				file2.write('\n')
				file2.close
			bushcoords = []
			treecoords = []
			bushcountarr = []
			treecountarr = []
			bushcount = []
			treecount = []
		else:
			
			print('Else loop', dest)
			print('[STATUS] Writing to textfile')
	
			if bushcount != 0:
				print('I found Bush')
				dir = os.path.join('textfiles',str(bins))
				file = dir + '_bush.txt'
				file1 = open(file,'a')
				print('writing')
				file1.write(dest)
				file1.write(' ')
				file1.write(str(bushcount))
				file1.write(' ')
				file1.write(str(bushcoords))
				file1.write('\n')
				print('wrote')
				file1.close
				print('wrote_close')
				
				
			elif bushcount==0:
				print('I did not find bush')
				dir = os.path.join('textfiles',str(bins))
				file = dir + '_bush.txt'
				file1 = open(file,'a')
				file1.write(dest)
				file1.write(' ')
				file1.write(str(0))
				file1.write('\n')
				file1.close
					
				
			if treecount !=0:
				print('I found tree')
				print('Else loop tree', dest)
				dir = os.path.join('textfiles',str(bins))
				file = dir + '_tree.txt'
				file2 = open(file,'a')
				file2.write(dest)
				file2.write(' ')
				file2.write(str(treecount))
				file2.write(' ')
				file2.write(str(treecoords))
				file2.write('\n')
				file2.close
				
			
			
			elif treecount==0:
				print('else no tree',dest)
				print('I did not find tree')
				dir = os.path.join('textfiles',str(bins))
				file = dir + '_tree.txt'
				file2 = open(file,'a')
				file2.write(dest)
				file2.write(' ')
				file2.write(str(0))
				file2.write('\n')
				file2.close
				
			bushcoords = []
			treecoords = []

# Clean all ',' '[' and ']' from .txt files				
for H in binss:
	
	dir = os.path.join('textfiles',str(H))
	file = dir +'_bush.txt'
	dest_out_bush = dir + '_clean_bush.txt'
	defs.cleaner(file,dest_out_bush)			

for G in binss:	
	dir = os.path.join('textfiles',str(G))
	file = dir +'_tree.txt'
	dest_out_tree = dir + '_clean_tree.txt'
	defs.cleaner(file,dest_out_tree)
# Final decision. Each filter has its preferred images, where they work 
# Exeptionally well on. Here, each filter thus gets a vote on the region,
# where the classes have been idebtified. SLIC then segments these
# regions. The SLIC region is then used as a mask, where the filters
# count the number of instances that they have found within this mask.
# The highest count wins and each filter votes for the final outcome.

# Open all text files
with open('textfiles/2_clean_bush.txt') as bushtxt1, open('textfiles/2_clean_tree.txt') as treetxt1, open('textfiles/3_clean_bush.txt') as bushtxt2, open('textfiles/3_clean_tree.txt') as treetxt2, open('textfiles/4_clean_bush.txt') as bushtxt3, open('textfiles/4_clean_tree.txt') as treetxt3, open('textfiles/6_clean_bush.txt') as bushtxt4, open('textfiles/6_clean_tree.txt') as treetxt4, open('textfiles/8_clean_bush.txt') as bushtxt5, open('textfiles/8_clean_tree.txt') as treetxt5:
	# Strip the lines of each textfile
	for b1,c1,b2,c2,b3,c3,b4,c4,b5,c5, in itertools.zip_longest(bushtxt1,treetxt1,bushtxt2,treetxt2,bushtxt3,treetxt3,bushtxt4,treetxt4,bushtxt5,treetxt5):
		
		b1 = b1.strip()
		c1 = c1.strip()
		b2 = b2.strip()
		c2 = c2.strip()
		b3 = b3.strip()
		c3 = c3.strip()
		b4 = b4.strip()
		c4 = c4.strip()
		b5 = b5.strip()
		c5 = c5.strip()
		# Get destination, coordinates and instance counts from the files
		frame_path_b1, pts_b1, label_count_b1 = defs.parse_line(b1)
		frame_path_c1, pts_c1, label_count_c1 = defs.parse_line(c1)
		frame_path_b2, pts_b2, label_count_b2 = defs.parse_line(b2)
		frame_path_c2, pts_c2, label_count_c2 = defs.parse_line(c2)
		frame_path_b3, pts_b3, label_count_b3 = defs.parse_line(b3)
		frame_path_c3, pts_c3, label_count_c3 = defs.parse_line(c3)
		frame_path_b4, pts_b4, label_count_b4 = defs.parse_line(b4)
		frame_path_c4, pts_c4, label_count_c4 = defs.parse_line(c4)
		frame_path_b5, pts_b5, label_count_b5 = defs.parse_line(b5)
		frame_path_c5, pts_c5, label_count_c5 = defs.parse_line(c5)		
		
		print(frame_path_c1)
		dir = os.path.join('pics',str(frame_path_c1))
		file = dir + '.jpg'
        # #load the image
		img1 = cv2.imread(frame_path_b1)
		img1 = cv2.resize(img1,((1000,1000)))
		# Make blank images, where the wound windows can be printed to.
		blank_image1 = 255 * np.ones(shape=[1000, 1000,3], dtype=np.uint8)
		blank_image2 = 255 * np.ones(shape=[1000, 1000,3], dtype=np.uint8)
		blank_image3 = 255 * np.ones(shape=[1000, 1000,3], dtype=np.uint8)
		blank_image4 = 255 * np.ones(shape=[1000, 1000,3], dtype=np.uint8)
		blank_image5 = 255 * np.ones(shape=[1000, 1000,3], dtype=np.uint8)
		notuse = 255 * np.ones(shape=[1000, 1000,3], dtype=np.uint8)
		notuse2 = 255 * np.ones(shape=[1000, 1000,3], dtype=np.uint8)
		# Print the found windows to empty images
		if label_count_b1 != 0 and label_count_c1 !=0:
			
			for bb1 in range(label_count_b1):
				label_pts_bb1 = pts_b1[bb1*4:bb1*4+4]
				rb1=cv2.rectangle(blank_image1,(label_pts_bb1[0],label_pts_bb1[1]),(label_pts_bb1[2],label_pts_bb1[3]),(0,255,0),cv2.FILLED)
	
			for cc1 in range(label_count_c1):
				label_pts_cc1 = pts_c1[cc1*4:cc1*4+4]
				rc1=cv2.rectangle(blank_image1,(label_pts_cc1[0],label_pts_cc1[1]),(label_pts_cc1[2],label_pts_cc1[3]),(255,0,0),cv2.FILLED)
			
			for bb2 in range(label_count_b2):
				label_pts_bb2 = pts_b2[bb2*4:bb2*4+4]
				rb2=cv2.rectangle(blank_image2,(label_pts_bb2[0],label_pts_bb2[1]),(label_pts_bb2[2],label_pts_bb2[3]),(0,255,0),cv2.FILLED)
	
			for cc2 in range(label_count_c2):
				label_pts_cc2 = pts_c2[cc2*4:cc2*4+4]
				rc2=cv2.rectangle(blank_image2,(label_pts_cc2[0],label_pts_cc2[1]),(label_pts_cc2[2],label_pts_cc2[3]),(255,0,0),cv2.FILLED)
			
			for bb3 in range(label_count_b3):
				label_pts_bb3 = pts_b3[bb3*4:bb3*4+4]
				rb3=cv2.rectangle(blank_image3,(label_pts_bb2[0],label_pts_bb2[1]),(label_pts_bb2[2],label_pts_bb2[3]),(0,255,0),cv2.FILLED)
	
			for cc3 in range(label_count_c3):
				label_pts_cc3 = pts_c3[cc3*4:cc3*4+4]
				rc3=cv2.rectangle(blank_image3,(label_pts_cc3[0],label_pts_cc3[1]),(label_pts_cc3[2],label_pts_cc3[3]),(255,0,0),cv2.FILLED)

			for bb4 in range(label_count_b4):
				label_pts_bb4 = pts_b4[bb4*4:bb4*4+4]
				rb4=cv2.rectangle(blank_image4,(label_pts_bb4[0],label_pts_bb4[1]),(label_pts_bb4[2],label_pts_bb4[3]),(0,255,0),cv2.FILLED)
	
			for cc4 in range(label_count_c4):
				label_pts_cc4 = pts_c4[cc4*4:cc4*4+4]
				rc4=cv2.rectangle(blank_image4,(label_pts_cc4[0],label_pts_cc4[1]),(label_pts_cc4[2],label_pts_cc4[3]),(255,0,0),cv2.FILLED)

			for bb5 in range(label_count_b5):
				label_pts_bb5 = pts_b5[bb5*4:bb5*4+4]
				rb5=cv2.rectangle(blank_image5,(label_pts_bb5[0],label_pts_bb5[1]),(label_pts_bb5[2],label_pts_bb5[3]),(0,255,0),cv2.FILLED)
	
			for cc5 in range(label_count_c5):
				label_pts_cc5 = pts_c5[cc5*4:cc5*4+4]
				rc5=cv2.rectangle(blank_image5,(label_pts_cc5[0],label_pts_cc5[1]),(label_pts_cc5[2],label_pts_cc5[3]),(255,0,0),cv2.FILLED)
			
			# REGION VOTE: Add all filters' printed images together, so
			# that they decide on the final regions, where instances are
			ad = cv2.add(blank_image1,blank_image2,notuse)
			ad2 = cv2.add(ad,blank_image3)
			ad3 = cv2.add(ad2,blank_image4)
			ad4 = cv2.add(ad3,blank_image5)
			plt.imshow(ad4)
			plt.show()
			# Get a SLIC mask of these regions
			add_slic_raw = seg.slic(ad4,n_segments=5, compactness=2,max_iter=11, convert2lab=True, enforce_connectivity = True)
			# Instance counts (Green = leafy things, Red = woody mass)
			greencount = 0
			redcount = 0
			greencounta = 0
			redcounta = 0
			greencountb = 0
			redcountb = 0
			add_slic =  rescale_intensity(add_slic_raw, out_range=(0, 255)).astype("uint8")
			npMask = np.array(add_slic)
			col = cv2.cvtColor(npMask,cv2.COLOR_GRAY2RGB)
			# Due to the choice of lettimg SLIC decide on maximum 5 
			# superpixels, there needs to be provision made for
			# instances where there are only 2, 3, 4 superpixels.
			# This is done with the np.any statements, where the grey - 
			# scale value of these pixels have been identified beforehand.
			if np.any(add_slic == 127):
				print('three regions')
				cond = col==127
				##### Filter1
				pixels1=np.where(cond,blank_image1,col)
				pixels1 = cv2.cvtColor(pixels1,cv2.COLOR_BGR2GRAY)
				hist1 = cv2.calcHist([pixels1],[0],None,[256],[0,256])
				if hist1[29]>hist1[150]:
					print('I vote RED')
					redcount+=1
				elif hist1[150]>hist1[150]:
					print('I vote GREEN')
					greencount+=1
				else:
					print('Conservation Mode: I vote RED')
					redcount+=1
					
					
				##### Filter2
				pixels2=np.where(cond,blank_image2,col)
				pixels2 = cv2.cvtColor(pixels2,cv2.COLOR_BGR2GRAY)
				hist2 = cv2.calcHist([pixels2],[0],None,[256],[0,256])
				if hist2[29]>hist2[150]:
					print('I vote RED')
					redcount+=1
				elif hist2[150]>hist2[150]:
					print('I vote GREEN')
					greencount+=1
				else:
					print('Conservation Mode: I vote RED')
					redcount+=1
					
				##### Filter3
				pixels3=np.where(cond,blank_image3,col)
				pixels3 = cv2.cvtColor(pixels3,cv2.COLOR_BGR2GRAY)
				hist3 = cv2.calcHist([pixels3],[0],None,[256],[0,256])
				if hist3[29]>hist3[150]:
					print('I vote RED')
					redcount+=1
				elif hist3[150]>hist3[150]:
					print('I vote GREEN')
					greencount+=1
				else:
					print('Conservation Mode: I vote RED')
					redcount+=1	
				
				##### Filter4
				pixels4=np.where(cond,blank_image4,col)
				pixels4 = cv2.cvtColor(pixels4,cv2.COLOR_BGR2GRAY)
				hist4 = cv2.calcHist([pixels4],[0],None,[256],[0,256])
				if hist4[29]>hist4[150]:
					print('I vote RED')
					redcount+=1
				elif hist4[150]>hist4[150]:
					print('I vote GREEN')
					greencount+=1
				else:
					print('Conservation Mode: I vote RED')
					redcount+=1		
					
				##### Filter5
				pixels5=np.where(cond,blank_image5,col)
				pixels5 = cv2.cvtColor(pixels5,cv2.COLOR_BGR2GRAY)
				hist5 = cv2.calcHist([pixels5],[0],None,[256],[0,256])
				if hist5[29]>hist5[150]:
					print('I vote RED')
					redcount+=1
				elif hist5[150]>hist5[150]:
					print('I vote GREEN')
					greencount+=1
				else:
					print('Conservation Mode: I vote RED')
					redcount+=1		
				
				
					
				if redcount> greencount:
					print("Making notuse red")
					notuse2[add_slic== 127] = RED
				else:
					notuse2[add_slic == 127] = GREEN
					print("Making notuse GREEN")

				
				
			elif np.any(add_slic == 85):
				print('Four Regions')
				greencount = 0
				redcount = 0
				cond1 = col== 85
				cond2 = col == 170
				
				##### Filter1
				pixels1a=np.where(cond1,blank_image1,col)
				pixels1a = cv2.cvtColor(pixels1a,cv2.COLOR_BGR2GRAY)
				hist1a = cv2.calcHist([pixels1a],[0],None,[256],[0,256])
				if hist1a[29]>hist1a[150]:
					print('I vote RED')
					redcounta+=1
				elif hist1a[150]>hist1a[150]:
					print('I vote GREEN')
					greencounta+=1
				else:
					print('Conservation Mode: I vote RED')
					redcounta+=1
				
			
				
				
				pixels1b=np.where(cond2,blank_image1,col)
				pixels1b = cv2.cvtColor(pixels1b,cv2.COLOR_BGR2GRAY)
				hist1b = cv2.calcHist([pixels1b],[0],None,[256],[0,256])
				if hist1b[29]>hist1b[150]:
					print('I vote RED')
					redcountb+=1
				elif hist1b[150]>hist1b[150]:
					print('I vote GREEN')
					greencountb+=1
				else:
					print('Conservation Mode: I vote RED')
					redcountb+=1
				
				
				
				
				##### Filter2
				pixels2a=np.where(cond1,blank_image2,col)
				pixels12a = cv2.cvtColor(pixels2a,cv2.COLOR_BGR2GRAY)
				hist2a = cv2.calcHist([pixels2a],[0],None,[256],[0,256])
				if hist2a[29]>hist2a[150]:
					print('I vote RED')
					redcounta+=1
				elif hist2a[150]>hist2a[150]:
					print('I vote GREEN')
					greencounta+=1
				else:
					print('Conservation Mode: I vote RED')
					redcounta+=1
				
				
				pixels2b=np.where(cond2,blank_image2,col)
				pixels2b = cv2.cvtColor(pixels2b,cv2.COLOR_BGR2GRAY)
				hist2b = cv2.calcHist([pixels2b],[0],None,[256],[0,256])
				if hist2b[29]>hist2b[150]:
					print('I vote RED')
					redcountb+=1
				elif hist2b[150]>hist2b[150]:
					print('I vote GREEN')
					greencountb+=1
				else:
					print('Conservation Mode: I vote RED')
					redcountb+=1
				
			
				
				##### Filter3
				pixels3a=np.where(cond1,blank_image3,col)
				pixels13a = cv2.cvtColor(pixels3a,cv2.COLOR_BGR2GRAY)
				hist3a = cv2.calcHist([pixels3a],[0],None,[256],[0,256])
				if hist3a[29]>hist3a[150]:
					print('I vote RED')
					redcounta+=1
				elif hist3a[150]>hist3a[150]:
					print('I vote GREEN')
					greencounat+=1
				else:
					print('Conservation Mode: I vote RED')
					redcounta+=1
				
				pixels3b=np.where(cond2,blank_image3,col)
				pixels3b = cv2.cvtColor(pixels3b,cv2.COLOR_BGR2GRAY)
				hist3b = cv2.calcHist([pixels3b],[0],None,[256],[0,256])
				if hist3b[29]>hist3b[150]:
					print('I vote RED')
					redcountb+=1
				elif hist3b[150]>hist3b[150]:
					print('I vote GREEN')
					greencountb+=1
				else:
					print('Conservation Mode: I vote RED')
					redcountb+=1
				

				
				##### Filter4
				pixels4a=np.where(cond1,blank_image4,col)
				pixels14a = cv2.cvtColor(pixels4a,cv2.COLOR_BGR2GRAY)
				hist4a = cv2.calcHist([pixels4a],[0],None,[256],[0,256])
				if hist4a[29]>hist4a[150]:
					print('I vote RED')
					redcounta+=1
				elif hist4a[150]>hist4a[150]:
					print('I vote GREEN')
					greencounta+=1
				else:
					print('Conservation Mode: I vote RED')
					redcounta+=1
				
			
				pixels4b=np.where(cond2,blank_image4,col)
				pixels4b = cv2.cvtColor(pixels4b,cv2.COLOR_BGR2GRAY)
				hist4b = cv2.calcHist([pixels4b],[0],None,[256],[0,256])
				if hist4b[29]>hist4b[150]:
					print('I vote RED')
					redcountb+=1
				elif hist4b[150]>hist4b[150]:
					print('I vote GREEN')
					greencountb+=1
				else:
					print('Conservation Mode: I vote RED')
					redcountb+=1
			
				##### Filter5
				pixels5a=np.where(cond1,blank_image5,col)
				pixels15a = cv2.cvtColor(pixels5a,cv2.COLOR_BGR2GRAY)
				hist5a = cv2.calcHist([pixels5a],[0],None,[256],[0,256])
				if hist5a[29]>hist5a[150]:
					print('I vote RED')
					redcounta+=1
				elif hist5a[150]>hist5a[150]:
					print('I vote GREEN')
					greencounta+=1
				else:
					print('Conservation Mode: I vote RED')
					redcounta+=1
				
		
				pixels5b=np.where(cond2,blank_image5,col)
				pixels5b = cv2.cvtColor(pixels5b,cv2.COLOR_BGR2GRAY)
				hist5b = cv2.calcHist([pixels5b],[0],None,[256],[0,256])
				if hist5b[29]>hist5b[150]:
					print('I vote RED')
					redcountb+=1
				elif hist5b[150]>hist5b[150]:
					print('I vote GREEN')
					greencountb+=1
				else:
					print('Conservation Mode: I vote RED')
					redcountb+=1
				
			
				
				
				
				if redcountb> greencountb:
					print("Making notuse red")
					notuse2[add_slic== 170] = RED
				else:
					notuse2[add_slic == 170] = GREEN
					print("Making notuse GREEN")
				
				
				if redcounta> greencounta:
					print("Making notuse red")
					notuse2[add_slic== 85] = RED
				else:
					notuse2[add_slic == 85] = GREEN
					print("Making notuse GREEN")
				
				
				
			############################################################
			cond = col>1
			###########Mask Filter1 #####################################
			
			if np.array(cond).shape != np.array(blank_image1).shape:
				x=image.transpose(1,0,2)
				f = np.flip(x,1)
				pixels1=np.where(cond,f,col)
					
			else:
				pixels1=np.where(cond,blank_image1,col)
			
			pixels1 = cv2.cvtColor(pixels1,cv2.COLOR_BGR2GRAY)
			hist1 = cv2.calcHist([pixels1],[0],None,[256],[0,256])
			if hist1[29]>hist1[150]:
				print('I vote RED')
				redcount+=1
			elif hist1[150]>hist1[150]:
				print('I vote GREEN')
				greencount+=1
			else:
				print('Conservation Mode: I vote RED')
				redcount+=1
			########## Mask Filter2 #####################################
			if np.array(cond).shape != np.array(blank_image2).shape:
				x=image.transpose(1,0,2)
				f = np.flip(x,1)
				pixels1=np.where(cond,f,col)	
			else:
				pixels2=np.where(cond,blank_image2,col)
			pixels2 = cv2.cvtColor(pixels2,cv2.COLOR_BGR2GRAY)
			hist2 = cv2.calcHist([pixels2],[0],None,[256],[0,256])
			if hist2[29]>hist1[150]:
				print('I vote RED')
				redcount+=1
			elif hist2[150]>hist1[150]:
				print('I vote GREEN')
				greencount+=1
			else:
				print('Conservation Mode: I vote RED')
				redcount+=1
			########## Mask Filter3 #####################################
			if np.array(cond).shape != np.array(blank_image3).shape:
				x=image.transpose(1,0,2)
				f = np.flip(x,1)
				pixels3=np.where(cond,f,col)	
			else:
				pixels3=np.where(cond,blank_image3,col)
			pixels3 = cv2.cvtColor(pixels3,cv2.COLOR_BGR2GRAY)
			hist3 = cv2.calcHist([pixels3],[0],None,[256],[0,256])
			if hist3[29]>hist1[150]:
				print('I vote RED')
				redcount+=1
			elif hist3[150]>hist1[150]:
				print('I vote GREEN')
				greencount+=1
			else:
				print('Conservation Mode: I vote RED')
				redcount+=1
			########## Mask Filter4 #####################################
			if np.array(cond).shape != np.array(blank_image4).shape:
				x=image.transpose(1,0,2)
				f = np.flip(x,1)
				pixels4=np.where(cond,f,col)	
			else:
				pixels4=np.where(cond,blank_image4,col)
			pixels4 = cv2.cvtColor(pixels4,cv2.COLOR_BGR2GRAY)	
			hist4 = cv2.calcHist([pixels4],[0],None,[256],[0,256])
			if hist3[29]>hist1[150]:
				print('I vote RED')
				redcount+=1
			elif hist3[150]>hist1[150]:
				print('I vote GREEN')
				greencount+=1
			else:
				print('Conservation Mode: I vote RED')
				redcount+=1
			########## Mask Filter5 #####################################
			if np.array(cond).shape != np.array(blank_image5).shape:
				x=image.transpose(1,0,2)
				f = np.flip(x,1)
				pixels5=np.where(cond,f,col)	
			else:
				pixels5=np.where(cond,blank_image5,col)
			pixels5 = cv2.cvtColor(pixels5,cv2.COLOR_BGR2GRAY)	
			hist5 = cv2.calcHist([pixels5],[0],None,[256],[0,256])
			if hist5[29]>hist1[150]:
				print('I vote RED')
				redcount+=1
			elif hist5[150]>hist1[150]:
				print('I vote GREEN')
				greencount+=1
			else:
				print('Conservation Mode: I vote RED')
				redcount+=1
			###########################################################
			
			
			
			if redcount> greencount:
				print("Making notuse red")
				notuse2[add_slic== 255] = RED
			else:
				notuse2[add_slic == 255] = GREEN
				print("Making notuse GREEN")

			plt.subplot(1, 3, 1)
			plt.title('Original')
			plt.axis('off')
			plt.imshow(img1, cmap="gray")
			plt.subplot(1, 3, 2)
			plt.title('SLIC Representation')
			plt.axis('off')
			plt.imshow(add_slic, cmap="gray")
			plt.subplot(1, 3, 3)
			plt.title('Final Decision')
			plt.axis('off')
			plt.imshow(notuse2, cmap="gray")
			plt.show()

		# plt.show()

