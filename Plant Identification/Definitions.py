# This script contains all functions used in the project: Plant 
# Identification of Namibian Encroacher Bush using Compuer Vision.
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

class definitions:
	
	##################### Feature descriptor: Histogram ################
	# The histogram feature descriptor calculates the histogram of an 
	# HSV image.
	# Inputs: RGB image, bin size of histogram and None
	# Returns: 1D array of the histogram
	
	def fd_histogram(image, bins, mask=None):
	    # convert the image to HSV color-space
	    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	    # compute the color histogram
	    hist  = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
	    # normalize the histogram
	    cv2.normalize(hist, hist)
	    # return the histogram
	    return hist.flatten()
	
	##################### Sliding Window ###############################
	# A sliding window is used to identify and predict classes within 
	# an image.
	# Inputs: RGB image, step size for sliding window and window size
	# i.e. tuple of window width and height
	
	def slidingwindow(image, stepSize, windowSize):
		for y in range(0,image.shape[0], stepSize):
			for x in range(0,image.shape[1],stepSize):
				yield(x,y,image[y:y+windowSize[1], x:x+windowSize[0]])
				
	
	############### Parameter Report ###################################
	# A report for the results of RandomizedsearchCV
	# Imput: Results from andomizedsearchCV
	# Returns: Text files containing the search results
	
	def report(results,n_top = 3):
		for i in range(1,n_top+1):
			candidates = np.flatnonzero(results['rank_test_score'] == i)
			for candidate in candidates:
				print('Model with rank{0}\n'.format(i))
				print('Mean validation score{0:.3f}(std:{1:.3f}\n)'.format(results['mean_test_score'][candidate],results['std_test_score'][candidate]))
				print('Parameters: {0}'.format(results['params'][candidate]))
				file ='textfiles/params_smaller_bin.txt'
				filep = open(file,'a')
				filep.write('Model with rank{0}\n'.format(i))
				filep.write(' ')
				filep.write('Mean validation score{0:.3f}(std:{1:.3f}\n)'.format(results['mean_test_score'][candidate],results['std_test_score'][candidate]))
				filep.write(' ')
				filep.write('Parameters: {0}'.format(results['params'][candidate]))
				filep.write('\n')
				filep.close
				
	################# Read Lines of Textfiles ##########################
	# This function reads the lines of the text files, where the image
	# destinations are stored in, number of instances found and the 
	# corresponding window coordinates.
	# Input: Line of textfile
	# Output: path of the file, coordinates and instance count
	def parse_line(line):
    
	    toks = line.split(' ')
	    
	    frame_path = toks[0]  
	    label_count = int(toks[1])
	    pts = toks[2:]
	    pts = [ int(x) for x in pts ] 
			
	    return frame_path, pts, label_count
	 
	####################### Cleaner ####################################
	# Cleans all ",","[", "]" from text files.
	# Input: destination of 'dirty' and 'clean' text files
	# Output: Clean text file
	def cleaner(infile,outfile):
		delete_list = [",","[", "]"]
		fin = open(infile)
		fout = open(outfile, "w+")
		
		for line in fin:
			for word in delete_list:
				line = line.replace(word, "")
			fout.write(line)
		fin.close()
		fout.close()
		
	
	##################### Filter 6  ####################################			
	#	Filter 6 removes shadows, sand and sky, as well as background noise
	# Only core information is left in the image. It works exceptionally 
	# for training on trees and larger bushes. If the histogram bin size
	# is increased above 4, then it is shown that training with this 
	# filter is successful in identifying both classes.
	# Input: rgb image, lower and upper nound of colour segmentation
	# Returns: feature vector (global_features) and corresponing labels
	# (labels)

