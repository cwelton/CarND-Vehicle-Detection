#!/usr/bin/env python
'''
Provides functions for extracting feature vectors and classifying images.
'''

import cv2
import matplotlib.image as mpimg
import numpy as np
import pickle
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog


class FeatureState(object):

    def __init__(self, cspace='RGB', spatial_size=(32, 32), hist_bins=32,
                 orient=9, pix_per_cell=8, cell_per_block=2, hog_channel="ALL",
                 spatial_feat=False, hist_feat=False, hog_feat=False):
        self.cspace = cspace
        self.spatial_size = spatial_size
        self.hist_bins = hist_bins
        self.orient = orient
        self.pix_per_cell = pix_per_cell
        self.cell_per_block = cell_per_block
        self.hog_channel = hog_channel
        self.spatial_feat = spatial_feat
        self.hist_feat = hist_feat
        self.hog_feat = hog_feat
        self.scaler = None
        self.model = None

        print("Using Colorspace:", cspace)
        if spatial_feat:
            print("Calculating SPATIAL features using:", spatial_size, "scaled image")    

        if hist_feat:
            print("Calculating HIST features using:", hist_bins, "bins")

        if hog_feat:
            print('Calculating HOG features using:',orient,'orientations',pix_per_cell,
                  'pixels per cell and', cell_per_block,'cells per block')


    def fit_scaler(self, X):
        self.scaler = StandardScaler().fit(X)
        return self.scale(X)

    def scale(self, X):
        return self.scaler.transform(X)
    
    def set_model(self, model):
        self.model = model
        
    def save(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self, f)
            print("model written to", filename)

    @classmethod
    def load(_, filename):
        print("Using existing model")
        with open(filename, "rb") as f:
            return pickle.load(f)



def bin_spatial(img, size=(32, 32)):
    '''Calculate a spatial feature vector for an image.'''

    return cv2.resize(img, size).ravel()


def color_hist(img, nbins=32, bins_range=(0, 256)):
    '''Calculate a color histogram feature vector for an image.'''

    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    return hist_features


def get_hog_features(img, state, vis=False, feature_vec=True):
    '''Calculate a histogram of gradients (HOG) feature vector for an image.'''
    
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=state.orient,
                                  pixels_per_cell=(state.pix_per_cell, state.pix_per_cell),
                                  cells_per_block=(state.cell_per_block, state.cell_per_block),
                                  transform_sqrt=True, visualise=vis,
                                  feature_vector=feature_vec)
        return features, hog_image
    
    # Otherwise call with one output
    else:      
        features = hog(img, orientations=state.orient,
                       pixels_per_cell=(state.pix_per_cell, state.pix_per_cell),
                       cells_per_block=(state.cell_per_block, state.cell_per_block),
                       transform_sqrt=True, visualise=vis,
                       feature_vector=feature_vec)
        return features

def normalized_colorspace(img, cspace):
    '''Convert an image into a color space and normalize color channels.'''
    
    # apply color conversion if other than 'RGB'
    if cspace == 'RGB':
        feature_image = np.copy(img)
        im_max = feature_image.max()
        if im_max > 1:
            feature_image *= (1.0/im_max)
            
    elif cspace == 'HSV':
        feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        feature_image[:,:,0] *= (1.0/360)  # Hue
        feature_image[:,:,1:2] *= (1.0/100)  # Saturation and Value
        
    elif cspace == 'LUV':
        feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)  # includes negative values
        
    elif cspace == 'HLS':
        feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        feature_image[:,:,0] *= (1.0/360)  # Hue
        feature_image[:,:,1:2] *= (1.0/100)  # Saturation and Lumenocity        
        
    elif cspace == 'YUV':
        feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        
    elif cspace == 'YCrCb':
        feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        
    elif cspace == 'Composite':
        hls = normalized_colorspace(img, 'HLS')
        yCrCb = normalized_colorspace(img, 'YCrCb')
        feature_image = np.dstack([hls,yCrCb])
        
    else:
        raise Exception("Unexpected cspace = %s" % cspace)

    return feature_image

def single_img_features(img, state):
    '''Calculate a composite feature vector for an image.'''
    
    #1) Define an empty list to receive features
    img_features = []

    #2) Convert into the desired colorspace
    feature_image = normalized_colorspace(img, state.cspace)
    
    #3) Compute spatial features if flag is set
    if state.spatial_feat == True:
        spatial_features = bin_spatial(feature_image, size=state.spatial_size)
        img_features.append(spatial_features)

    #4) Compute histogram features if flag is set
    if state.hist_feat == True:
        hist_features = color_hist(feature_image, nbins=state.hist_bins)
        img_features.append(hist_features)

    #5) Compute HOG features if flag is set
    if state.hog_feat == True:
        if state.hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.extend(get_hog_features(feature_image[:,:,channel], state,
                                                     vis=False, feature_vec=True))
        else:
            hog_features = get_hog_features(feature_image[:,:,hog_channel], state,
                                            feature_vec=True)
            
        #8) Append features to list
        img_features.append(hog_features)

    #9) Return concatenated array of features
    return np.concatenate(img_features)



def extract_features(imgs, state):
    '''Calculate composite feature vectors for a set of images'''
    
    # Iterate through the list of images
    # Read in each one by one and extract the features
    features = []
    for fname in imgs:
        img = mpimg.imread(fname)
        img_features = single_img_features(img, state)
        features.append(img_features)

    return features

def _visualization():
    import matplotlib.pyplot as plt
    
    state = FeatureState(cspace = "YCrCb", spatial_feat = True, hist_feat = True,
                         hog_feat = True, orient = 12, pix_per_cell = 8,
                         cell_per_block = 2)
    car = mpimg.imread('examples/vehicle1.png')
    noncar = mpimg.imread('examples/nonvehicle1.png')

    car_cmap = normalized_colorspace(car, state.cspace)
    noncar_cmap = normalized_colorspace(noncar, state.cspace)
    
    car_feat, car_vis = get_hog_features(car_cmap[:,:,0], state, vis=True)
    noncar_feat, noncar_vis = get_hog_features(noncar_cmap[:,:,0], state, vis=True)

    plt.subplot(2,2,1).set_title('vehicle')
    plt.imshow(car_cmap[:,:,0], cmap='gray')
    plt.subplot(2,2,2).set_title('vehicle hog')
    plt.imshow(car_vis, cmap='gray')
    plt.subplot(2,2,3).set_title('nonvehicle')
    plt.imshow(noncar_cmap[:,:,0], cmap='gray')
    plt.subplot(2,2,4).set_title('nonvehice hog')
    plt.imshow(noncar_vis, cmap='gray')
    plt.show()

def _main():
    import glob
    import time
    import random
    from sklearn.model_selection import GridSearchCV
    from sklearn.model_selection import train_test_split, cross_val_score
    import os

    # Divide up into cars and notcars
    #cars = glob.glob('data/vehicles/GTI*/*.png')
    #notcars = glob.glob('data/non-vehicles/GTI/*.png')
    cars = glob.glob('data/vehicles/*/*.png')
    notcars = glob.glob('data/non-vehicles/*/*.png')
    random.shuffle(cars)
    random.shuffle(notcars)

    # Save/Restore of model
    model_name = "linear_svm_model.p"
    use_prior_model = os.path.exists(model_name)
    
    # Use a linear model, or do a grid search to find best model
    grid_search = False
    cross_validate = False
    sample = False
    
    # Reduce the sample size because HOG features are slow to compute
    if sample:
        sample_size = 500
    else:
        sample_size = min(len(cars),len(notcars))

    cars = cars[0:sample_size]
    notcars = notcars[0:sample_size]
    
    print("Using",sample_size,"samples for cars and not-cars")

    # Cross validate for a more robust score
    if use_prior_model:
        state = FeatureState.load(model_name)
    else:
        state = FeatureState(cspace = "YCrCb", spatial_feat = True, hist_feat = True,
                             hog_feat = True, orient = 12, pix_per_cell = 8,
                             cell_per_block = 2)

    print("Extracting Features")
    t=time.time()
    car_features = extract_features(cars, state)
    notcar_features = extract_features(notcars, state)
    t2 = time.time()
    print("Features extracted in", round(t2-t, 2), 'seconds')
    print("Feature vector length:", len(car_features[0]))
    
    # Create an array stack of feature vectors
    X = np.vstack((car_features, notcar_features)).astype(np.float64)                        
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))
    
    # Fit a per-column scaler, and apply the scalar to our data
    scaled_X = state.fit_scaler(X)

    # Shuffle and split data into training and test data
    X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2)

    if use_prior_model:
        clf = state.model

    elif grid_search:
        print("Performing Grid Search")
        tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]},
                            {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]    
        clf = GridSearchCV(SVC(C=1), tuned_parameters, cv=5)
        clf.fit(X_train, y_train)
        print(clf.best_params_)

        # These were the selected parameters on an earlier run
        clf = SVC(kernel='rbf', C=10, gamma=0.0001)
        clf.fit(X_train, y_train)

    elif cross_validate:
        print("Cross validating")
        clf = SVC(kernel='linear', C=10)

        # Cross_val_score will shuffle and cross validate the data between 5 different
        # train/test sets.
        t = time.time()
        scores = cross_val_score(clf, X_train, y_train, cv=5)
        t2 = time.time()    
        print("Cross validation mean score:",
              np.mean(scores), "+/-", np.std(scores), "in",
              round(t2-t, 2), "seconds")
        return

    else:
        print("Training linear model")
        clf = SVC(kernel='linear', C=10)
        t = time.time()
        clf.fit(X_train, y_train)
        t2 = time.time()
        print("Training completed in", round(t2-t, 2), "seconds")

    t = time.time()
    score = clf.score(X_test, y_test)
    t2 = time.time()
    print("Validation score:", score, "in", round(t2-t, 2), "seconds")

    if not use_prior_model:
        state.set_model(clf)
        state.save(model_name)

if __name__ == '__main__':
    #_visualization()
    _main()
