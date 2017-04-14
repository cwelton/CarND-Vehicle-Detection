#!/usr/bin/env python

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
from features import *
from scipy.ndimage.measurements import label

# Define a single function that can extract features using hog sub-sampling
# and make predictions
def find_cars(img, state, box, scale):

    xstart, ystart = box[0]
    xstop, ystop = box[1]
    
    img = img.astype(np.float32)/255
    
    img_tosearch = img[ystart:ystop,xstart:xstop,:]
    ctrans_tosearch = normalized_colorspace(img_tosearch, state.cspace)
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch,
                                     (np.int(imshape[1]/scale),
                                      np.int(imshape[0]/scale)))
        
    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // state.pix_per_cell)-1
    nyblocks = (ch1.shape[0] // state.pix_per_cell)-1 
    nfeat_per_block = state.orient*state.cell_per_block**2
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // state.pix_per_cell)-1 
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step
    boxes = []
    
    # Compute individual channel HOG features for the entire image
    if state.hog_feat:
        hog1 = get_hog_features(ch1, state, feature_vec=False)
        hog2 = get_hog_features(ch2, state, feature_vec=False)
        hog3 = get_hog_features(ch3, state, feature_vec=False)
        
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            xleft = xpos*state.pix_per_cell
            ytop = ypos*state.pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))

            # Build feature vector
            features = []
            if state.spatial_feat:
                spatial_features = bin_spatial(subimg, size=state.spatial_size)
                features.append(spatial_features)
                
            if state.hist_feat:
                hist_features = color_hist(subimg, nbins=state.hist_bins)
                features.append(hist_features)

            if state.hog_feat:
                # Extract HOG for this patch
                hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
                features.append(hog_features)

            # Scale features and make a prediction
            test_features = state.scale(np.array([np.concatenate(features)]))
            test_prediction = state.model.predict(test_features)
            
            if test_prediction == 1:
                left = np.int(xleft*scale)+xstart
                top = np.int(ytop*scale)+ystart
                right = left + np.int(window*scale)
                bottom = top + np.int(window*scale)
                boxes.append(((left, top),(right, bottom)))
                
    return boxes

def draw_boxes(img, boxes, color=(0,0,255), width=6):
    draw_img = np.copy(img)    
    for box in boxes:
        cv2.rectangle(draw_img, box[0], box[1], color, width)
    return draw_img


class Heatmap(object):

    def __init__(self, img):
        self.heatmap = np.zeros_like(img[:,:,0]).astype(np.float)

    def add_heat(self, boxes, weight=1):
        '''Adds weight to each box in the supplied list of boxes.
           Assumes each "box" takes the form ((x1, y1), (x2, y2))
        '''
        for box in boxes:
            self.heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += weight

    def decay(self, factor=0.85):
        self.heatmap *= factor

    def get_labels(self, threshold=1):
        thresh = self.heatmap.copy()
        thresh[thresh <= threshold] = 0
        return label(thresh)

    def draw_labels(self, img, threshold=1):
        out_img = img.copy()
        labels = self.get_labels(threshold)
        
        for car_number in range(1, labels[1]+1):
            # Find pixels with each car_number label value
            nonzero = (labels[0] == car_number).nonzero()
            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            # Define a bounding box based on min/max x and y
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
            # Draw the box on the image
            cv2.rectangle(out_img, bbox[0], bbox[1], (0,0,255), 6)

        # Return the image
        return out_img
        

def _main():

    #img = mpimg.imread('test_images/test1.jpg')
    #img2 = mpimg.imread('test_images/test2.jpg')
    #img3 = mpimg.imread('test_images/test3.jpg')
    
    img = mpimg.imread('test_images/0960_original.jpg')
    img2 = mpimg.imread('test_images/0961_original.jpg')
    img3 = mpimg.imread('test_images/0962_original.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)

    h, w = img.shape[0:2]
    search_1 = ((int(w*0.05), int(h*0.55)), (int(w*0.95), int(h*0.75)))
    search_2 = ((0, int(h*0.55)), (w, int(h*0.9)))
        
    state = FeatureState.load("linear_svm_model.p")
    heat = Heatmap(img)
    
    boxes = []
    boxes += find_cars(img, state, search_1, scale=1)
    boxes += find_cars(img, state, search_2, scale=2)
    heat.add_heat(boxes)
    out1 = draw_boxes(img, boxes)
    hmap1 = np.clip(heat.heatmap, 0, 255)
    labels1 = heat.draw_labels(img, threshold=1)

    plt.subplot(3,4,1).set_title("image 1")
    plt.imshow(img)
    plt.subplot(3,4,2).set_title("boxes 1")
    plt.imshow(out1)
    plt.subplot(3,4,3).set_title("heat 1")
    plt.imshow(hmap1)
    plt.subplot(3,4,4).set_title("labels 1")
    plt.imshow(labels1)

    # heat = Heatmap(img2)    
    boxes = []
    boxes += find_cars(img2, state, search_1, scale=1)
    boxes += find_cars(img2, state, search_2, scale=2) 
    heat.decay()
    heat.add_heat(boxes)
    out2 = draw_boxes(img2, boxes)
    hmap2 = np.clip(heat.heatmap, 0, 255)
    labels2 = heat.draw_labels(img2, threshold=1)

    plt.subplot(3,4,5).set_title("image 2")
    plt.imshow(img2)
    plt.subplot(3,4,6).set_title("boxes 2")
    plt.imshow(out2)
    plt.subplot(3,4,7).set_title("heat 2")
    plt.imshow(hmap2)
    plt.subplot(3,4,8).set_title("labels 2")
    plt.imshow(labels2)    

    # heat = Heatmap(img3)    
    boxes = []
    boxes += find_cars(img3, state, search_1, scale=1)
    boxes += find_cars(img3, state, search_2, scale=2) 
    heat.decay()
    heat.add_heat(boxes)
    out3 = draw_boxes(img3, boxes)
    hmap3 = np.clip(heat.heatmap, 0, 255)    
    labels3 = heat.draw_labels(img3, threshold=1)

    plt.subplot(3,4,9).set_title("image 3")
    plt.imshow(img3)
    plt.subplot(3,4,10).set_title("boxes 3")
    plt.imshow(out3)
    plt.subplot(3,4,11).set_title("heat 3")
    plt.imshow(hmap3)
    plt.subplot(3,4,12).set_title("labels 3")
    plt.imshow(labels3)    
    
    plt.show()

    
if __name__ == '__main__':    
    _main()
