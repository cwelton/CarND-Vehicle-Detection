#!/usr/bin/env python

from moviepy.editor import VideoFileClip
from features import FeatureState
from windows import find_cars, draw_boxes, Heatmap

class VideoContext(object):

    def __init__(self, modelname, clipname):
        self.state = FeatureState.load(modelname)
        self.clip_in = VideoFileClip(clipname)
        self.shape = self.clip_in.size
        self.heatmap = None

    def save(self, outname):
        self.clip = self.clip_in.fl_image(self.next_image)
        self.clip.write_videofile(outname, audio=False)

    def next_image(self, img):
        if self.heatmap is None:
            self.heatmap = Heatmap(img)
        else:
            self.heatmap.decay()
        
        return self.process_image(img)

    def process_image(self, img):
        h, w = img.shape[0:2]
        search_1 = ((int(w*0.05), int(h*0.55)), (int(w*0.95), int(h*0.75)))
        search_2 = ((0, int(h*0.55)), (w, int(h*0.9)))

        boxes = find_cars(img, self.state, search_1, scale=1)
        self.heatmap.add_heat(boxes)
        boxes = find_cars(img, self.state, search_2, scale=2)
        self.heatmap.add_heat(boxes)
        return self.heatmap.draw_labels(img)

if __name__ == '__main__':
    v = VideoContext("linear_svm_model.p", "project_video.mp4")
    v.save("output.mp4")
    
