# Helper functions for creating custom augmentations
# for Pytorch projects

import numpy as np
import cv2


class BinaryThreshold:
    ''' Threshold given np.array such that values are either value {min, max}
    '''
    def __init__(self, threshold=127, target_val=255):
        '''
        :param target_val: value to set all values to that are >= threshold
        :param threshold: value to compare all array elements by and assign new value
        '''
        self.threshold = threshold
        self.target_val = target_val

    def __call__(self, data: np.array) -> np.array:
        ret, bin_thresh = cv2.threshold(data, self.threshold, self.target_val, cv2.THRESH_BINARY)
        return bin_thresh

