import torch
import torchvision.transforms.functional as F
import numpy as np
import cv2

#This will return a fully padded image 
class FullPad:

    def __init__(self, target_image_width, target_image_height ):
        self.target_image_width = target_image_width
        self.target_image_height = target_image_height


    def __call__(self, image):
        
        #print(type(image))

        #get the width and height from the original image
        s = image.size()
        width = s[-1]
        height = s[-2]

        #if image is bigger than target size then just return the image
        if width > self.target_image_width and height > self.target_image_height:
            return image

        horizontal_offset = int((self.target_image_width - width) / 2)
        vertical_offset = int((self.target_image_height - height) / 2)

        np_img = np.asarray(image)
        #sample each of the four corners of the image to get an approsimate background colour
        top_left_value = np_img[0][0][0]
        bottom_left_value = np_img[0][-1][0]
        top_right_value = np_img[0][0][-1]
        bottom_right_value = np_img[0][-1][-1]

        average_background_value = (top_left_value.item() + bottom_left_value.item() + top_right_value.item() + bottom_right_value.item()) / 4
        #print(average_background_value)
        padding = (horizontal_offset, vertical_offset)
        
        return F.pad(image, padding, average_background_value, 'constant')


class SquarePad:
    def __call__(self, image):
        s = image.size()

        max_wh = np.max([s[-1], s[-2]])
        hp = int((max_wh - s[-1]) / 2)
        vp = int((max_wh - s[-2]) / 2)
        
        
        #sample each of the four corners of the image to get an approsimate background colour
        top_left_value = image[0][0][0]
        bottom_left_value = image[0][-1][0]
        top_right_value = image[0][0][-1]
        bottom_right_value = image[0][-1][-1]

        average_value = (top_left_value.item() + bottom_left_value.item() + top_right_value.item() + bottom_right_value.item()) / 4
        #print(average_value)

        padding = (hp, vp, hp, vp)

        return F.pad(image, padding, average_value, 'constant')
       

class ReflectPad(torch.nn.Module):
    def forward (self,image):

        #get the width and height from the original image
        s = image.size()
        width = s[-1]
        height = s[-2]

        #if image is bigger than target size then just return the image
        if width > 299 and height > 299:
            return image

        horizontal_offset = int((299 - width) / 2)
        vertical_offset = int((299 - height) / 2)

        if horizontal_offset < 0:
            horizontal_offset = 0

        if vertical_offset < 0:
            vertical_offset = 0

        numpy_image = image.numpy()
        #numpy_image = np.float32(image)
        cv2_image = np.transpose(numpy_image, (1, 2, 0))
        cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_RGB2BGR)

        #do the openCV stuff here
        cv2_image = cv2.copyMakeBorder(cv2_image, vertical_offset, vertical_offset , horizontal_offset, horizontal_offset, cv2.BORDER_REFLECT)
        cv2_image = cv2.resize(cv2_image, (299,299),interpolation= cv2.INTER_NEAREST)
        new_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        new_image = np.transpose(new_image, (2, 0, 1))
        new_tensor = torch.from_numpy(new_image)
        return new_tensor   