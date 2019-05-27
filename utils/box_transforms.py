import random
from torchvision import transforms

def RectSize(rect):
    top, bottom, left, right = rect #TODO (left,top,right,bottom), but it's tied on metadate
    w = abs(right - left)
    h = abs(bottom - top)
    return h, w
    
def CropBox(rect, crop_area):
    top, bottom, left, right = rect
    x0, y0, x1, y1 = crop_area
    
    if x0>=right or x1<=left or y0>=bottom or y1<=top:
        print("Invalid dimensions")
        return None
    
    left = max(0, left - x0)
    top = max(0, top - y0)
    right = min(right - x0, x1)
    bottom = min(bottom - y0, y1)
    
    return (top, bottom, left, right)

def ResizeBox(rect, scale):
    top, bottom, left, right = rect
    scale_x, scale_y = scale
    
    top, bottom = round(top*scale_y), round(bottom*scale_y)
    left, right = round(left*scale_x), round(right*scale_x)
    
    return (top, bottom, left, right)

def CleverRandomCropArea(rect, img_size, crop_size=(320, 320)):
    
    top, bottom, left, right = rect
    w, h = img_size
    crop_w, crop_h = crop_size
    
    if w < crop_w or h < crop_h:
        print("Need to resize first")
        return (0, 0, img_size[0], img_size[1])
    
    x0 = random.randint(max(0, right-crop_w), min(left, w - crop_w))
    y0 = random.randint(max(0, bottom-crop_h), min(top, h - crop_h))
    x1 = x0 + crop_w
    y1 = y0 + crop_h
    return (x0, y0, x1, y1)

def expand_box(box, img_size=(320,320), rate=0.2):
    top, bottom, left, right = box
    
    h = abs(bottom-top)
    w = abs(right-left)
    
    img_h, img_w = img_size
    
    left = max(0, left - rate*w)
    top = max(0, top - rate*h)
    right = min(right + rate*w, img_w)
    bottom = min(bottom + rate*h, img_h)
    
    return (top, bottom, left, right)

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, box=None):
        for t in self.transforms:
            img, box = t(img, box)
        return img, box

class ResizeWithBox(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, box=None):
        w, h = img.size
        
        img = transforms.Resize(self.size)(img)
        
        if box:
            if isinstance(self.size, int):
                dim = min(w,h)
                scale = self.size/dim
                box = ResizeBox(box, (scale,scale))
            else:
                scale_y = self.size[0]/h
                scale_x = self.size[1]/w
                box = ResizeBox(box, (scale_x,scale_y))
        
        return img, box


class RandomCropWithBox(object):
    def __init__(self, size):
        if isinstance(size, int):  
            self.size = (size, size)
        else:
            self.size = size

    def __call__(self, img, box=None):
        if box:
            crop_rect = CleverRandomCropArea(box, img.size, crop_size=self.size)
            img = img.crop(crop_rect)
            box = CropBox(box, crop_rect)
        else:
            img = transforms.RandomCrop(self.size)(img)
        
        return img, box


class CenterCropWithBox(object):
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size

    def __call__(self, img, box=None):
        if box:
            w, h = img.size
            top, bottom, left, right = box
            
            x0, y0, x1, y1 = w/2 - self.size[1]/2, h/2 - self.size[0]/2,  w/2 + self.size[1]/2, h/2 + self.size[0]/2
            
            offset_y = min(0, top-y0)
            if offset_y == 0:
                offset_y = max(0, bottom-y1)
            
            offset_x = min(0, left-x0)
            if offset_x == 0:
                offset_x = max(0, right-x1)
            
            x0, y0, x1, y1 = x0+offset_x, y0+offset_y, x1+offset_x, y1+offset_y
            
            img = img.crop((x0, y0, x1, y1))
            box = CropBox(box, (x0, y0, x1, y1))
        else:
            img = transforms.CenterCrop(self.size)(img)
        
        return img, box
