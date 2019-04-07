import random

def RectSize(rect):
    top, bottom, left, right = rect
    
    w = abs(right - left)
    h = abs(bottom - top)
    
    return w, h
    

def Crop(rect, crop_area):
    '''Clamp bbox with crop area'''
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

def Resize(rect, scale):
    '''Resize bbox'''
    top, bottom, left, right = rect
    scale_x, scale_y = scale
    
    top, bottom = round(top*scale_y), round(bottom*scale_y)
    left, right = round(left*scale_x), round(right*scale_x)
    
    return (top, bottom, left, right)

def CleverRandomCropArea(rect, img_size, crop_size=(640, 640)):
    
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