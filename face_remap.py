import torch
import torch.nn.functional as F
import math
import numpy as np
import cv2

from PIL import Image, ImageFilter
from comfy.utils import common_upscale

def resize_tensor(tensor, new_size):
    # 注意：tensor 的形状应该是 [C, H, W]，这里假设已经有该格式的 tensor
    # 需要将 [C, H, W] 转换为 [N, C, H, W] 形状，N = 1（因为只有一个图像）
    tensor = tensor.unsqueeze(0)  # [1, C, H, W]

    # 使用 interpolate 进行 resize
    tensor_resized = F.interpolate(tensor, size=new_size, mode='bilinear', align_corners=False)

    # 去掉 batch 维度，恢复到 [C, H, W] 形状
    return tensor_resized.squeeze(0)

def tensor2np(tensor: torch.Tensor):
    if len(tensor.shape) == 3:  # Single image
        return np.clip(255.0 * tensor.cpu().numpy(), 0, 255).astype(np.uint8)
    else:  # Batch of images
        return [np.clip(255.0 * t.cpu().numpy(), 0, 255).astype(np.uint8) for t in tensor]

def mask2image(mask:torch.Tensor)  -> Image:
    masks = tensor2np(mask)
    for m in masks:
        _mask = Image.fromarray(m).convert("L")
        _image = Image.new("RGBA", _mask.size, color='white')
        _image = Image.composite(
            _image, Image.new("RGBA", _mask.size, color='black'), _mask)
    return _image

def tensor2pil(t_image: torch.Tensor)  -> Image:
    return Image.fromarray(np.clip(255.0 * t_image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def mask_area(image:Image) -> tuple:
    np_img_array = np.asarray(image.convert('RGBA'))
    cv2_image =  cv2.cvtColor(np_img_array, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, 0)
    locs = np.where(thresh == 255)
    x1 = np.min(locs[1]) if len(locs[1]) > 0 else 0
    x2 = np.max(locs[1]) if len(locs[1]) > 0 else image.width
    y1 = np.min(locs[0]) if len(locs[0]) > 0 else 0
    y2 = np.max(locs[0]) if len(locs[0]) > 0 else image.height
    x1, y1, x2, y2 = min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)
    return (x1, y1, x2 - x1, y2 - y1)

def fn_x(a,b):
    return math.sqrt(a**2 + b**2)

def empty_image(width, height, batch_size=1, color=0):
    r = torch.full([batch_size, height, width, 1], ((color >> 16) & 0xFF) / 0xFF)
    g = torch.full([batch_size, height, width, 1], ((color >> 8) & 0xFF) / 0xFF)
    b = torch.full([batch_size, height, width, 1], ((color) & 0xFF) / 0xFF)
    return torch.cat((r, g, b), dim=-1)

def mask_to_box(mask):
    if mask.dim() == 2:
            mask = torch.unsqueeze(mask, 0)
    if mask.shape[0] > 0:
        mask = torch.unsqueeze(mask[0], 0)

    _mask = mask2image(mask).convert('RGB')
    _mask = _mask.filter(ImageFilter.GaussianBlur(radius=20))
    return mask_area(_mask)

def get_image_size(image):
    if image.shape[0] > 0:
        image = torch.unsqueeze(image[0], 0)
    _image = tensor2pil(image)
    return (_image.width, _image.height)

def get_center(x,y,w,h):
    return (x+w/2,y+h/2)

def create_mask_from_coordinates(empty_img, x_start, y_start, x_end, y_end):
    # 创建与 empty_img 大小相同的全零 mask [1, H, W]
    mask = torch.zeros_like(empty_img[:, :, :, 0])  # 获取 empty_img 的高度和宽度，并创建一个[1, H, W]的全零张量

    # 确保目标区域不超出边界
    y_end = min(y_end, empty_img.shape[1])  # 对应 height
    x_end = min(x_end, empty_img.shape[2])  # 对应 width

    # 将指定范围的区域设置为 1，表示这个区域是蒙版区域
    mask[:, y_start:y_end, x_start:x_end] = 1.0  # [1, H, W] 的 mask，其中范围内的部分为1

    return mask

class FaceRemap:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):
        ratio_mode = ['width', 'height', 'catercorner']
        return {
            'required': {
                "object_image": ("IMAGE", ),
                'object_mask': ("MASK", ),
                "coordinate_image": ("IMAGE", ),
                'coordinate_mask': ("MASK", ),
                "ratio_mode": (ratio_mode,), 
            }
        }
    
    def mask_box_detect(self, object_image, object_mask, coordinate_image, coordinate_mask, ratio_mode):

        (x1,y1,w1,h1) = mask_to_box(object_mask)
        (img1_w,img1_h) = get_image_size(object_image)
        (x2,y2,w2,h2) = mask_to_box(coordinate_mask)
        (img2_w,img2_h) = get_image_size(coordinate_image)
        (px1, py1) = get_center(x1,y1,w1,h1)
        (px2, py2) = get_center(x2,y2,w2,h2)

        ratio = 1
        if ratio_mode == 'width':
            ratio = w2/w1
        elif ratio_mode == 'height':
            ratio = h2/h1
        else: 
            ratio = fn_x(w2,h2) / fn_x(w1, h1)

        x = int(px2 - px1 * ratio)
        y = int(py2 - py1 * ratio)
        width = int(img1_w * ratio)
        height = int(img1_h * ratio)

        print(f"x: {x}, y: {y}, width: {width}, height: {height}, {object_image.shape}, {object_mask.shape}")

        empty_img = empty_image(img2_w, img2_h)

        print(f'{empty_img.shape}')

        object_image_resized = object_image.movedim(-1,1)
        object_image_resized = common_upscale(object_image_resized, width, height, "lanczos", "center")
        object_image_resized = object_image_resized.movedim(1,-1)

        print(f'{object_image_resized.shape}')

        # 计算放置范围
        x_end = min(x + width, img2_w)
        y_end = min(y + height, img2_h)

        # 计算实际放置的位置
        x_start = max(x, 0)
        y_start = max(y, 0)

        updated_imageY = empty_img.clone()
        # 使用切片将 object_image_resized 插入到 empty_img 中
        updated_imageY[:, y_start:y_end, x_start:x_end, :] = object_image_resized[:, :y_end - y_start, :x_end - x_start, :]
        mask = create_mask_from_coordinates(empty_img, x_start, y_start, x_end, y_end)

        return (updated_imageY, mask) 
    
    RETURN_TYPES = ("IMAGE","MASK")
    FUNCTION = "mask_box_detect"
    CATEGORY = 'face_remap'
    OUTPUT_NODE = True

NODE_CLASS_MAPPINGS = {
    "FaceRemap": FaceRemap,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FaceRemap": 'FaceRemap'
}
