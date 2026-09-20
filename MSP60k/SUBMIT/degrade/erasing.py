import os
import random
from PIL import Image
import math
import numpy as np
import cv2
class Occlusion_Adding(object):
    def __init__(self, max_erasing_size):
        super(Occlusion_Adding, self).__init__()   #调用父类的构造函数
        self.occlusion_list = ['bench', 'bike', 'car', 'card', 'chair', 
                               'firehydrant', 'motorbike', 
                               'person', 'post', 'roadsign', 'stone', 'umbrella','balloon']
        self.area_ratio_dict = {'bench':0.4*max_erasing_size, 'bike':0.6*max_erasing_size, 
                                'car': 0.45*max_erasing_size, 'card': 0.15*max_erasing_size, 
                                'chair': 0.65*max_erasing_size, 'firehydrant': 0.2*max_erasing_size, 
                                'motorbike': 0.5*max_erasing_size, 'pedestrian': 0.4*max_erasing_size,
                                'person': 0.4*max_erasing_size, 'post': 0.25*max_erasing_size, 
                                'roadsign': 0.4*max_erasing_size, 
                                'stone': 0.15*max_erasing_size, 
                                'umbrella': 0.25*max_erasing_size,
                                'balloon': 0.3*max_erasing_size}
    
    def __call__(self, holistic_img, occlusion_path):
        # occlusion_type = random.choice(self.occlusion_list)
        occlusion_type = 'person'
        # print("occlusion_type:", occlusion_type)
        i = random.randint(1, 4)
        occlusion_path = os.path.join(occlusion_path, occlusion_type + str(i) + '.png')
        verse = Image.open(occlusion_path)
        verse = verse.convert("RGBA")
        w_h_ratio = verse.size[0] / verse.size[1]
        
        holistic_width = holistic_img.shape[1]
        holistic_height = holistic_img.shape[0]

        holistic_area = holistic_width * holistic_height
        area_ratio = self.area_ratio_dict[occlusion_type]
        
        new_occlude_area = area_ratio * holistic_area
        occlude_h = int(round(math.sqrt(new_occlude_area / w_h_ratio)))
        occlude_w = int(round(math.sqrt(new_occlude_area * w_h_ratio)))

        
        verse_resized = verse.resize((occlude_w, occlude_h))
        
        # 确定遮挡位置
        location_x, location_y = self.get_occlusion_location(occlusion_type, occlude_w, occlude_h, holistic_width, holistic_height)
      
        # 创建一个RGBA的空白图像，并将verse_np粘贴上去
        overlay = Image.new('RGBA', (holistic_width, holistic_height))
        overlay.paste(verse_resized, (location_x, location_y), verse_resized)
        
        # 将overlay转换为NumPy数组
        overlay_np = np.array(overlay)
        
        # 将overlay_np和holistic_img进行融合
        alpha_mask = overlay_np[..., 3] / 255.0
        overlay_np = cv2.cvtColor(overlay_np, cv2.COLOR_RGBA2BGR)
        for c in range(0, 3):
            holistic_img[..., c] = alpha_mask * overlay_np[..., c] + (1 - alpha_mask) * holistic_img[..., c]

        return holistic_img

    def get_occlusion_location(self, occlusion_type, occlude_w, occlude_h, holistic_width, holistic_height):
        # 初始化位置
        location = (0, 0)
        
        # 根据不同的遮挡类型确定位置
        if occlusion_type in ['stone', 'motorbike', 'bench', 'bike', 'car', 'card', 'chair', 'post']:
            location = (0, holistic_height - occlude_h)
        elif occlusion_type in ['roadsign', 'firehydrant', 'pedestrian', 'person']:
            
            location = (holistic_width - occlude_w, holistic_height - occlude_h)
        elif occlusion_type in ['umbrella', 'kite']:
            location = (0, 0)
        elif occlusion_type in ['balloon']:
            location_w = random.randint(-int(occlude_w/2), holistic_width-10)
            location = (location_w, int(holistic_height/2)- occlude_h)       
        return location

def add_random_occlusion(holistic_img, occlusion_folder='/data/jinjiandong/datasets/MSP/sucai', max_erasing_size=1):
    max_erasing_size = 1
    occlusion_adder = Occlusion_Adding(max_erasing_size=max_erasing_size)
    occluded_image = occlusion_adder(holistic_img, occlusion_folder)
    
    return occluded_image



# def test_random_occlusion_function(image_path, occlusion_folder, save_path, occlusion_list=None, max_erasing_size=1, output_format='JPEG'):
#     # 调用添加随机遮挡函数
#     occluded_image = add_random_occlusion(image_path, occlusion_folder, save_path, occlusion_list=occlusion_list, max_erasing_size=max_erasing_size, output_format=output_format)
#     occluded_image.save(save_path, format=output_format)
#     print(f"Image with random occlusion saved to {save_path}")
# # 执行测试函数
# image_path = '/data/jinjiandong/datasets/MSP/images/part17_20231212112816_00313_009.jpg'

#     # 素材文件夹路径
# occlusion_folder = '/data/jinjiandong/datasets/MSP/sucai'

# # 保存结果的路径
# save_path = 'occluded_image.jpg'
# test_random_occlusion_function(image_path, occlusion_folder, save_path)