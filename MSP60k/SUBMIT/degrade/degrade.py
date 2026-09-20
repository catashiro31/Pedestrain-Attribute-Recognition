import cv2
import numpy as np
import os
from erasing import add_random_occlusion

def add_gaussian_noise(image, mean=0, std=25):
    """
    向图像添加高斯噪声。
    
    参数:
        image: 输入图像。
        mean: 高斯噪声的均值。均值越大，噪声越亮。
        std: 高斯噪声的标准差。标准差越大，噪声的强度越高。
        
    返回:
        含噪声的图像。
    """
    noise = np.random.normal(mean, std, image.shape).astype(np.uint8)
    noisy_image = cv2.add(image, noise)
    return noisy_image

def add_salt_and_pepper_noise(image, salt_prob=0.01, pepper_prob=0.01):
    """
    向图像添加椒盐噪声。
    
    参数:
        image: 输入图像。
        salt_prob: 盐噪声的概率。概率越大，图像中出现的白点越多。
        pepper_prob: 椒噪声的概率。概率越大，图像中出现的黑点越多。
        
    返回:
        含噪声的图像。
    """
    noisy_image = np.copy(image)
    num_salt = np.ceil(salt_prob * image.size)
    num_pepper = np.ceil(pepper_prob * image.size)
    
    # Add Salt noise
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    noisy_image[coords[0], coords[1], :] = 1

    # Add Pepper noise
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    noisy_image[coords[0], coords[1], :] = 0
    
    return noisy_image

def apply_motion_blur(image, size=15):
    """
    应用运动模糊。
    
    参数:
        image: 输入图像。
        size: 模糊核的大小。核越大，模糊程度越高。
        
    返回:
        模糊后的图像。
    """
    kernel_motion_blur = np.zeros((size, size))
    kernel_motion_blur[int((size-1)/2), :] = np.ones(size)
    kernel_motion_blur = kernel_motion_blur / size

    blurred_image = cv2.filter2D(image, -1, kernel_motion_blur)
    return blurred_image

def apply_gaussian_blur(image, kernel_size=5, sigma=1):
    """
    应用高斯模糊。
    
    参数:
        image: 输入图像。
        kernel_size: 高斯核的大小。核越大，模糊程度越高。
        sigma: 高斯核的标准差。标准差越大，模糊程度越高。
        
    返回:
        模糊后的图像。
    """
    blurred_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    return blurred_image

def apply_median_blur(image, kernel_size=5):
    """
    应用中值模糊。
    
    参数:
        image: 输入图像。
        kernel_size: 中值滤波核的大小。核越大，模糊程度越高。
        
    返回:
        模糊后的图像。
    """
    return cv2.medianBlur(image, kernel_size)

def apply_poisson_noise(image):
    """
    应用泊松噪声。
    
    参数:
        image: 输入图像。
        
    返回:
        含噪声的图像。
    """
    vals = len(np.unique(image))
    vals = 2 ** np.ceil(np.log2(vals))
    noisy = np.random.poisson(image * vals) / float(vals)
    noisy_image = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy_image

def apply_speckle_noise(image, std=0.1):
    """
    应用散斑噪声。
    
    参数:
        image: 输入图像。
        std: 噪声的标准差。标准差越大，噪声的强度越高。
        
    返回:
        含噪声的图像。
    """
    noise = np.random.randn(*image.shape) * std
    noisy_image = image + image * noise
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    return noisy_image

def apply_brightness_change(image, factor=0.5):
    """
    改变图像亮度。
    
    参数:
        image: 输入图像。
        factor: 亮度变化因子。因子越小，图像越暗；因子越大，图像越亮。
        
    返回:
        亮度变化后的图像。
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = hsv[:, :, 2] * factor
    changed_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return changed_image

def apply_contrast_change(image, alpha=1.5, beta=0):
    """
    改变图像对比度。
    
    参数:
        image: 输入图像。
        alpha: 对比度变化因子。因子越大，对比度越高。
        beta: 调整因子。用于对图像进行整体亮度调整。
        
    返回:
        对比度变化后的图像。
    """
    new_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return new_image

def apply_random_erasing(image, sl=0.02, sh=0.4, r1=0.3):
    """
    应用随机擦除。
    
    参数:
        image: 输入图像。
        sl: 擦除区域最小比例。
        sh: 擦除区域最大比例。
        r1: 擦除区域长宽比。
        
    返回:
        擦除后的图像。
    """
    h, w, _ = image.shape
    area = h * w

    target_area = np.random.uniform(sl, sh) * area
    aspect_ratio = np.random.uniform(r1, 1/r1)

    h_erasing = int(round(np.sqrt(target_area * aspect_ratio)))
    w_erasing = int(round(np.sqrt(target_area / aspect_ratio)))

    if h_erasing < h and w_erasing < w:
        x1 = np.random.randint(0, h - h_erasing)
        y1 = np.random.randint(0, w - w_erasing)

        image[x1:x1 + h_erasing, y1:y1 + w_erasing, :] = 0
        
    return image
import random
def apply_object_erasingv2(image, objects_folder):
    h, w, _ = image.shape
    
    # 获取素材文件夹中的所有物体图片路径
    object_images = [os.path.join(objects_folder, filename) for filename in os.listdir(objects_folder)]
    
    # 从物体图片中随机选择一张
    object_image_path = random.choice(object_images)
    object_img = cv2.imread(object_image_path)
    
    # 调整物体图片的大小以匹配遮挡区域大小
    object_img = cv2.resize(object_img, (w, h))
    
    # 随机选择遮挡区域的位置
    x1 = np.random.randint(0, h - object_img.shape[0])
    y1 = np.random.randint(0, w - object_img.shape[1])
    
    # 将物体图片覆盖到原始图像上
    image[x1:x1 + object_img.shape[0], y1:y1 + object_img.shape[1], :] = object_img
    
    return image

from scipy.ndimage import gaussian_filter, map_coordinates
def elastic_transform(image, alpha=1, sigma=0.05):
    random_state = np.random.RandomState(None)
    shape = image.shape
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dz = np.zeros_like(dx)

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))

    return map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)

def apply_jpeg_compression(image, quality=50):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    result, encimg = cv2.imencode('.jpg', image, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    return decimg

def apply_resize(image, fx=0.5, fy=0.5):
    resized_image = cv2.resize(image, None, fx=fx, fy=fy, interpolation=cv2.INTER_LINEAR)
    return resized_image

def apply_downsample(image, factor=2):
    downsampled_image = image[::factor, ::factor]
    return downsampled_image

def add_fog(image, fog_intensity=0.5):
    fog_layer = np.full_like(image, 255, dtype=np.uint8)
    fogged_image = cv2.addWeighted(image, 1 - fog_intensity, fog_layer, fog_intensity, 0)
    return fogged_image

def get_noise(img, value=10):
    noise = np.random.uniform(0, 256, img.shape[0:2])
    v = value * 0.01
    noise[np.where(noise < (256 - v))] = 0

    k = np.array([[0, 0.1, 0],
                  [0.1, 8, 0.1],
                  [0, 0.1, 0]])

    noise = cv2.filter2D(noise, -1, k)
    return noise

def rain_blur(noise, length=10, angle=0, w=1):
    trans = cv2.getRotationMatrix2D((length / 2, length / 2), angle - 45, 1 - length / 100.0)
    dig = np.diag(np.ones(length))
    k = cv2.warpAffine(dig, trans, (length, length))
    k = cv2.GaussianBlur(k, (w, w), 0)

    blurred = cv2.filter2D(noise, -1, k)

    cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
    blurred = np.array(blurred, dtype=np.uint8)
    return blurred

def add_rain(image, rain_intensity=10, rain_length=10, rain_angle=0, rain_width=1, beta=0.8):
    noise = get_noise(image, value=rain_intensity)
    rain = rain_blur(noise, length=rain_length, angle=rain_angle, w=rain_width)
    
    rain = np.expand_dims(rain, 2)
    rain_effect = np.concatenate((image, rain), axis=2)

    rain_result = image.copy()
    rain = np.array(rain, dtype=np.float32)
    rain_result[:, :, 0] = rain_result[:, :, 0] * (255 - rain[:, :, 0]) / 255.0 + beta * rain[:, :, 0]
    rain_result[:, :, 1] = rain_result[:, :, 1] * (255 - rain[:, :, 0]) / 255 + beta * rain[:, :, 0]
    rain_result[:, :, 2] = rain_result[:, :, 2] * (255 - rain[:, :, 0]) / 255 + beta * rain[:, :, 0]
    
    return rain_result

def grid_distortion(image, num_steps=10, distort_limit=0.3):
    h, w = image.shape[:2]
    step = w // num_steps

    distortions = np.random.uniform(-distort_limit, distort_limit, size=(num_steps + 1, w))
    map_x, map_y = np.meshgrid(np.arange(w), np.arange(h))

    for i in range(1, num_steps):
        map_x[:, i * step:(i + 1) * step] = np.linspace(i * step + distortions[i, :step], (i + 1) * step + distortions[i + 1, :step], step)
        map_y[i * step:(i + 1) * step, :] = np.linspace(i * step + distortions[:step, i], (i + 1) * step + distortions[:-step, i + 1], step)

    return cv2.remap(image, map_x.astype(np.float32), map_y.astype(np.float32), interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

def add_snow(image, snow_intensity=0.5):
    snow = np.random.normal(size=image.shape[:2], loc=200, scale=20)
    snow = cv2.GaussianBlur(snow, (7, 7), 0)
    snow = snow[:, :, np.newaxis]
    snow = np.repeat(snow, 3, axis=2)
    snow_image = cv2.addWeighted(image, 1 - snow_intensity, snow.astype(np.uint8), snow_intensity, 0)
    return snow_image

def degrade_brightness(image, factor=0.5):
    """
    降低图像的亮度。

    :param image: 输入图像。
    :param factor: 降低亮度的因子。取值范围为[0, 1]，默认为0.5。
    :return: 降低亮度后的图像。
    """
    degraded_image = cv2.convertScaleAbs(image, alpha=factor, beta=0)
    return degraded_image

def degrade_contrast(image, alpha=0.5, beta=0):
    """
    降低图像的对比度。

    :param image: 输入图像。
    :param alpha: 对比度调整因子。大于1提高对比度，小于1降低对比度。默认为0.5。
    :param beta: 亮度调整因子，默认为0。
    :return: 降低对比度后的图像。
    """
    degraded_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return degraded_image

def degrade_color_distortion(image, sigma_s=10, sigma_r=0.1):
    """
    引入颜色失真。

    :param image: 输入图像。
    :param sigma_s: 空间高斯标准差。默认为10。
    :param sigma_r: 色彩相似性高斯标准差。默认为0.1。
    :return: 失真后的图像。
    """
    degraded_image = cv2.bilateralFilter(image, -1, sigma_s, sigma_r)
    return degraded_image

def process_image(image_path, output_path, degrade_functions, params):
    image = cv2.imread(image_path)
    for func, param in zip(degrade_functions, params):
        image = func(image, **param)
    cv2.imwrite(output_path, image)

def batch_process_images(input_dir, output_dir, degrade_functions, params):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for filename in os.listdir(input_dir):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            process_image(input_path, output_path, degrade_functions, params)
example_image='/data/jinjiandong/datasets/MSP/images/part17_20231212112816_00313_009.jpg'

# # Example usage:
# # Single image processing
# degrade_funcs = [add_gaussian_noise, apply_motion_blur, apply_brightness_change]
# params = [{'mean': 0, 'std': 25}, {'size': 15}, {'factor': 0.5}]
# process_image('input.jpg', 'output.jpg', degrade_funcs, params)

# Batch processing
# batch_process_images('input_images', 'output_images', degrade_funcs, params)

# List of degradation functions and their parameter sets
degradation_tests = [
    (add_gaussian_noise, [{'mean': 3, 'std': 3}]),
    (add_salt_and_pepper_noise, [{'salt_prob': 0.01, 'pepper_prob': 0.01}]),
    (apply_motion_blur, [{'size': 13}]),
    (apply_gaussian_blur, [{'kernel_size': 5, 'sigma': 2}]),
    (apply_brightness_change, [{'factor': 0.5}]),
    (apply_contrast_change, [{'alpha': 2.5, 'beta': 0}]),
    (apply_random_erasing, [{'sl': 0.02, 'sh': 0.1, 'r1': 0.3}]),
    (apply_jpeg_compression, [{'quality': 20}]),
    (add_fog, [{'fog_intensity': 0.6}]),
    (apply_median_blur, [{'kernel_size': 5}]),
    (elastic_transform, [{'alpha': 1.5, 'sigma': 0.2}]),
    (add_snow, [{'snow_intensity': 0.5}]),
    (add_random_occlusion, [{}])
]


def test_degradation_functions(image_path, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    original_image = cv2.imread(image_path)
    cv2.imwrite(os.path.join(output_dir, 'original.jpg'), original_image)
    for func, param_list in degradation_tests:
        func_name = func.__name__
        for i, params in enumerate(param_list):
            image = original_image.copy()  # Reset to original image for each test
            degraded_image = func(image, **params)
            param_str = '_'.join(f'{k}{v}' for k, v in params.items())
            output_path = os.path.join(output_dir, f'{func_name}_{param_str}.jpg')
            cv2.imwrite(output_path, degraded_image)
            print(f'Saved {output_path}')
# # Example usage
# test_degradation_functions(example_image, '/data/jinjiandong/datasets/MSP/degrades_test')
incompatible_degradations = {
    'apply_contrast_change': ['add_fog', 'apply_brightness_change'],
    'add_fog': ['apply_contrast_change'],
    'apply_brightness_change':['apply_brightness_change'],
    'apply_motion_blur':['apply_gaussian_blur', 'apply_median_blur'],
    'apply_gaussian_blur':['apply_motion_blur', 'apply_median_blur'],
    'apply_median_blur':['apply_gaussian_blur', 'apply_motion_blur'],
    'add_gaussian_noise':['add_salt_and_pepper_noise'],
    'add_salt_and_pepper_noise':['add_gaussian_noise'],
    'add_random_occlusion':['apply_random_erasing'],
    'apply_random_erasing':['add_random_occlusion'],
}

def check_and_replace_degradations(chosen_degradations):
    """
    Check chosen degradations for incompatibilities and replace if necessary.
    """
    chosen_degradations_filtered = []

    for func, param_list in chosen_degradations:
        chosen_degradations_filtered.append((func, param_list)) 

    for idx, (func, param_list) in enumerate(chosen_degradations_filtered):
        if func.__name__ in incompatible_degradations.keys():
            incompatible = incompatible_degradations[func.__name__]
            if any(f.__name__ in incompatible for f, _ in chosen_degradations_filtered):
                del chosen_degradations_filtered[idx]
                    
                new_func, new_params = random.choice(degradation_tests)
                while new_func.__name__ in incompatible or any(new_func.__name__ == f.__name__ for f, _ in chosen_degradations_filtered) or any(f.__name__  in incompatible_degradations[new_func.__name__]  for f, _ in chosen_degradations_filtered if new_func.__name__ in incompatible_degradations.keys()):
                    new_func, new_params = random.choice(degradation_tests)
                chosen_degradations_filtered.append((new_func, new_params))
        
    return chosen_degradations_filtered

from tqdm import tqdm
def test_random_degradations_batch(floder, image_paths, random_degradation_count, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    images_degrades={}
    for image_path in tqdm(image_paths):
        original_image = cv2.imread(os.path.join(floder, image_path))
        image = original_image.copy()

        chosen_degradations = random.sample(degradation_tests, random_degradation_count)
        chosen_degradations_filtered = check_and_replace_degradations(chosen_degradations)    
        degradation_details = []
        for func, param_list in chosen_degradations_filtered:
            params = random.choice(param_list)
            image = func(image, **params)
            degradation_details.append((func.__name__, params))
        filename = os.path.basename(image_path)
        name, ext = os.path.splitext(filename)
        output_path = os.path.join(output_dir, f'{name}.jpg')
        cv2.imwrite(output_path, image)
        images_degrades.update({image_path:degradation_details})
        # print(f'Saved {output_path} with degradations: {degradation_details}')
    return images_degrades

import json
with open('/data/jinjiandong/datasets/MSP/514split/dataset_split.json', 'r') as f:
    dataset = json.load(f)
import random
with open('/data/jinjiandong/datasets/MSP/dataset514_withdegrade.json', 'r') as f:
    degrade_dataset = json.load(f)
# # 从 dataset['train'] 中随机选择，排除包含 'part4' 的图片
# train_degrade_imgs_name = [name for name in dataset['train'].keys() if 'part4' not in name]
# train_degrade_imgs_name = random.choices(train_degrade_imgs_name, k=int(len(dataset['train'].keys())/2))

# # 从 dataset['val'] 中随机选择，排除包含 'part4' 的图片
# val_degrade_imgs_name = [name for name in dataset['val'].keys() if 'part4' not in name]
# val_degrade_imgs_name = random.choices(val_degrade_imgs_name, k=int(len(dataset['val'].keys())/2))

# # 从 dataset['test'] 中随机选择，排除包含 'part4' 的图片
# test_degrade_imgs_name = [name for name in dataset['test'].keys() if 'part4' not in name]
# test_degrade_imgs_name = random.choices(test_degrade_imgs_name, k=int(len(dataset['test'].keys())/2))

# degrade_imgs_name = train_degrade_imgs_name+val_degrade_imgs_name+test_degrade_imgs_name
degrade_imgs_name = degrade_dataset['degrade_imgs']
dataset.update({"degrade_imgs": degrade_imgs_name})
images_degrades = test_random_degradations_batch('./images', degrade_imgs_name,
                               random_degradation_count=3, output_dir='./degrades_images')

# dataset.update({"degrade_details": images_degrades})
# with open('dataset514_withdegrade.json', 'w', encoding='utf-8') as file:
#     json.dump(dataset, file, ensure_ascii=False, indent=4)
    
# import os
# import shutil

# # 创建目标文件夹
# target_folder = '/data/jinjiandong/datasets/MSP/unselected_images'
# os.makedirs(target_folder, exist_ok=True)

# # 获取所有图片文件名
# all_train_imgs = list(dataset['train'].keys())
# all_val_imgs = list(dataset['val'].keys())
# all_test_imgs = list(dataset['test'].keys())

# # 移动未被选择的训练集图片
# for img_name in all_train_imgs:
#     if img_name not in degrade_imgs_name:
#         src_path = os.path.join('/data/jinjiandong/datasets/MSP/images', img_name)  # 原始训练集图片路径
#         dest_path = os.path.join(target_folder, img_name)
#         shutil.copy2(src_path, dest_path)

# # 移动未被选择的验证集图片
# for img_name in all_val_imgs:
#     if img_name not in degrade_imgs_name :
#         src_path = os.path.join('/data/jinjiandong/datasets/MSP/images', img_name)  # 原始验证集图片路径
#         dest_path = os.path.join(target_folder, img_name)
#         shutil.copy2(src_path, dest_path)

# # 移动未被选择的测试集图片
# for img_name in all_test_imgs:
#     if img_name not in degrade_imgs_name:
#         src_path = os.path.join('/data/jinjiandong/datasets/MSP/images', img_name)  # 原始测试集图片路径
#         dest_path = os.path.join(target_folder, img_name)
#         shutil.copy2(src_path, dest_path)
