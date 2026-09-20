attributes=[
    "Female",
    "Child", "Adult", "Elderly",
    "Fat", "Normal", "Thin",
    "Bald", "Long hair", "Black hair", "Hat", "Glasses", "Mask", "Helmet", "Scarf", "Gloves",
    "Front", "Back", "Side",
    "Short sleeves", "Long sleeves", "Shirt", "Jacket", "Suit",  "Vest", "Cotton-padded coat", "Coat", "Graduation gown", "Chef uniform",
    "Trousers", "Shorts", "Jeans", "Long skirt", "Short skirt", "Dress", 
    "Leather shoes", "Casual shoes", "Boots", "Sandals", "Other shoes",
    "Backpack", "Shoulder bag", "Handbag", "Plastic bag", "Paper bag", "Suitcase", "Others", 
    "Making a phone call", "Smoking", "Hands behind back", "Arms crossed",
    "Walking", "Running", "Standing", "Riding a bicycle", "Riding an scooter", "Riding a skateboard"
]
ori2new={
'leqing':	    'part0_',
'part1':	    'part1_',
'youtian':      'part2_',
'sucai'	:       'part4_',
'nongmao':	    'part5_',
'huashu':       'part6_',
'nannong':	    'part7_',
'part3':	    'part8_',
'zaxiang':	    'part9_',
'hs':	        'part10_',
'nfy':	        'part11_',
'hys':	        'part12_',
'mclz':         'part13_',
'AHU_XSF':	    'part14_',
'CleanAHU':	    'part15_',
'Ski':	        'part16_',
'Agdao':	    'part17_'
}
# sence_group={
#     "Indoors China":["leqing","nongmao","huashu",'hs'],
#     "Outdoors China":["CleanAHU",'zaxiang','D', 'youtian'],
#     "Kitchens China":['nannong','part3','nfy','mclz'],
#     "Mixed China":['part1'],
#     "Ski":["Ski"],
#     'Outdoors Agdao': ['Agdao'],
#     "Sucai":["sucai"],
# }
sence_group={
    "Constuction Site":["part1","youtian"],
    "Market":["leqing",'nongmao','huashu'],
    "Kitchens":['nannong','part3','hs','nfy','hys','mclz'],
    "Ski Resort":['Ski'],
    "School":["sucai"],
    'Outdoors1': ['CleanAHU',"AHU_XSF"],
    'Outdoors2': ['zaxiang'],
    'Outdoors3': ['Agdao'],
}

sence_split={
    "split_1":{'trainval': ["Constuction Site", "Market", "Kitchens", "School", "Ski Resort"],'test':["Outdoors1", 'Outdoors2',"Outdoors3"],"drop":[]},
    "split_2":{'trainval': ["Market","Outdoors1", 'Outdoors2',"Outdoors3"],'test':["Constuction Site","Kitchens", "School", "Ski Resort"],"drop":[]},
}

every_part_count={
              'part0_': 1076, 
              'part1_': 4110, 
              'part2_': 293, 
              'part8_': 145, 
              'part4_': 5896, 
              'part5_': 15613,
              'part6_': 695, 
              'part7_': 1181, 
              'part9_': 4657,
              'part10_': 67,
              'part11_': 57, 
              'part12_': 107,
              'part13_': 108, 
              'part14_': 383,
              'part15_': 8405, 
              'part16_': 5780,
              'part17_': 11549
              }
import json
import os 

# sence_image={name:[] for name in sence_group.keys()}
with open('./merged_labels.json', 'r', encoding='utf-8') as f:
    split_data = json.load(f)
with open('./Multi_Sences2.json', 'r', encoding='utf-8') as f:
    sence_image = json.load(f)['image_names']
# sence_group_count={'Constuction Site': 8806, 'Market': 34768, 'Kitchens': 3330, 'Ski Resort': 11560, 'School': 11792, 'Outdoors1': 17576, 'Outdoors2': 9314, 'Outdoors3': 23098}
# sence_split_count={'split_1': {'trainval': 35128, 'test': 24994}, 'split_2': {'trainval': 42378, 'test': 17744}}
sence_group_count ={'Constuction Site': 4403, 'Market': 17384, 'Kitchens': 1665, 'Ski Resort': 5780, 'School': 5896, 'Outdoors1': 8788, 'Outdoors2': 4657, 'Outdoors3': 11549}
sence_split_count = {'split_1': {'trainval': 35128, 'test': 24994}, 'split_2': {'trainval': 42378, 'test': 17744}}
sence_attr_dist={sence:{attr:0 for attr in attributes} for sence in sence_group.keys()}
# for sence, part_list in sence_group.items():
#     cur_sence_images = sence_image[sence]
#     for img in cur_sence_images:
#         img_label = split_data[img]
#         for aidx, label  in enumerate(img_label):
#             if label:
#                 sence_attr_dist[sence][attributes[aidx]] +=1
# print(sence_attr_dist)
labels={'Female': 24077, 'Child': 6767, 'Adult': 51513, 'Elderly': 1845, 'Fat': 1314, 'Normal': 58594, 'Thin': 202, 'Bald': 853, 'Long hair': 10143, 'Black hair': 46880, 'Hat': 12645, 'Glasses': 4057, 'Mask': 13909, 'Helmet': 1879, 'Scarf': 422, 'Gloves': 2312, 'Front': 23160, 'Back': 22697, 'Side': 14274, 'Short sleeves': 20697, 'Long sleeves': 19570, 'Shirt': 19916, 'Jacket': 5178, 'Suit': 218, 'Vest': 1587, 'Cotton-padded coat': 5631, 'Coat': 188, 'Graduation gown': 353, 'Chef uniform': 1920, 'Trousers': 36921, 'Shorts': 8549, 'Jeans': 1943, 'Long skirt': 972, 'Short skirt': 772, 'Dress': 3249, 'Leather shoes': 958, 'Casual shoes': 17623, 'Boots': 2621, 'Sandals': 9343, 'Other shoes': 124, 'Backpack': 2324, 'Shoulder bag': 4635, 'Handbag': 1356, 'Plastic bag': 4134, 'Paper bag': 115, 'Suitcase': 76, 'Others': 185, 'Making a phone call': 751, 'Smoking': 186, 'Hands behind back': 832, 'Arms crossed': 994, 'Walking': 25656, 'Running': 146, 'Standing': 25213, 'Riding a bicycle': 633, 'Riding an scooter': 1374, 'Riding a skateboard': 13}
for k, v in labels.items():
    if v<600:
        print(k,v)
    # for part in part_list:
    #     real_part = ori2new[part]
        

# for split in sence_split:
#     for sence in sence_split[split]['trainval']:
#         for part in sence_group[sence]:
#             real_part = ori2new[part]
#             sence_group_count[sence]+=every_part_count[real_part]
#             sence_split_count[split]['trainval']+=every_part_count[real_part]
            
#     for sence in sence_split[split]['test']:
#         for part in sence_group[sence]:
#             real_part = ori2new[part]
#             sence_group_count[sence]+=every_part_count[real_part]
#             sence_split_count[split]['test']+=every_part_count[real_part]
# for k,v in sence_group_count.items():
#     sence_group_count[k] = v//2
# print(sence_group_count)
# print(sence_split_count)

# for k in sence_group.keys():
#     for sence in sence_group[k]:
#         realname=ori2new[sence]
#         for img in os.listdir('./images'):
#             if realname in img:
#                 sence_image[k].append(img)
# sences={
#     "every_part_count": every_part_count,
#     "sence_group":sence_group,
#     "sence_group_count":sence_group_count,
#     "split":sence_split,
#     "split_count":sence_split_count,
#     "image_names":sence_image,
# }                
# output_file_path = './Multi_Sences.json'
# with open(output_file_path, 'w', encoding='utf-8') as output_file:
#     json.dump(sences, output_file, ensure_ascii=False, indent=4)
    



# attributes_count={'Female': 24077, 'Child': 6767, 'Adult': 51513, 'Elderly': 1845, 'Fat': 1314, 'Normal': 58594, 'Thin': 202, 'Bald': 853, 'Long hair': 10143, 'Black hair': 46880, 'Hat': 12645, 'Glasses': 4057, 'Mask': 13909, 'Helmet': 1879, 'Scarf': 422, 'Gloves': 2312, 'Front': 23160, 'Back': 22697, 'Side': 14274, 'Short sleeves': 20697, 'Long sleeves': 19570, 'Shirt': 19916, 'Jacket': 5178, 'Suit': 218, 'Vest': 1587, 'Cotton-padded coat': 5631, 'Coat': 188, 'Graduation gown': 353, 'Chef uniform': 1920, 'Trousers': 36921, 'Shorts': 8549, 'Jeans': 1943, 'Long skirt': 972, 'Short skirt': 772, 'Dress': 3249, 'Leather shoes': 958, 'Casual shoes': 17623, 'Boots': 2621, 'Sandals': 9343, 'Other shoes': 124, 'Backpack': 2324, 'Shoulder bag': 4635, 'Handbag': 1356, 'Plastic bag': 4134, 'Paper bag': 115, 'Suitcase': 76, 'Others': 185, 'Making a phone call': 751, 'Smoking': 186, 'Hands behind back': 832, 'Arms crossed': 994, 'Walking': 25656, 'Running': 146, 'Standing': 25213, 'Riding a bicycle': 633, 'Riding an scooter': 1374, 'Riding a skateboard': 13}
# pedes_count={3: 10, 4: 350, 5: 1440, 6: 3158, 7: 7431, 8: 15061, 9: 14108, 10: 10323, 11: 5479, 12: 2092, 13: 556, 14: 101, 15: 9, 16: 4}

# import json
# import numpy as np
# import matplotlib.pyplot as plt

# num_attributes = len(attributes)
# co_occurrence_matrix = np.zeros((num_attributes, num_attributes))
# with open('./merged_labels.json', 'r', encoding='utf-8') as f:
#     split_data = json.load(f)
# # 计算共现矩阵
# for labels in split_data.values():
#     labels = np.array(labels)
#     for i in range(num_attributes):
#         if labels[i] == 1:
#             for j in range(num_attributes):
#                 if labels[j] == 1:
#                     co_occurrence_matrix[i, j] += 1
# # 对共现矩阵取对数
# log_co_occurrence_matrix = np.log(co_occurrence_matrix)

# # 将 -inf 替换为 0
# log_co_occurrence_matrix = np.nan_to_num(log_co_occurrence_matrix, neginf=0)

# # 可视化共现矩阵
# plt.figure(figsize=(10, 10))
# plt.imshow(log_co_occurrence_matrix, cmap='Blues', interpolation='nearest')
# plt.colorbar()
# plt.xticks(range(num_attributes), attributes, rotation=90)
# plt.yticks(range(num_attributes), attributes)
# plt.title('Log Co-occurrence Matrix of Attributes')
# plt.show()