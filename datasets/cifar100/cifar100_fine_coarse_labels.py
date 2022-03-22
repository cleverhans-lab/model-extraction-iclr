import pprint as pp

fine_id_coarse_id = {
    0: 4,
    1: 1,
    2: 14,
    3: 8,
    4: 0,
    5: 6,
    6: 7,
    7: 7,
    8: 18,
    9: 3,
    10: 3,
    11: 14,
    12: 9,
    13: 18,
    14: 7,
    15: 11,
    16: 3,
    17: 9,
    18: 7,
    19: 11,
    20: 6,
    21: 11,
    22: 5,
    23: 10,
    24: 7,
    25: 6,
    26: 13,
    27: 15,
    28: 3,
    29: 15,
    30: 0,
    31: 11,
    32: 1,
    33: 10,
    34: 12,
    35: 14,
    36: 16,
    37: 9,
    38: 11,
    39: 5,
    40: 5,
    41: 19,
    42: 8,
    43: 8,
    44: 15,
    45: 13,
    46: 14,
    47: 17,
    48: 18,
    49: 10,
    50: 16,
    51: 4,
    52: 17,
    53: 4,
    54: 2,
    55: 0,
    56: 17,
    57: 4,
    58: 18,
    59: 17,
    60: 10,
    61: 3,
    62: 2,
    63: 12,
    64: 12,
    65: 16,
    66: 12,
    67: 1,
    68: 9,
    69: 19,
    70: 2,
    71: 10,
    72: 0,
    73: 1,
    74: 16,
    75: 12,
    76: 9,
    77: 13,
    78: 15,
    79: 13,
    80: 16,
    81: 19,
    82: 2,
    83: 4,
    84: 6,
    85: 19,
    86: 5,
    87: 5,
    88: 8,
    89: 19,
    90: 18,
    91: 1,
    92: 2,
    93: 15,
    94: 6,
    95: 0,
    96: 17,
    97: 8,
    98: 14,
    99: 13,
}

fine_labels = [
    'apple',  # id 0
    'aquarium_fish',
    'baby',
    'bear',
    'beaver',
    'bed',
    'bee',
    'beetle',
    'bicycle',
    'bottle',
    'bowl',
    'boy',
    'bridge',
    'bus',
    'butterfly',
    'camel',
    'can',
    'castle',
    'caterpillar',
    'cattle',
    'chair',
    'chimpanzee',
    'clock',
    'cloud',
    'cockroach',
    'couch',
    'crab',
    'crocodile',
    'cup',
    'dinosaur',
    'dolphin',
    'elephant',
    'flatfish',
    'forest',
    'fox',
    'girl',
    'hamster',
    'house',
    'kangaroo',
    'computer_keyboard',
    'lamp',
    'lawn_mower',
    'leopard',
    'lion',
    'lizard',
    'lobster',
    'man',
    'maple_tree',
    'motorcycle',
    'mountain',
    'mouse',
    'mushroom',
    'oak_tree',
    'orange',
    'orchid',
    'otter',
    'palm_tree',
    'pear',
    'pickup_truck',
    'pine_tree',
    'plain',
    'plate',
    'poppy',
    'porcupine',
    'possum',
    'rabbit',
    'raccoon',
    'ray',
    'road',
    'rocket',
    'rose',
    'sea',
    'seal',
    'shark',
    'shrew',
    'skunk',
    'skyscraper',
    'snail',
    'snake',
    'spider',
    'squirrel',
    'streetcar',
    'sunflower',
    'sweet_pepper',
    'table',
    'tank',
    'telephone',
    'television',
    'tiger',
    'tractor',
    'train',
    'trout',
    'tulip',
    'turtle',
    'wardrobe',
    'whale',
    'willow_tree',
    'wolf',
    'woman',
    'worm',
]

mapping_coarse_fine = {
    'aquatic mammals': ['beaver', 'dolphin', 'otter', 'seal', 'whale'],
    'fish': ['aquarium_fish', 'flatfish', 'ray', 'shark', 'trout'],
    'flowers': ['orchid', 'poppy', 'rose', 'sunflower', 'tulip'],
    'food containers': ['bottle', 'bowl', 'can', 'cup', 'plate'],
    'fruit and vegetables': ['apple', 'mushroom', 'orange', 'pear',
                             'sweet_pepper'],
    'household electrical device': ['clock', 'computer_keyboard', 'lamp',
                                    'telephone', 'television'],
    'household furniture': ['bed', 'chair', 'couch', 'table', 'wardrobe'],
    'insects': ['bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach'],
    'large carnivores': ['bear', 'leopard', 'lion', 'tiger', 'wolf'],
    'large man-made outdoor things': ['bridge', 'castle', 'house', 'road',
                                      'skyscraper'],
    'large natural outdoor scenes': ['cloud', 'forest', 'mountain', 'plain',
                                     'sea'],
    'large omnivores and herbivores': ['camel', 'cattle', 'chimpanzee',
                                       'elephant', 'kangaroo'],
    'medium-sized mammals': ['fox', 'porcupine', 'possum', 'raccoon', 'skunk'],
    'non-insect invertebrates': ['crab', 'lobster', 'snail', 'spider', 'worm'],
    'people': ['baby', 'boy', 'girl', 'man', 'woman'],
    'reptiles': ['crocodile', 'dinosaur', 'lizard', 'snake', 'turtle'],
    'small mammals': ['hamster', 'mouse', 'rabbit', 'shrew', 'squirrel'],
    'trees': ['maple_tree', 'oak_tree', 'palm_tree', 'pine_tree',
              'willow_tree'],
    'vehicles 1': ['bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train'],
    'vehicles 2': ['lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor'],
}


def print_fine_labels():
    for id, label in enumerate(fine_labels):
        print(id, " ", label)


def new_dicts():
    # fine label name -> id of fine label
    fine_id = dict()
    # id of fine label -> fine label name
    id_fine = dict()
    for id, label in enumerate(fine_labels):
        fine_id[label] = id
        id_fine[id] = label

    # coarse label name -> id of coarse label
    coarse_id = dict()
    # id of coarse label -> name of the coarse label
    id_coarse = dict()
    # name of fine label -> name of coarse label
    fine_coarse = dict()
    # id of fine label -> id of coarse label
    fine_id_coarse_id = dict()
    # id of coarse label -> id of fine label
    coarse_id_fine_id = dict()
    for id, (coarse, fines) in enumerate(mapping_coarse_fine.items()):
        coarse_id[coarse] = id
        id_coarse[id] = coarse
        fine_labels_ids = []
        for fine in fines:
            fine_coarse[fine] = coarse
            fine_label_id = fine_id[fine]
            fine_id_coarse_id[fine_label_id] = id
            fine_labels_ids.append(fine_label_id)
        coarse_id_fine_id[id] = fine_labels_ids

    dicts = ['fine_id', 'id_fine', 'coarse_id', 'id_coarse', 'fine_coarse',
             'fine_id_coarse_id', 'coarse_id_fine_id']
    for dic in dicts:
        dic_value = locals()[dic]
        print(dic + ' = ')
        pp.pprint(dic_value)


if __name__ == "__main__":
    print_fine_labels()
    pp.pprint(mapping_coarse_fine)
    new_dicts()
