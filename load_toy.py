import numpy as np
import cv2
import os
import json
from tqdm import tqdm
from scipy.spatial.transform import Rotation


def getTrans(positions, quaternions):
    Trans_list = []
    for pos, q in zip(positions, quaternions):
        Trans = np.identity(4)
        rot = Rotation.from_quat(q).as_matrix()
        Trans[:3, :3] = rot
        Trans[:3, 3] = pos
        Trans_list.append(Trans)
    return Trans_list


def save_metadata(scene_path):
    meta_path = os.path.join(scene_path, 'metadata.json')
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    
    # FXIME
    object_classes = meta['segmentation_labels']
    
    cam = meta['camera']
    fov = cam['field_of_view']
    intrin = cam['K']
    extrin = cam['R'] #N*4*4
    trans_mats = getTrans(cam['positions'], cam['quaternions'])
    
    train_list = sorted(meta['split_ids']['train'])
    test_list = sorted(meta['split_ids']['test'])
    
    train_transfile = {}
    train_transfile['camera_angle_x'] = fov
    train_transfile['K'] = intrin
    train_transfile['frames'] = []
    
    test_transfile = {}
    test_transfile['camera_angle_x'] = fov
    test_transfile['K'] = intrin
    test_transfile['frames'] = []
    
    for idx in train_list:
        curr_frame = {}
        curr_frame['file_path'] = f"./train/rgb_{idx}"
        curr_frame['label_path'] = f"./train/label_{idx}"
        curr_frame['transform_matrix'] = trans_mats[idx].tolist()
        train_transfile['frames'].append(curr_frame)
       
    for idx in test_list:
        curr_frame = {}
        curr_frame['file_path'] = f"./test/rgb_{idx}"
        curr_frame['label_path'] = f"./test/label_{idx}"
        curr_frame['transform_matrix'] = trans_mats[idx].tolist()
        test_transfile['frames'].append(curr_frame)
     
    return train_transfile, test_transfile, train_list, test_list, object_classes


def generate_label_img(input_path, output_path, idx_list, all_gray_vals):
    for img_idx in idx_list:
        rgb_img = cv2.imread(os.path.join(input_path, 
                                          f'rgba_{str(img_idx).zfill(5)}.png'))
        seg_img = cv2.imread(os.path.join(input_path, 
                                          f'segmentation_{str(img_idx).zfill(5)}.png'))
        gray_img = cv2.cvtColor(seg_img, cv2.COLOR_RGB2GRAY)
        
        unique_gray = np.unique(gray_img) # sorted in acending order
        # update the list with all possible gray values
        all_gray_vals = np.unique(np.concatenate((all_gray_vals, unique_gray)))
        
        new_seg_img = np.zeros(gray_img.shape)
        for i, gray_val in enumerate(unique_gray):
            if i==0:
                continue
            new_seg_img[gray_img == gray_val] = i

        cv2.imwrite(os.path.join(output_path, f'rgb_{img_idx}.png'), rgb_img)
        cv2.imwrite(os.path.join(output_path, f'label_{img_idx}.png'), new_seg_img)

    return all_gray_vals


def preprocessing():
    data_dir = "./dataset/klevr"
    scene_list = sorted(os.listdir(data_dir), key=lambda i : int(i))
    
    out_dir = os.path.join(data_dir, '../klevr_processed')
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    
    # Store all possible gray values in a list
    all_gray_vals = np.zeros(1)
    
    # TODO: check the number of objects in different scenes
    object_list = []
    
    for idx, scene in tqdm(enumerate(scene_list), total=len(scene_list)):
        # FIXME:
        if idx>30-1:
            break
        
        # create new directories to save the preprocessed one
        new_scene_path = os.path.join(out_dir, scene) # with scene idx
        if not os.path.exists(new_scene_path):
            print(f"create directory: {new_scene_path}")
            os.mkdir(new_scene_path)
            
        seg_train_path = os.path.join(new_scene_path, 'train')
        if not os.path.exists(seg_train_path):
            print(f"create directory: {seg_train_path}")
            os.mkdir(seg_train_path)
        seg_test_path = os.path.join(new_scene_path, 'test')
        if not os.path.exists(seg_test_path):
            print(f"create directory: {seg_test_path}")
            os.mkdir(seg_test_path)
        
        scene_path = os.path.join(data_dir, scene)
        # save the transforms json
        train_transfiles, test_transfiles, train_idx, test_idx, object_labels = save_metadata(scene_path)
        with open(os.path.join(new_scene_path, 'transforms_train.json'), 'w') as file:
            json.dump(train_transfiles, file, indent=4)
            
        with open(os.path.join(new_scene_path, 'transforms_test.json'), 'w') as file:
            json.dump(test_transfiles, file, indent=4)
        
        local_gray_vals = np.zeros(0)
        local_gray_vals = generate_label_img(scene_path, seg_train_path, train_idx, local_gray_vals)
        local_gray_vals = generate_label_img(scene_path, seg_test_path, test_idx, local_gray_vals)
        all_gray_vals = np.unique(np.concatenate((all_gray_vals, local_gray_vals)))

        # TODO: added to check the uniqueness of gray color
        if len(object_labels)<len(local_gray_vals)-1:
            print(f"In scene {scene}, The number of gray colors are greater than the objec classes!")
        curr_label={}
        curr_label['scene'] = scene
        curr_label['label'] = object_labels
        curr_label['gray_vals'] = local_gray_vals.tolist()
        object_list.append(curr_label)

    # TODO
    with open(os.path.join(out_dir, 'label_list.json'), 'w') as file:
        json.dump(object_list, file, indent=4)

    print(f"There are {len(all_gray_vals)} possible gray values, {all_gray_vals}")


if __name__ == "__main__":
    preprocessing()
