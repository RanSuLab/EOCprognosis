from scipy.misc import imsave
import openslide
import numpy as np
import os
import random
import shutil
import xlwt
import datetime
import pandas as pd
from PIL import ImageStat

# ovarian_cancer get_patches  level_cout=2,3,level[0] 20×magnification, level count=4, level[1] 20×magnification, level_count=1,out

#write svs path to csv
def write_path_to_csv(path):
    path_list = []
    number = 0
    for first_path_name in os.listdir(path):
        first_path = path+first_path_name
        for second_path_name in os.listdir(first_path):
            if second_path_name.endswith('svs'):
                svs_path = first_path+'/'+second_path_name
                path_list.append(svs_path)
                number+=1
    pd_path_list = pd.DataFrame(path_list)
    pd_path_list.to_csv('./svs_path.csv',header=None,index=None)
    print('Finish ',number,' pathes!')

# write_path_to_csv('J:/Dling/TCGA/TCGA_OV/tcga_ov_data/')

def judge_save(region,new_path,svs_name,patch_size,digit_num_all,digit_num_i,a,threshold):


    image_rgb = region.convert('RGB')
    image_gray = region.convert('L')
    image = np.array(image_gray)

    image_right = np.array(image_rgb)
    image_right = image_right + 0
    image_right = image_right.transpose()
    image_right_0 = image_right[0]
    image_right_1 = image_right[1]
    image_right_2 = image_right[2]
    mean_0 = np.mean(image_right_0)
    mean_1 = np.mean(image_right_1)
    mean_2 = np.mean(image_right_2)
    max_value = max(mean_0, mean_1, mean_2)
    min_value = min(mean_0, mean_1, mean_2)

    max_value_2 = np.max(image)
    min_value_2 = np.min(image)


    # pixel with a value larger than 200 belongs to background area
    count = int(np.sum(image<200))
    
    if (count / (patch_size * patch_size)) > threshold:
        image_rgb.save(new_path + svs_name + '_' + '0' * (digit_num_all - digit_num_i) + str(a + 1) + '.tif')
        return True
    else:
        return False


def cal_patches(level,slide,patch_size,patch_path,svs_name,patches_num,name,w,h,digit_num_all):
    finished_num = 0
    new_path = patch_path + name+ '/' + svs_name + '/'
    if not os.path.exists(new_path):
        os.makedirs(new_path)
    threshold_list = [0.5]
    for threshold in threshold_list:
        i = 0
        time_start = datetime.datetime.now()
        while i < patches_num:
            digit_num_i = len(str(i + 1))
            x_point = random.randint(0, w - patch_size)
            y_point = random.randint(0, h - patch_size)
            # remember read_region((width,height))
            region = slide.read_region((x_point,y_point), level, (patch_size, patch_size))
            if judge_save(region, new_path, svs_name,patch_size,digit_num_all, digit_num_i, i,threshold):
                i+=1
            time_end = datetime.datetime.now()
            time_use = int((time_end-time_start).seconds)
            if time_use>300 and i==0:
                finished_num = i
                break
        if i>=patches_num:
            finished_num = i
            break
    return finished_num

def deal_patches(slide,svs_path,patch_size,sample_number,patch_path,svs_name,unfinished,level):
    [w, h] = slide.level_dimensions[0]
    patches_num = int((w / patch_size) * (h / patch_size) * 0.05)
    digit_num_all = len(str(patches_num))
    if sample_number < 10:
        f_num = cal_patches(level, slide, patch_size, patch_path, svs_name, patches_num, 'cancer', w, h, digit_num_all)
        if f_num < patches_num:
            unfinished.append(svs_path)
        return f_num, patches_num


    else:
        f_num = cal_patches(level, slide, patch_size, patch_path, svs_name, patches_num, 'contrast', w, h,
                            digit_num_all)
        if f_num < patches_num:
            unfinished.append(svs_path)
        return f_num, patches_num
    # if sample_number > 9:
    #     cal_patches(level, slide, patch_size, patch_path, svs_name, patches_num, 'contrast', w, h, digit_num_all)


def get_patches(csv_path,patch_path,patch_size):
    svs_pathes = np.array(pd.read_csv(csv_path,header=None))
    finished_patches = 0
    need_patches = 0
    unfinished = []
    for i in range(len(svs_pathes)):
    # for i in range(x, x+1):

        start_time = datetime.datetime.now()
        svs_path = str(svs_pathes[i][0])
        svs_name = (svs_path.split('/')[-1]).split('.')[0]
        sample_number = int((svs_name.split('-')[3])[:2])
        slide = openslide.open_slide(svs_path)
        print('Deal with ',str(i+1),'st images named ',svs_name, end=' ')
        if slide.level_count==1:
            print('')
        elif slide.level_count == 4:
            level = 1
            f_num, patches_num = deal_patches(slide, svs_path, patch_size, sample_number, patch_path, svs_name, unfinished, level)
            finished_patches = f_num
            need_patches = patches_num

        else:
            level = 0
            f_num, patches_num = deal_patches(slide, svs_path, patch_size, sample_number, patch_path, svs_name, unfinished, level)
            finished_patches = f_num
            need_patches = patches_num

        end_time = datetime.datetime.now()
        time = str((end_time - start_time).seconds)
        print('with '+time+' seconds, '+'finished '+str(finished_patches)+'/'+str(need_patches)+' in '+str(datetime.datetime.now()).split('.')[0])
    unfininshed_list = pd.DataFrame(unfinished)
    unfininshed_list.to_csv('./unfininshed.csv', header=None, index=None)
    print('Finish!')


get_patches('./new_svs_pathes.csv','J:/Dling/Ovarian Cancer/patches_3/',512)
