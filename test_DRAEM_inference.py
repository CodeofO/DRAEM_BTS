import torch
import json
import base64
import natsort
import torch.nn.functional as F
from DRAEM_module.data_loader import MVTecDRAEMInferenceDataset
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score, precision_recall_curve, confusion_matrix
from DRAEM_module.model_unet import ReconstructiveSubNetwork, DiscriminativeSubNetwork
import os
import cv2
import shutil
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm
import pandas as pd

def normalization(img):
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    return img

def test(args):        
    
    checkpoint_path = args["model_path"]
    top_rate = args['top_rate']

    result_save_path = os.path.join(args["save_dir"], "result")
    os.mkdir(result_save_path)
    
    device_cuda = 'cuda'
    model = ReconstructiveSubNetwork(in_channels=1, out_channels=1)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device_cuda))
    model.to(device=device_cuda)
    model.eval()

    model_seg = DiscriminativeSubNetwork(in_channels=2, out_channels=2)
    model_seg.load_state_dict(torch.load(checkpoint_path[:-5] + "_seg.pckl", map_location=device_cuda))

    model_seg.to(device=device_cuda)
    model_seg.eval()

    dataset = MVTecDRAEMInferenceDataset(args)
    dataloader = DataLoader(dataset, batch_size=1,
                            shuffle=False, num_workers=0)

    mask_cnt = 0

    anomaly_score_prediction = []
    image_name = list()

    for i_batch, sample_batched in tqdm(enumerate(dataloader)):

        gray_batch = sample_batched["image"].to(device=device_cuda)

        gray_rec = model(gray_batch)
        joined_in = torch.cat((gray_rec.detach(), gray_batch), dim=1)
        out_mask = model_seg(joined_in)            
        out_mask_sm = torch.softmax(out_mask, dim=1)
        
        soft_img = np.transpose(out_mask_sm[:, 1, :, :].cpu().detach().numpy(), [1, 2, 0])
        soft_img = cv2.resize(soft_img, (512, 512))
        soft_img_f = soft_img.flatten()
        soft_img_f_s = np.argsort(soft_img_f)[np.int8(-len(soft_img_f)*top_rate):]
        t10_v = soft_img_f[soft_img_f_s]
        
        if args['top_rate'] != 'max':
            image_score = list()
            cors = list()
            for i in range(len(t10_v)):
                cor = np.where(soft_img == t10_v[i])
                v = soft_img[cor[0], cor[1]]
                cors.append(cor)
                image_score.append(v)

            image_score = np.mean(t10_v)
    
        else:
            image_score = np.max(t10_v)

        anomaly_score_prediction.append(image_score)
    
        om_sm_0 = soft_img * 255    
        plt.figure()
        plt.imshow(om_sm_0, cmap='gray')
        for i in range(len(cors)):
            circle = patches.Circle((cors[i][1][0], cors[i][0][0]), radius=3, edgecolor='r', facecolor='r')
            plt.gca().add_patch(circle)
        plt.xticks([])  # x_label 제거
        plt.yticks([])  # y_label 제거
        plt.savefig(os.path.join(result_save_path, sample_batched['file_name'][0]), bbox_inches='tight', pad_inches=0)      
        plt.close()

        image_name.append(sample_batched['file_name'][0])
        mask_cnt += 1

    result_csv = pd.DataFrame(data=image_name, columns=['IMAGE_NAME'])
    result_csv['PRED'] = anomaly_score_prediction
    result_csv.to_csv(os.path.join(args["save_dir"], "inference_result.csv"), index=False)

def check_IQI(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=21)
    kernel1 = np.ones((5,1), np.uint8)
    kernel2 = np.ones((1,5), np.uint8)


    # 그라디언트 크기 계산
    gradient_magnitude = np.sqrt(sobel_x ** 2 + sobel_x ** 2)

    # 경계를 0과 1로 이루어진 이진 마스크로 변환
    binary_mask = np.uint8(gradient_magnitude > np.percentile(gradient_magnitude, 93))

    #팽창
    binary_mask = cv2.dilate(binary_mask, kernel2, iterations=3)

    #특정 크기 이하의 덩어리를 제거
    contours, hierarchy = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) < 3000:
            cv2.drawContours(binary_mask, [cnt], -1, 0, -1)


    #침식
    binary_mask = cv2.erode(binary_mask, kernel2, iterations=3)
    binary_mask = cv2.erode(binary_mask, kernel1, iterations=1)

    #특정 크기 이하의 덩어리를 제거
    contours, hierarchy = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) < 1000:
            cv2.drawContours(binary_mask, [cnt], -1, 0, -1)

    binary_mask = cv2.dilate(binary_mask, kernel1, iterations=2)
    binary_mask = cv2.dilate(binary_mask, kernel2, iterations=2)

    contours, hierarchy = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return len(contours)

def find_IQI(image_list):
    check_IQI_list = []
    found_iqi_break = False
    for i in range(len(image_list)):
        image = image_list[i]
        if check_IQI(image) <= 1:
            image = image_list[i+1]
            if check_IQI(image) <= 1:
                check_IQI_list.append(i-1)
                found_iqi_break = True
                break
    if found_iqi_break:
        for i in range(len(image_list)-1, 0, -1):
            image = image_list[i]

            if check_IQI(image) <= 1:
                image = image_list[i-1]
                if check_IQI(image) <= 1:
                    check_IQI_list.append(i+1)
                    break

    return check_IQI_list

def make_test_json(data_path, config):
    img_name_list = natsort.natsorted(os.listdir(data_path))
    img_list = []
    for img_name in img_name_list:
        img_list.append(cv2.resize(cv2.imread(os.path.join(data_path, img_name)), (config["target_size"][0], config["target_size"][1])))

    IQI_list = find_IQI(img_list)

    main_json = dict()
    len_img = len(img_name_list)
    for num in range(len_img):
        main_json[img_name_list[num]] = dict()
        main_json[img_name_list[num]]["img_name"] = img_name_list[num]
        if num < IQI_list[0] or num > IQI_list[1]:
            main_json[img_name_list[num]]["part"] = "end tap"
        elif num == IQI_list[0] or num == IQI_list[1]:
            main_json[img_name_list[num]]["part"] = "IQI"
        else:
            main_json[img_name_list[num]]["part"] = "welding"
    
    for num in range(len_img):
        result_save_path = os.path.join(config["save_dir"], "result")
        cam_img_path = os.path.join(result_save_path, img_name_list[num])
        img = cv2.imread(cam_img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        _, buffer = cv2.imencode('.png', img)
        base64_encoded = base64.b64encode(buffer).decode('utf-8')
        main_json[img_name_list[num]]["cam_img"] = base64_encoded
    
    img_name_list = natsort.natsorted(os.listdir(data_path))

    df1 = pd.read_csv(os.path.join(config["save_dir"], "inference_result.csv"))

    for img_name in img_name_list:
        img = cv2.imread(os.path.join(data_path, img_name), cv2.IMREAD_GRAYSCALE)
        img = normalization(img)
        img = cv2.resize(img, (config["target_size"][0], config["target_size"][1]), interpolation = cv2.INTER_CUBIC)
        _, buffer = cv2.imencode('.png', img)
        base64_encoded = base64.b64encode(buffer).decode('utf-8')
        main_json[img_name]["img"] = base64_encoded
        main_json[img_name]["predict_score"] = float(df1[df1["IMAGE_NAME"] == img_name]["PRED"])

    return main_json

def save_test_result(save_path, main_json):
    with open(save_path, "w") as f:
        json.dump(main_json, f, ensure_ascii = False, indent = 4)