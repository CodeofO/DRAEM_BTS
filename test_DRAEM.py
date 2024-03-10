import torch
import json
import natsort
import base64
import torch.nn.functional as F
from DRAEM_module.data_loader import MVTecDRAEMTestDataset
from torch.utils.data import DataLoader
import numpy as np
from DRAEM_module.model_unet import ReconstructiveSubNetwork, DiscriminativeSubNetwork
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, precision_recall_curve, roc_curve, auc
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

    dataset = MVTecDRAEMTestDataset(args)
    dataloader = DataLoader(dataset, batch_size=1,
                            shuffle=False, num_workers=0)

    mask_cnt = 0

    anomaly_score_prediction = []
    image_name = list()

    for i_batch, sample_batched in tqdm(enumerate(dataloader)):

        gray_batch = sample_batched["image"].to(device=device_cuda)

        is_normal = sample_batched["has_anomaly"].detach().numpy()[0 ,0]

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
        plt.xticks([])
        plt.yticks([])
        # 이미지 저장 경로
        plt.savefig(os.path.join(result_save_path, sample_batched['file_name'][0]), bbox_inches='tight', pad_inches=0)      
        plt.close()

        image_name.append(sample_batched['file_name'][0])
        mask_cnt += 1

    result_csv = pd.DataFrame(data=image_name, columns=['IMAGE_NAME'])
    result_csv['PRED'] = anomaly_score_prediction
    result_csv.to_csv(os.path.join(args["save_dir"], "test_result.csv"), index=False)

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

    df1 = pd.read_csv(os.path.join(config["save_dir"], "test_result.csv"))

    for img_name in img_name_list:
        img = cv2.imread(os.path.join(data_path, img_name), cv2.IMREAD_GRAYSCALE)
        img = normalization(img)
        img = cv2.resize(img, (config["target_size"][0], config["target_size"][1]), interpolation = cv2.INTER_CUBIC)
        _, buffer = cv2.imencode('.png', img)
        base64_encoded = base64.b64encode(buffer).decode('utf-8')
        main_json[img_name]["img"] = base64_encoded
        main_json[img_name]["predict_score"] = float(df1[df1["IMAGE_NAME"] == img_name]["PRED"])

    return main_json

def evaluate_data(y_true, Y_pred):

    TP = 0
    FP = 0
    FN = 0
    TN = 0

    for i in range(len(Y_pred)):
        if y_true[i] == 0 and Y_pred[i] == 0:
            TN = TN + 1
        elif y_true[i] == 0 and Y_pred[i] == 1:
            FP = FP + 1
        elif y_true[i] == 1 and Y_pred[i] == 0:
            FN = FN + 1
        elif y_true[i] == 1 and Y_pred[i] == 1:
            TP = TP + 1

    Recall = recall_score(y_true, Y_pred)
    Precision = precision_score(y_true, Y_pred)
    Accuracy = accuracy_score(y_true, Y_pred)
    F1_Score = f1_score(y_true, Y_pred)
    return TN, FP, FN, TP, Accuracy, F1_Score, Recall, Precision

def make_curve(config, main_json):
    y_true_df = pd.read_csv(os.path.join(config["save_dir"], "ground_truth.csv"))
    y_true = list(y_true_df["ground_truth"])
    y_pred_df = pd.read_csv(os.path.join(config["save_dir"], "test_result.csv"))
    y_pred_list = list(y_pred_df["PRED"])
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    total_thresholds = []
    total_Accuracy = []
    total_F1_Score = []
    total_Recall = []
    total_Precision = []
    total_TN = []
    total_FP = []
    total_FN = []
    total_TP = []

    for threshold in thresholds:
        y_pred = []
        for i in range(len(y_pred_list)):
            if y_pred_list[i] <= threshold:
                y_pred.append(0)
            else:
                y_pred.append(1)

        TN, FP, FN, TP, Accuracy, F1_Score, Recall, Precision = evaluate_data(y_true, y_pred)
        total_Accuracy.append(str(round(Accuracy, 3)))
        total_F1_Score.append(str(round(F1_Score, 3)))
        total_Recall.append(str(round(Recall, 3)))
        total_Precision.append(str(round(Precision, 3)))
        total_TN.append(str(TN))
        total_FP.append(str(FP))
        total_FN.append(str(FN))
        total_TP.append(str(TP))
        total_thresholds.append(str(threshold))

    main_json["threshold"] = total_thresholds
    main_json["tn"] = total_TN
    main_json["fp"] = total_FP
    main_json["fn"] = total_FN
    main_json["tp"] = total_TP
    main_json["accuracy"] = total_Accuracy
    main_json["fonescore"] = total_F1_Score
    main_json["recall"] = total_Recall
    main_json["precision"] = total_Precision

    precision, recall, thresholds = precision_recall_curve(y_true = y_true, probas_pred = y_pred_list, drop_intermediate = True)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, marker = ".")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.grid(True)
    plt.savefig(f'{config["save_dir"]}/precision_recall_curve.png')
    plt.close()
    
    fpr, tpr, thresholds = roc_curve(y_true = y_true, y_score = y_pred_list, drop_intermediate = True)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(auc(fpr, tpr)))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig(f'{config["save_dir"]}/roc_curve.png')
    plt.close()

    return main_json

def put_json_roc_pr(data_path, main_json):
    for img_name in ["precision_recall_curve.png", "roc_curve.png"]:
        with open(os.path.join(data_path, img_name), "rb") as image_file:
            image_data = image_file.read()
        
        image_base64 = base64.b64encode(image_data).decode()
        if img_name == "precision_recall_curve.png":
            main_json["prcurve"] = image_base64
        else:
            main_json["roccurve"] = image_base64
    
    return main_json

def save_test_result(save_path, main_json):
    with open(save_path, "w") as f:
        json.dump(main_json, f, ensure_ascii = False, indent = 4)