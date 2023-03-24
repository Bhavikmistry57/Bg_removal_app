from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
import uvicorn
import aiofiles
import cv2
################################################################################################3

import os
from skimage import io, transform
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms#, utils
# import torch.optim as optim
import math
import numpy as np
from PIL import Image
import glob
import imutils
import re
import asyncio
import sys, random
from torchvision import models, transforms

import uuid
from segmentation.yolo_classification.detect import YoloHeader
from segmentation.data_loader import RescaleT
from segmentation.data_loader import ToTensor
from segmentation.data_loader import ToTensorLab
from segmentation.data_loader import SalObjDataset

from segmentation.model import U2NET # full size version 173.6 MB
from segmentation.model import U2NETP # small version u2net 4.7 MB

#import cv2
from segmentation.edge_function_for_static import *
from pydantic import BaseModel

##############################################################################################################################

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/segmentation", StaticFiles(directory="segmentation"), name="segmentation")



MAX_IMAGE_SIZE = 1024

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
################################################################################################################################

# class combine_pred(BaseModel):
#     image_path: str
    # need_original_size: bool
    # original_height: int
    # original_width: int

# def __init__(self):
classification_model_path = "/home/milan/FastApi//segmentation/saved_models/classification/logo_nologo_classification.pth"
classification_model = torch.load(classification_model_path,map_location=torch.device('cpu'))
classification_model.eval()
yolo_obj = YoloHeader()





class_names = {
        0: 'logo',
        1: 'no logo',
        }

model_name = 'u2net'#u2netp

img_name_list = []
prediction_dir = "/home/milan/FastApi/segmentation/static/output/results/"

# file1 = open("./MyFile.txt", "w")
# file1.close()






product_model_path = "/home/milan/FastApi/segmentation/saved_models/Product/u2net_bce_itr_3500_train_0.079868_tar_0.008470.pth"
human_model_path = "/home/milan/FastApi/segmentation/saved_models/Human/u2net_bce_itr_10500_train_0.085318_tar_0.009622.pth"
logo_model_1_path = "/home/milan/FastApi/segmentation/saved_models/Logo/u2net_bce_itr_61000_train_0.203429_tar_0.014888.pth"
logo_model_2_path = "/home/milan/FastApi/segmentation/saved_models/Logo/u2net_bce_itr_49000_train_0.151424_tar_0.010312.pth"


if(model_name=='u2net'):
    print("...load U2NET---173.6 MB")
    net_1 = U2NET(3,1)
    net_2 = U2NET(3,1)
    net_3 = U2NET(3,1)
    net_4 = U2NET(3,1)
    
elif(model_name=='u2netp'):
    print("...load U2NEP---4.7 MB")
    net_1 = U2NETP(3,1)
    net_2 = U2NETP(3,1)
    net_3 = U2NETP(3,1)
    net_4 = U2NETP(3,1)

if torch.cuda.is_available():
    net_1.load_state_dict(torch.load(product_model_path))
    net_1.cuda()
    
    net_2.load_state_dict(torch.load(human_model_path))
    net_2.cuda()
    
    net_3.load_state_dict(torch.load(logo_model_1_path))
    net_3.cuda()
    
    net_4.load_state_dict(torch.load(logo_model_2_path))
    net_4.cuda()
    
else:
    net_1.load_state_dict(torch.load(product_model_path, map_location=torch.device('cpu')))
    net_2.load_state_dict(torch.load(product_model_path, map_location=torch.device('cpu')))
    net_3.load_state_dict(torch.load(product_model_path, map_location=torch.device('cpu')))
    net_4.load_state_dict(torch.load(product_model_path, map_location=torch.device('cpu')))
    
net_1.eval()
net_2.eval()
net_3.eval()
net_4.eval()

############################################################################################################

@app.post("/Remove_Background")
async def classification_fun(image_path):
        # image = cv2.imread(image_path)
    print("hellooooo1")
    need_original_size= True
    original_height=1000
    original_width=1000
    # img = await image_path.read()
    # if image_path.content_type not in ['image/jpeg', 'image/png']:
    #     raise HTTPException(status_code=406, detail="Please upload only .jpeg files")
    # async with aiofiles.open(f"/home/milan/FastApi/segmentation/static/img/{image_path.filename}", "wb") as f:
    #     await f.write(img)
    
    # image_path = f"/home/milan/FastApi/segmentation/static/img/{image_path.filename}"
    print(image_path)
    image=Image.open(image_path).convert('RGB')
    preprocess=transforms.Compose([
    transforms.Resize(size=256),
    transforms.CenterCrop(size=224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225])
    ])
        
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    
    with torch.no_grad():
        inputs=preprocess(image).unsqueeze(0).to(device)
        outputs = classification_model(inputs)
        preds = torch.max(outputs, 1)
        print(type(preds))
        print(preds.indices.item())
        preds = preds.indices.item()
        # preds = int(preds)
        label=class_names.get(preds)
        print("-------------===============-------------------")
        print(label)
        print("-------------===============-------------------")
        
        if preds == 0:
            final_net_list = [net_3, net_4]
            # print("LOGOOOOOOO")
        elif preds ==1:
            Human_status = yolo_obj.predict(image)
            if Human_status == True:
                final_net_list = [net_2]
                # print("HUMANNNNNNNNNNNN")
            else:
                final_net_list = [net_1]
                # print("PRODUCTTTTTTTTTTTTTTTTTTTTT")
            
    

        print(image_path, need_original_size, original_height, original_width)
        full_name, success_parameter = add_2_images_orig(image_path, need_original_size, original_height, original_width, final_net_list)
        return {f"http://192.168.2.132/segmentation/static/output/results/{full_name[0]}.png"}, success_parameter


def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)
    dn = (d-mi)/(ma-mi)
    print("hellooooo2/////////////")
    return dn
    
    

def save_output(image_name,pred,d_dir):
    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()
    im = Image.fromarray(predict_np*255).convert('RGB')
    unique_id = str(uuid.uuid4().hex)
    
    img_name = image_name.split(os.sep)[-1].split(".")[0] + unique_id + "." +image_name.split(os.sep)[-1].split(".")[-1]
    
    image = io.imread(image_name)
    imo = im.resize((image.shape[1],image.shape[0]),resample=Image.BILINEAR)

    pb_np = np.array(imo)
    print("hello")
    
    aaa = img_name.split(".")
    bbb = aaa[0:-1]
    imidx = bbb[0]
    for i in range(1,len(bbb)):
        imidx = imidx +unique_id+"." + bbb[i]

#         imo.save(d_dir+imidx+'.png')
    cv2.imwrite(f'{d_dir}{imidx}.png',pb_np)
    print("hellooooo3")
    return img_name

def main(image_path, final_net):
    img_transform=transforms.Compose([RescaleT(512),ToTensorLab(flag=0)])
    image = io.imread(image_path)
    
    #imname = image
    imidx = np.array([0])
    print("hellooooo4")
    label_name_list = []
    if(0==len(label_name_list)):
        label_3 = np.zeros(image.shape)
    else:
        label_3 = io.imread(label_name_list[0])

    label = np.zeros(label_3.shape[0:2])
    if(3==len(label_3.shape)):
        label = label_3[:,:,0]
    elif(2==len(label_3.shape)):
        label = label_3

    if(3==len(image.shape) and 2==len(label.shape)):
        label = label[:,:,np.newaxis]
    elif(2==len(image.shape) and 2==len(label.shape)):
        image = image[:,:,np.newaxis]
        label = label[:,:,np.newaxis]
    
    sample = {'imidx':imidx, 'image':image, 'label':label}
    transform = True
    if transform:
        sample = img_transform(sample)
    print("hellooooo5")
    image_1 = torch.unsqueeze(sample['image'],0)
    inputs = image_1.type(torch.FloatTensor)
    print("it is stopped here")
    inputs_v = Variable(inputs, requires_grad=False)
    d0, d1, d2, d3, d4, d5, d6 = final_net(inputs_v)
    
    pred = d1[:,0,:,:]
    pred = normPRED(pred)
#         print("Before memory use",size(process.memory_info().rss))
    del d0, d1, d2, d3, d4, d5, d6

    if not os.path.exists(prediction_dir):
        os.makedirs(prediction_dir, exist_ok=True)
    img_name = save_output(image_path,pred,prediction_dir)
    print("hellooooo6")
    return img_name, image_path


    
def add_2_images_orig( image_path, need_original_size, original_height, original_width, final_net_list):
    success_parameter = True
    imidx_list = []
    for final_net in final_net_list:
        
        print(image_path, need_original_size)
        try:
            ip_img_name,image_path = main(image_path, final_net)

        except:
            torch.cuda.empty_cache()
            success_parameter = False
            imidx = ""
            return imidx, success_parameter
        print("hellooooo7")
        out_name = ""
        image_name_list = ip_img_name.split(".")


        imidx = image_name_list[0]
        # out_name = imidx+".png"
        out_name = imidx+".png"
        final_path = prediction_dir + out_name

        # print(final_path)
        full_name = imidx+""
        save_path = prediction_dir + full_name +".png"
        torch.cuda.empty_cache()

        print("hellooooo8")
        img = cv2.imread(image_path)
        mask = cv2.imread(final_path)
        cv2.imwrite("./segmentation/static/prediction.png", mask)

        ms = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        image_height = ms.shape[0] 
        image_width = ms.shape[1] 

        img_added = np.dstack((img, ms))
        img_2 = img_added
        if img_2 is None:
            kernel_size = 3
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            ms = cv2.erode(ms, kernel)
            final_op_img = np.dstack((img, ms))
        else:
            final_op_img = img_2
        # print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

        if need_original_size == "True":
            final_op_img = cv2.resize(final_op_img, (original_width, original_height))

        print("hellooooo9")
#         # print(need_original_size)        
        if need_original_size == "False":
            thresh = cv2.threshold(ms, 30, 255, cv2.THRESH_BINARY)[1]
            new_top_left=[0,0]
            new_bottom_right =[0,0]
            thresh_where = np.argwhere(thresh == 255)

            top_left = np.min(thresh_where, axis=0)
            bottom_right = np.max(thresh_where, axis=0)


            new_top_left[0] = int(top_left[0])-10
            new_top_left[1] = int(top_left[1])-10
            new_bottom_right[0] =int(bottom_right[0])+10
            new_bottom_right[1] = int(bottom_right[1])+10

            if (new_top_left[0])<0:
                new_top_left[0] = 0
#             
            if (new_top_left[1])<0:
                new_top_left[1] = 0

            if (new_bottom_right[0])>image_height:
                new_bottom_right[0] = image_height

            if (new_bottom_right[1])>image_width:
                new_bottom_right[1] = image_width


            # print(new_top_left[0],new_bottom_right[0], new_top_left[1],new_bottom_right[1])
            # print("#####################################################")
            final_op_img = final_op_img[new_top_left[0]:new_bottom_right[0], new_top_left[1]:new_bottom_right[1]]
            final_op_img = cv2.resize(final_op_img, (original_width, original_height))
        cv2.imwrite(final_path,final_op_img)
        imidx_list.append(imidx)
    print(imidx_list)
    return imidx_list, success_parameter

# def run_functions():
#     classification_fun("/home/milan/100_images/18.png")
#     # normPRED()
#     # main()
#     # save_output()
#     # add_2_images_orig()

# run_functions()
#asyncio.run(classification_fun("/home/milan/100_images/02b484f43777cd5b.jpg"))






#####################################################################################################################

@app.get("/index")
async def read_root():
    return("helllooooooooooooooo")


@app.post("/file/")
async def create_upload_file(file: UploadFile = File(...)):
    img = await file.read()
    if file.content_type not in ['image/jpeg', 'image/png']:
        raise HTTPException(status_code=406, detail="Please upload only .jpeg files")
    async with aiofiles.open(f"/home/milan/FastApi/static/input/{file.filename}", "wb") as f:
        await f.write(img)
        image = cv2.imread(f"/home/milan/FastApi/static/input/{file.filename}")
        img_resize = cv2.resize(image,(100,100))
        cv2.imwrite(f"/home/milan/FastApi/static/output/{file.filename}",img_resize)
    return {"URL": f"http://192.168.0.116/static/output/{file.filename}"}




# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=80)