import requests
from concurrent.futures import ThreadPoolExecutor
import glob
import time
import cv2
import aiohttp
import asyncio
import glob


import requests
from concurrent.futures import ThreadPoolExecutor
import glob
import time



image_list = glob.glob("/home/milan/100_images/*")
list_of_urls = []
url = "http://192.168.2.180/Remove_Background?image_path="
for image_path in image_list:
    updt_url = url+image_path
    list_of_urls.append(updt_url) 
print(list_of_urls)

def get_url(args):
    headers = {}
    for each_url in list_of_urls:
    #payload={'need_original_size': 'False'}
        res = requests.request("POST", each_url, headers=headers, files=args, verify = False)
        print(res)
# res = []
# res = glob.glob("/home/ubuntu/brandspot_django/U_2_Net_Django/segmentation/static/100_images/*.png")​
# print(len(res))
# list_of_urls=[]

# for i in range (len(res)):
#     inside_list = [('image_file',('img.jpg',open(str(res[i]),'rb'),'image/jpeg'))]
#     list_of_urls.append(inside_list)

print("List_is_ready")
print("##########################################################################################################################################")
start = time.time()
with ThreadPoolExecutor(max_workers=3) as pool:
    response_list = list(pool.map(get_url, list_of_urls))
end = time.time()  
    
count_200 = 0
for response in response_list:
    print(str(response))
    print(response.json())
    
    if "200" in str(response):
    	count_200 = count_200 + 1
    # print(response.text)
    
print(":::::::::::::::::::::::::::::::::::::::::::")
print(end-start)
print(count_200)
print(":::::::::::::::::::::::::::::::::::::::::::")

# async def async_func():
   

#     list_of_urls = []
#     image_list = glob.glob("/home/milan/100_images/*")
#     #print(image_list)
#     url = "http://192.168.2.180/Remove_Background?image_path="
#     for image_path in image_list:
#         updt_url = url+image_path
#         list_of_urls.append(updt_url) 

#     async with aiohttp.ClientSession() as session:
#         for each_url in list_of_urls:
#             async with session.post(each_url) as response:
#                 res = await response.text()
#                 print("Status:", response.status)
#                 print(res)
                
                
# if __name__ == "__main__":
        
#     start_time = time.time()
#     asyncio.run(async_func())
#     print("--- It took %s seconds ---" % (time.time() - start_time))
# # loop = asyncio.get_event_loop()
# # loop.run_until_complete(main())