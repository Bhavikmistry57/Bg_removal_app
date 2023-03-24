import asyncio
import aiohttp
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
#from time import time
import glob

list_of_urls = []
# image_list = glob.glob("/home/milan/100_images/*")
image_list = glob.glob("/home/milan/FastApi/segmentation/static/img/*")
#print(image_list)
url = "http://192.168.2.132/Remove_Background?image_path="
for image_path in image_list:
    updt_url = url+image_path
    list_of_urls.append(updt_url) 
#print(list_of_urls)

def threadpool(each_url):
    res = requests.post(each_url)
    print(res)
    print(res.json())

start_time = time.time()
with ThreadPoolExecutor(max_workers=3) as executor:
    for each_url in list_of_urls:
        processes = {executor.submit(threadpool, each_url)}
end_time = time.time()
for task in as_completed(processes): 
    print(task.result())




if __name__ == "__main__":
    #start_time = time.time()
    threadpool(each_url)
    print(f'Time taken: {(end_time - start_time)}')















# async def by_aiohttp_concurrency(url, session):
#     # use aiohttp

#     async with aiohttp.ClientSession() as session:
#         tasks = [] 
#         url = "http://192.168.2.180/Remove_Background?image_path=/home/milan/Original_Object_Images/01a94e36bca0a0e6.jpg"
#         tasks.append(asyncio.create_task(def by_aiohttp_concurrency(url, session)))

#         original_result = await asyncio.gather(*tasks)
#         for res in original_result:
#             print(res)

# if __name__ == "__main__":
#     total = 5
        
#     start_time = time.time()
#     loop = asyncio.get_event_loop()
#     a1 = loop.create_task(by_aiohttp_concurrency(total))
#     loop.run_until_complete(a1)
#     asyncio.run(by_aiohttp_concurrency(total))
#     print("--- It took %s seconds ---" % (time.time() - start_time))