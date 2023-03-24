import asyncio
import aiohttp
import glob
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

# URL = 'https://httpbin.org/uuid'


async def fetch(session, url):
    async with session.post(url) as response:
        print("not working")
        json_response = await response.text()
        print(response.status)
        print(json_response)


async def main():

    #list_of_urls = []
    image_list = glob.glob("/home/milan/100_images/*")
    #print(image_list)
    # url = "http://192.168.2.180/Remove_Background?image_path="
    # for image_path in image_list:
    #     updt_url = url+image_path
    #     list_of_urls.append(updt_url) 
    #print(list_of_urls)

    async with aiohttp.ClientSession() as session:
        print("hellooooooo")
        tasks = []
        #for each_url in list_of_urls:
            #print(each_url)
        for image_path in image_list:
            url = f"http://192.168.2.180/Remove_Background?image_path={image_path}"
            print(url)
            tasks.append(asyncio.ensure_future(fetch(session, url)))
        print("hellooo2")
        original_result = await asyncio.gather(*tasks)
        for result in original_result:
            print(result)


if __name__ == "__main__":
        
    start_time = time.time()
    asyncio.run(main())
    print("--- It took %s seconds ---" % (time.time() - start_time))
