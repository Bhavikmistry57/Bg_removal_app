from multiprocessing import cpu_count



# Socket Path

bind = 'unix:/home/milan/FastApi/gunicorn.sock'



# Worker Options

workers = cpu_count() * 2 + 1
print(workers)

worker_class = 'uvicorn.workers.UvicornWorker'  



# Logging Options

loglevel = 'debug'

accesslog = '/home/milan/FastApi/access_log'

errorlog =  '/home/milan/FastApi/error_log'
