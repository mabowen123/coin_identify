import os
import requests
from threading import Thread, Semaphore

# 图片链接列表
image_urls = [...]  # 你的图片链接列表

# 下载保存路径
download_path = "./images"

# 最大线程数
max_threads = 50

# 创建保存图片的目录
if not os.path.exists(download_path):
    os.makedirs(download_path)

# 信号量，控制并发线程数
semaphore = Semaphore(max_threads)

def download_image(url):
    # 获取信号量，阻塞直到获取到信号量
    semaphore.acquire()
    try:
        # 发送请求并下载图片
        response = requests.get(url)
        if response.status_code == 200:
            # 提取图片名称
            image_name = os.path.basename(url)
            # 保存图片到指定路径
            with open(os.path.join(download_path, image_name), 'wb') as f:
                f.write(response.content)
            print(f"Downloaded {image_name}")
    except Exception as e:
        print(f"Error downloading image from {url}: {e}")
    finally:
        # 释放信号量
        semaphore.release()

# 创建并启动线程
threads = []
for url in image_urls:
    thread = Thread(target=download_image, args=(url,))
    thread.start()
    threads.append(thread)

# 等待所有线程完成
for thread in threads:
    thread.join()

print("All images downloaded.")
