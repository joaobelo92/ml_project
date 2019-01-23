import requests
import os
import shutil
import concurrent.futures

dataset_name = "chair"
dataset_id = "n03002096"
f = requests.get(f"http://image-net.org/api/text/imagenet.synset.geturls?wnid={dataset_id}")
dl_dir = f"data/image_net/{dataset_name}"

urls = f.text.splitlines()

if not os.path.exists(dl_dir):
    os.makedirs(dl_dir)


def get_img(count):
    path = os.path.join(dl_dir, f"{dataset_name}_{count}.jpg")
    r = requests.get(urls[count], stream=True)
    if r.status_code == 200:
        with open(path, 'wb') as f:
            r.raw.decode_content = True
            shutil.copyfileobj(r.raw, f)

with concurrent.futures.ThreadPoolExecutor(max_workers=50) as e:
    for i in range(len(urls)):
        e.submit(get_img, i)
