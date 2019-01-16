from PIL import Image
from os import listdir
import os
from os.path import isfile, join
import random

data_dir = "data/blender/step_5"
aug_dir = "textures"
out_dir = "data/augmented/step_5"

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

data_files = [join(data_dir, f) for f in listdir(data_dir)
              if (isfile(join(data_dir, f)) and f.endswith(('.png', 'jpg')))]
aug_files = [join(aug_dir, f) for f in listdir(aug_dir)
             if (isfile(join(aug_dir, f)) and f.endswith(('.png', 'jpg')))]

for data in data_files:
    background = Image.open(random.choice(aug_files))
    foreground = Image.open(data)

    background = background.resize(foreground.size, Image.ANTIALIAS)
    background.paste(foreground, (0, 0), foreground)
    filename, _ = os.path.splitext(os.path.basename(foreground.filename))
    background.save(join(out_dir, filename + '.jpg'), "JPEG")
