# Copyright 2022 (C) antillia.com. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# 2022/10/01 Copyright (C) antillia.com

# expand.py

import os
import sys
import shutil
import glob
from PIL import Image, ImageDraw
import traceback

class ImageResizer:

  def __init__(self):
    pass

  def resize(self, input_dir, output_dir, target_size=224):
    subdirs = os.listdir(input_dir)
    print(subdirs)
    for subdir in subdirs:
      output_subdir = os.path.join(output_dir, subdir)
      if os.path.exists(output_subdir):
        shutil.rmtrees(output_subdir) 

      if not os.path.exists(output_subdir):
        os.makedirs(output_subdir) 

      subdir_path = os.path.join(input_dir, subdir) 
      files = glob.glob(subdir_path + "/*.png")
      print("--- dir {}  num files {}".format(subdir_path, len(files)))
      for file in files:  
        basename = os.path.basename(file)

        output_filepath = os.path.join(output_subdir, basename)

        try:
          image = Image.open(file)
          image = image.convert('L')

          w, h = image.size
          rw = target_size
          rh = target_size
          if w > rw and h > rh :

            basename = os.path.basename(file)
            resized_image = image.resize((rw, rh))
            
            resized_image.save(output_filepath) #, quality=95)
            print("---- saved {}".format(output_filepath))
          else:
            print("-------w > {}  h > {}".format(w, h))            
        except:
          traceback.print_exc()
          print("---------{}".format(output_filepath))
          input("-----Hit any key")
  
if __name__ == "__main__":
  input_dir = "./INbreast+MIAS+DDSM Dataset"
  output_dir = "./INbreast+MIAS+DDSM_Dataset_224x244_master"

  try:
    if not os.path.exists(input_dir):
      raise Exception("Not found " + input_dir)
    
    if os.path.exists(output_dir):
      shutil.rmtree(output_dir)
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)
    resizer = ImageResizer()
    resizer.resize(input_dir, output_dir)

  except:
    traceback.print_exc()
