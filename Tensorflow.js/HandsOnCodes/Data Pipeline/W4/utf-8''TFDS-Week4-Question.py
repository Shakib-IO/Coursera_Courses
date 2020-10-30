#!/usr/bin/env python
# coding: utf-8

# # Adding a Dataset of Your Own to TFDS

# In[37]:


import os
import textwrap
import scipy.io
import pandas as pd

from os import getcwd


# ## IMDB Faces Dataset
# 
# This is the largest publicly available dataset of face images with gender and age labels for training.
# 
# Source: https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/
# 
# The IMDb Faces dataset provides a separate .mat file which can be loaded with Matlab containing all the meta information. The format is as follows:  
# **dob**: date of birth (Matlab serial date number)  
# **photo_taken**: year when the photo was taken  
# **full_path**: path to file  
# **gender**: 0 for female and 1 for male, NaN if unknown  
# **name**: name of the celebrity  
# **face_location**: location of the face (bounding box)  
# **face_score**: detector score (the higher the better). Inf implies that no face was found in the image and the face_location then just returns the entire image  
# **second_face_score**: detector score of the face with the second highest score. This is useful to ignore images with more than one face. second_face_score is NaN if no second face was detected.  
# **celeb_names**: list of all celebrity names  
# **celeb_id**: index of celebrity name  

# Next, let's inspect the dataset

# ## Exploring the Data

# In[39]:


# Inspect the directory structure
imdb_crop_file_path = f"{getcwd()}/../tmp2/imdb_crop"
files = os.listdir(imdb_crop_file_path)
print(textwrap.fill(' '.join(sorted(files)), 80))


# In[40]:


# Inspect the meta data
imdb_mat_file_path = f"{getcwd()}/../tmp2/imdb_crop/imdb.mat"
meta = scipy.io.loadmat(imdb_mat_file_path)


# In[41]:


meta


# ## Extraction

# Let's clear up the clutter by going to the metadata's most useful key (imdb) and start exploring all the other keys inside it

# In[42]:


root = meta['imdb'][0, 0]


# In[43]:


desc = root.dtype.descr
desc


# In[44]:


# EXERCISE: Fill in the missing code below.

full_path = root["full_path"][0]

# Do the same for other attributes
names = root['name'][0] # YOUR CODE HERE
dob = root['dob'][0]# YOUR CODE HERE
gender = root['gender'][0]# YOUR CODE HERE
photo_taken = root['photo_taken'][0]# YOUR CODE HERE
face_score = root['face_score'][0]# YOUR CODE HERE
face_locations = root['face_location'][0]# YOUR CODE HERE
second_face_score = root['second_face_score'][0]# YOUR CODE HERE
celeb_names = root['celeb_names'][0]# YOUR CODE HERE
celeb_ids = root['celeb_id'][0]# YOUR CODE HERE

print('Filepaths: {}\n\n'
      'Names: {}\n\n'
      'Dates of birth: {}\n\n'
      'Genders: {}\n\n'
      'Years when the photos were taken: {}\n\n'
      'Face scores: {}\n\n'
      'Face locations: {}\n\n'
      'Second face scores: {}\n\n'
      'Celeb IDs: {}\n\n'
      .format(full_path, names, dob, gender, photo_taken, face_score, face_locations, second_face_score, celeb_ids))


# In[45]:


print('Celeb names: {}\n\n'.format(celeb_names))


# Display all the distinct keys and their corresponding values

# In[46]:


names = [x[0] for x in desc]
names


# In[47]:


values = {key: root[key][0] for key in names}
values


# ## Cleanup

# Pop out the celeb names as they are not relevant for creating the records.

# In[48]:


del values['celeb_names']
names.pop(names.index('celeb_names'))


# Let's see how many values are present in each key

# In[49]:


for key, value in values.items():
    print(key, len(value))


# ## Dataframe

# Now, let's try examining one example from the dataset. To do this, let's load all the attributes that we've extracted just now into a Pandas dataframe

# In[50]:


df = pd.DataFrame(values, columns=names)
df.head()


# The Pandas dataframe may contain some Null values or nan. We will have to filter them later on.

# In[51]:


df.isna().sum()


# # TensorFlow Datasets
# 
# TFDS provides a way to transform all those datasets into a standard format, do the preprocessing necessary to make them ready for a machine learning pipeline, and provides a standard input pipeline using `tf.data`.
# 
# To enable this, each dataset implements a subclass of `DatasetBuilder`, which specifies:
# 
# * Where the data is coming from (i.e. its URL). 
# * What the dataset looks like (i.e. its features).  
# * How the data should be split (e.g. TRAIN and TEST). 
# * The individual records in the dataset.
# 
# The first time a dataset is used, the dataset is downloaded, prepared, and written to disk in a standard format. Subsequent access will read from those pre-processed files directly.

# ## Clone the TFDS Repository
# 
# The next step will be to clone the GitHub TFDS Repository. For this particular notebook, we will clone a particular version of the repository. You can clone the repository by running the following command:
# 
# ```
# !git clone https://github.com/tensorflow/datasets.git -b v1.2.0
# ```
# 
# However, for simplicity, we have already cloned this repository for you and placed the files locally. Therefore, there is no need to run the above command if you are running this notebook in Coursera environment.
# 
# Next, we set the current working directory to `/datasets/`.

# In[52]:


cd datasets


# If you want to contribute to TFDS' repo and add a new dataset, you can use the the following script to help you generate a template of the required python file. To use it, you must first clone the tfds repository and then run the following command:

# In[53]:


get_ipython().run_cell_magic('bash', '', '\npython tensorflow_datasets/scripts/create_new_dataset.py \\\n  --dataset my_dataset \\\n  --type image')


# If you wish to see the template generated by the `create_new_dataset.py` file, navigate to the folder indicated in the above cell output. Then go to the `/image/` folder and look for a file called `my_dataset.py`. Feel free to open the file and inspect it. You will see a template with place holders, indicated with the word `TODO`, where you have to fill in the information. 
# 
# Now we will use IPython's `%%writefile` in-built magic command to write whatever is in the current cell into a file. To create or overwrite a file you can use:
# ```
# %%writefile filename
# ```
# 
# Let's see an example:

# In[54]:


get_ipython().run_cell_magic('writefile', 'something.py', 'x = 10')


# Now that the file has been written, let's inspect its contents.

# In[55]:


get_ipython().system('cat something.py')


# ## Define the Dataset with `GeneratorBasedBuilder`
# 
# Most datasets subclass `tfds.core.GeneratorBasedBuilder`, which is a subclass of `tfds.core.DatasetBuilder` that simplifies defining a dataset. It works well for datasets that can be generated on a single machine. Its subclasses implement:
# 
# * `_info`: builds the DatasetInfo object describing the dataset
# 
# 
# * `_split_generators`: downloads the source data and defines the dataset splits
# 
# 
# * `_generate_examples`: yields (key, example) tuples in the dataset from the source data
# 
# In this exercise, you will use the `GeneratorBasedBuilder`.
# 
# ### EXERCISE: Fill in the missing code below.

# In[56]:


get_ipython().run_cell_magic('writefile', 'tensorflow_datasets/image/imdb_faces.py', '\n# coding=utf-8\n# Copyright 2019 The TensorFlow Datasets Authors.\n#\n# Licensed under the Apache License, Version 2.0 (the "License");\n# you may not use this file except in compliance with the License.\n# You may obtain a copy of the License at\n#\n#     http://www.apache.org/licenses/LICENSE-2.0\n#\n# Unless required by applicable law or agreed to in writing, software\n# distributed under the License is distributed on an "AS IS" BASIS,\n# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.\n# See the License for the specific language governing permissions and\n# limitations under the License.\n\n"""IMDB Faces dataset."""\nfrom __future__ import absolute_import\nfrom __future__ import division\nfrom __future__ import print_function\n\nimport collections\nimport re\n\nimport tensorflow as tf\nimport tensorflow_datasets.public_api as tfds\n\n_DESCRIPTION = """Since the publicly available face image datasets are often of small to medium size, rarely exceeding tens of thousands of images, this is an attempt to put together a diverse dataset in that domain."""\n\n_URL = ("https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/")\n_DATASET_ROOT_DIR = \'imdb_crop\'\n_ANNOTATION_FILE = \'imdb.mat\'\n\n\n_CITATION = """@article{Rothe-IJCV-2016,\n  author = {Rasmus Rothe and Radu Timofte and Luc Van Gool},\n  title = {Deep expectation of real and apparent age from a single image without facial landmarks},\n  journal = {International Journal of Computer Vision (IJCV)},\n  year = {2016},\n  month = {July},\n}\n@InProceedings{Rothe-ICCVW-2015,\n  author = {Rasmus Rothe and Radu Timofte and Luc Van Gool},\n  title = {DEX: Deep EXpectation of apparent age from a single image},\n  booktitle = {IEEE International Conference on Computer Vision Workshops (ICCVW)},\n  year = {2015},\n  month = {December},\n}\n"""\n\n# Source URL of the IMDB faces dataset\n_TARBALL_URL = "https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/imdb_crop.tar"\n\nclass ImdbFaces(tfds.core.GeneratorBasedBuilder):\n  """IMDB Faces dataset."""\n\n  VERSION = tfds.core.Version("0.1.0")\n  \n  def _info(self):\n    return tfds.core.DatasetInfo(\n        builder=self,\n        description=_DESCRIPTION,\n        # Describe the features of the dataset by following this url\n        # https://www.tensorflow.org/datasets/api_docs/python/tfds/features\n        features=tfds.features.FeaturesDict({\n            "image": tfds.features.Image(),\n            "gender":  tfds.features.ClassLabel(num_classes=2),\n            "dob": tf.int32,\n            "photo_taken": tf.int32,\n            "face_location": tfds.features.BBoxFeature(),\n            "face_score": tf.float32,\n            "second_face_score": tf.float32,\n            "celeb_id": tf.int32\n        }),\n        supervised_keys=("image", "gender"),\n        urls=[_URL],\n        citation=_CITATION)\n\n  def _split_generators(self, dl_manager):\n    # Download the dataset and then extract it.\n    download_path = dl_manager.download([_TARBALL_URL])\n    extracted_path = dl_manager.download_and_extract([_TARBALL_URL])\n\n    # Parsing the mat file which contains the list of train images\n    def parse_mat_file(file_name):\n      with tf.io.gfile.GFile(file_name, "rb") as f:\n        # Add a lazy import for scipy.io and import the loadmat method to \n        # load the annotation file\n        dataset = tfds.core.lazy_imports.scipy.io.loadmat(file_name)[\'imdb\']\n      return dataset\n\n    # Parsing the mat file by using scipy\'s loadmat method\n    # Pass the path to the annotation file using the downloaded/extracted paths above\n    meta = parse_mat_file(os.path.join(extracted_path[0], _DATASET_ROOT_DIR, _ANNOTATION_FILE))\n\n    # Get the names of celebrities from the metadata\n    celeb_names = meta[0, 0][\'celeb_names\'][0]\n\n    # Create tuples out of the distinct set of genders and celeb names\n    self.info.features[\'gender\'].names = (\'Female\', \'Male\')\n    self.info.features[\'celeb_id\'].names = tuple([x[0] for x in celeb_names])\n\n    return [\n        tfds.core.SplitGenerator(\n            name=tfds.Split.TRAIN,\n            gen_kwargs={\n                "image_dir": extracted_path[0],\n                "metadata": meta,\n            })\n    ]\n\n  def _get_bounding_box_values(self, bbox_annotations, img_width, img_height):\n    """Function to get normalized bounding box values.\n\n    Args:\n      bbox_annotations: list of bbox values in kitti format\n      img_width: image width\n      img_height: image height\n\n    Returns:\n      Normalized bounding box xmin, ymin, xmax, ymax values\n    """\n\n    ymin = bbox_annotations[0] / img_height\n    xmin = bbox_annotations[1] / img_width\n    ymax = bbox_annotations[2] / img_height\n    xmax = bbox_annotations[3] / img_width\n    return ymin, xmin, ymax, xmax\n  \n  def _get_image_shape(self, image_path):\n    image = tf.io.read_file(image_path)\n    image = tf.image.decode_image(image, channels=3)\n    shape = image.shape[:2]\n    return shape\n\n  def _generate_examples(self, image_dir, metadata):\n    # Add a lazy import for pandas here (pd)\n    pd = tfds.core.lazy_imports.pandas\n\n    # Extract the root dictionary from the metadata so that you can query all the keys inside it\n    root = metadata[0, 0]\n\n    """Extract image names, dobs, genders,  \n               face locations, \n               year when the photos were taken,\n               face scores (second face score too),\n               celeb ids\n    """\n    image_names = root["full_path"][0]\n    # Do the same for other attributes (dob, genders etc)\n    dobs = root[\'dob\'][0]\n    genders = root[\'gender\'][0]\n    face_locations = root[\'face_location\'][0]\n    photo_taken_years = root[\'photo_taken\'][0]\n    face_scores = root[\'face_score\'][0]\n    second_face_scores = root[\'second_face_score\'][0]\n    celeb_id = root[\'celeb_id\'][0]\n        \n    # Now create a dataframe out of all the features like you\'ve seen before\n    df = pd.DataFrame(\n        list(zip(image_names, \n                dobs,\n                genders,\n                face_locations,\n                photo_taken_years,\n                face_scores,\n                second_face_scores,\n                celeb_id)),\n        columns=[\'image_names\', \'dobs\', \'genders\', \'face_locations\', \'photo_taken_years\',\n                \'face_scores\', \'second_face_scores\', \'celeb_id\'])\n\n    # Filter dataframe by only having the rows with face_scores > 1.0\n    df = df[df[\'face_scores\'] > 1.0]\n\n\n    # Remove any records that contain Nulls/NaNs by checking for NaN with .isna()\n    df = df[~df[\'genders\'].isna()]\n    df = df[~df[\'second_face_scores\'].isna()]\n\n    # Cast genders to integers so that mapping can take place\n    df.genders = df.genders.astype(int)\n\n    # Iterate over all the rows in the dataframe and map each feature\n    for _, row in df.iterrows():\n      # Extract filename, gender, dob, photo_taken, \n      # face_score, second_face_score and celeb_id\n      filename = os.path.join(image_dir, _DATASET_ROOT_DIR, row[\'image_names\'][0])\n      gender = row[\'genders\']\n      dob = row[\'dobs\']\n      photo_taken = row[\'photo_taken_years\']\n      face_score = row[\'face_scores\']\n      second_face_score = row[\'second_face_scores\']\n      celeb_id = root[\'celeb_id\']\n\n      # Get the image shape\n      image_width, image_height = self._get_image_shape(filename)\n      # Normalize the bounding boxes by using the face coordinates and the image shape\n      bbox = self._get_bounding_box_values(row[\'face_locations\'][0], \n                                           image_width, image_height)\n\n      # Yield a feature dictionary \n      yield filename, {\n          "image": filename,\n          "gender": gender,\n          "dob": dob,\n          "photo_taken": photo_taken,\n          "face_location": tfds.features.BBox(\n                          ymin=min(bbox[0], 1),\n                          xmin=min(bbox[0], 1),\n                          ymax=min(bbox[0], 1),\n                          xmax=min(bbox[0], 1)\n          ),\n          "face_score": face_score,\n          "second_face_score": second_face_score,\n          "celeb_id": celeb_id\n      }')


# ## Add an Import for Registration
# 
# All subclasses of `tfds.core.DatasetBuilder` are automatically registered when their module is imported such that they can be accessed through `tfds.builder` and `tfds.load`.
# 
# If you're contributing the dataset to `tensorflow/datasets`, you must add the module import to its subdirectory's `__init__.py` (e.g. `image/__init__.py`), as shown below:

# In[57]:


get_ipython().run_cell_magic('writefile', 'tensorflow_datasets/image/__init__.py', '# coding=utf-8\n# Copyright 2019 The TensorFlow Datasets Authors.\n#\n# Licensed under the Apache License, Version 2.0 (the "License");\n# you may not use this file except in compliance with the License.\n# You may obtain a copy of the License at\n#\n#     http://www.apache.org/licenses/LICENSE-2.0\n#\n# Unless required by applicable law or agreed to in writing, software\n# distributed under the License is distributed on an "AS IS" BASIS,\n# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.\n# See the License for the specific language governing permissions and\n# limitations under the License.\n\n"""Image datasets."""\n\nfrom tensorflow_datasets.image.abstract_reasoning import AbstractReasoning\nfrom tensorflow_datasets.image.aflw2k3d import Aflw2k3d\nfrom tensorflow_datasets.image.bigearthnet import Bigearthnet\nfrom tensorflow_datasets.image.binarized_mnist import BinarizedMNIST\nfrom tensorflow_datasets.image.binary_alpha_digits import BinaryAlphaDigits\nfrom tensorflow_datasets.image.caltech import Caltech101\nfrom tensorflow_datasets.image.caltech_birds import CaltechBirds2010\nfrom tensorflow_datasets.image.cats_vs_dogs import CatsVsDogs\nfrom tensorflow_datasets.image.cbis_ddsm import CuratedBreastImagingDDSM\nfrom tensorflow_datasets.image.celeba import CelebA\nfrom tensorflow_datasets.image.celebahq import CelebAHq\nfrom tensorflow_datasets.image.chexpert import Chexpert\nfrom tensorflow_datasets.image.cifar import Cifar10\nfrom tensorflow_datasets.image.cifar import Cifar100\nfrom tensorflow_datasets.image.cifar10_corrupted import Cifar10Corrupted\nfrom tensorflow_datasets.image.clevr import CLEVR\nfrom tensorflow_datasets.image.coco import Coco\nfrom tensorflow_datasets.image.coco2014_legacy import Coco2014\nfrom tensorflow_datasets.image.coil100 import Coil100\nfrom tensorflow_datasets.image.colorectal_histology import ColorectalHistology\nfrom tensorflow_datasets.image.colorectal_histology import ColorectalHistologyLarge\nfrom tensorflow_datasets.image.cycle_gan import CycleGAN\nfrom tensorflow_datasets.image.deep_weeds import DeepWeeds\nfrom tensorflow_datasets.image.diabetic_retinopathy_detection import DiabeticRetinopathyDetection\nfrom tensorflow_datasets.image.downsampled_imagenet import DownsampledImagenet\nfrom tensorflow_datasets.image.dsprites import Dsprites\nfrom tensorflow_datasets.image.dtd import Dtd\nfrom tensorflow_datasets.image.eurosat import Eurosat\nfrom tensorflow_datasets.image.flowers import TFFlowers\nfrom tensorflow_datasets.image.food101 import Food101\nfrom tensorflow_datasets.image.horses_or_humans import HorsesOrHumans\nfrom tensorflow_datasets.image.image_folder import ImageLabelFolder\nfrom tensorflow_datasets.image.imagenet import Imagenet2012\nfrom tensorflow_datasets.image.imagenet2012_corrupted import Imagenet2012Corrupted\nfrom tensorflow_datasets.image.kitti import Kitti\nfrom tensorflow_datasets.image.lfw import LFW\nfrom tensorflow_datasets.image.lsun import Lsun\nfrom tensorflow_datasets.image.mnist import EMNIST\nfrom tensorflow_datasets.image.mnist import FashionMNIST\nfrom tensorflow_datasets.image.mnist import KMNIST\nfrom tensorflow_datasets.image.mnist import MNIST\nfrom tensorflow_datasets.image.mnist_corrupted import MNISTCorrupted\nfrom tensorflow_datasets.image.omniglot import Omniglot\nfrom tensorflow_datasets.image.open_images import OpenImagesV4\nfrom tensorflow_datasets.image.oxford_flowers102 import OxfordFlowers102\nfrom tensorflow_datasets.image.oxford_iiit_pet import OxfordIIITPet\nfrom tensorflow_datasets.image.patch_camelyon import PatchCamelyon\nfrom tensorflow_datasets.image.pet_finder import PetFinder\nfrom tensorflow_datasets.image.quickdraw import QuickdrawBitmap\nfrom tensorflow_datasets.image.resisc45 import Resisc45\nfrom tensorflow_datasets.image.rock_paper_scissors import RockPaperScissors\nfrom tensorflow_datasets.image.scene_parse_150 import SceneParse150\nfrom tensorflow_datasets.image.shapes3d import Shapes3d\nfrom tensorflow_datasets.image.smallnorb import Smallnorb\nfrom tensorflow_datasets.image.so2sat import So2sat\nfrom tensorflow_datasets.image.stanford_dogs import StanfordDogs\nfrom tensorflow_datasets.image.stanford_online_products import StanfordOnlineProducts\nfrom tensorflow_datasets.image.sun import Sun397\nfrom tensorflow_datasets.image.svhn import SvhnCropped\nfrom tensorflow_datasets.image.uc_merced import UcMerced\nfrom tensorflow_datasets.image.visual_domain_decathlon import VisualDomainDecathlon\n\n# EXERCISE: Import your dataset module here\n\n# YOUR CODE HERE\nfrom tensorflow_datasets.image.imdb_faces import ImdbFaces')


# ## URL Checksums
# 
# If you're contributing the dataset to `tensorflow/datasets`, add a checksums file for the dataset. On first download, the DownloadManager will automatically add the sizes and checksums for all downloaded URLs to that file. This ensures that on subsequent data generation, the downloaded files are as expected.

# In[58]:


get_ipython().system('touch tensorflow_datasets/url_checksums/imdb_faces.txt')


# ## Build the Dataset

# In[59]:


# EXERCISE: Fill in the name of your dataset.
# The name must be a string.
DATASET_NAME = "imdb_faces" # YOUR CODE HERE


# We then run the `download_and_prepare` script locally to build it, using the following command:
# 
# ```
# %%bash -s $DATASET_NAME
# python -m tensorflow_datasets.scripts.download_and_prepare \
#   --register_checksums \
#   --datasets=$1
# ```
# 
# **NOTE:** It may take more than 30 minutes to download the dataset and then write all the preprocessed files as TFRecords. Due to the enormous size of the data involved, we are unable to run the above script in the Coursera environment. 

# ## Load the Dataset
# 
# Once the dataset is built you can load it in the usual way, by using `tfds.load`, as shown below:
# 
# ```python
# import tensorflow_datasets as tfds
# dataset, info = tfds.load('imdb_faces', with_info=True)
# ```
# 
# **Note:** Since we couldn't build the `imdb_faces` dataset due to its size, we are unable to run the above code in the Coursera environment.

# ## Explore the Dataset
# 
# Once the dataset is loaded, you can explore it by using the following loop:
# 
# ```python
# for feature in tfds.as_numpy(dataset['train']):
#   for key, value in feature.items():
#     if key == 'image':
#       value = value.shape
#     print(key, value)
#   break
# ```
# 
# **Note:** Since we couldn't build the `imdb_faces` dataset due to its size, we are unable to run the above code in the Coursera environment.
# 
# The expected output from the code block shown above should be:
# 
# ```python
# >>>
# celeb_id 12387
# dob 722957
# face_location [1.         0.56327355 1.         1.        ]
# face_score 4.0612864
# gender 0
# image (96, 97, 3)
# photo_taken 2007
# second_face_score 3.6680346
# ```

# # Next steps for publishing
# 
# **Double-check the citation**  
# 
# It's important that DatasetInfo.citation includes a good citation for the dataset. It's hard and important work contributing a dataset to the community and we want to make it easy for dataset users to cite the work.
# 
# If the dataset's website has a specifically requested citation, use that (in BibTex format).
# 
# If the paper is on arXiv, find it there and click the bibtex link on the right-hand side.
# 
# If the paper is not on arXiv, find the paper on Google Scholar and click the double-quotation mark underneath the title and on the popup, click BibTeX.
# 
# If there is no associated paper (for example, there's just a website), you can use the BibTeX Online Editor to create a custom BibTeX entry (the drop-down menu has an Online entry type).
#   
# 
# **Add a test**   
# 
# Most datasets in TFDS should have a unit test and your reviewer may ask you to add one if you haven't already. See the testing section below.   
# **Check your code style**  
# 
# Follow the PEP 8 Python style guide, except TensorFlow uses 2 spaces instead of 4. Please conform to the Google Python Style Guide,
# 
# Most importantly, use tensorflow_datasets/oss_scripts/lint.sh to ensure your code is properly formatted. For example, to lint the image directory
# See TensorFlow code style guide for more information.
# 
# **Add release notes**
# Add the dataset to the release notes. The release note will be published for the next release.
# 
# **Send for review!**
# Send the pull request for review.
# 
# For more information, visit https://www.tensorflow.org/datasets/add_dataset

# # Submission Instructions

# In[60]:


# Now click the 'Submit Assignment' button above.


# # When you're done or would like to take a break, please run the two cells below to save your work and close the Notebook. This frees up resources for your fellow learners.

# In[61]:


get_ipython().run_cell_magic('javascript', '', '<!-- Save the notebook -->\nIPython.notebook.save_checkpoint();')


# In[ ]:


get_ipython().run_cell_magic('javascript', '', '<!-- Shutdown and close the notebook -->\nwindow.onbeforeunload = null\nwindow.close();\nIPython.notebook.session.delete();')

