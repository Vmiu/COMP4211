# COMP4211FP -- A Image Classifier for GenAI model used by a fake image <br />
Group members: LIU Muyuan, Miu Victor
## Background
With the rapid development of generative artificial intelligence models, an increasing number of AI models capable of generating various types of images and artworks are being deployed. As a result, more and more generated images appear on the internet. This phenomenon has both negative and positive effects. For instance, The spread of misinformation with some fake images can easily mislead their viewers. On the other hand, ordinary individuals with the assistance of an image generation AI model can create their own beautiful images and artworks easily. Our aim is to build a CNN model that can classify whether an image is real or fake, and identify which image GenAI model is used to generate this image. Ordinary individuals can use this model to distinguish misinformation online. While companies developing image GenAI models can use our model to determine if their training process has some biases, resulting in the generated images can be easily recognized because of some distinct styles.

## Problem Description
Our model will be trained on the AI-generated images from popular GenAI models, including Midjourney, BigGAN, Vector Quantized Diffusion Model, Stable Diffusion, Wukong, GLIDE, Ablated Diffusion Model, as well as the images from the real world. In essence, the model tries to solve a classification problem, where it will label each image with its origination (real world/AI model). The model allows users to upload photos into the model, and it will identify the most possible source they come from. 

## Dataset Description
Our dataset comes from a subset of GenImage[2][3], which is a million-scale Benchmark for detecting AI-Generated Images. GenImage contains 1000 classes images which are using the same classes in ImageNet, also a huge dataset contains real images. It includes data collected from some trending image generation models, for example Stable Diffusion and Midjourney. 

For the convenience of division of labor, we build our own train and test dataset using GenImage. But the number of classes are the same which have 8 classes corresponding to {adm,biggan,glide,mid journey,nature,sdv5,vqdm,wukong}

## Machine Learning tasks performed on the dataset:
In the current landscape, the development of various GenAI models for image generation has triggered users’ interest in their performance. Our objective is to construct a model capable of classifying a provided image to its originating GenAI. In essence, we are addressing a categorical classification challenge, where the goal is to categorize the input image into one of eight classes, each representing a different GenAI model or a real image.  The result will assist users in refining their selection of GenAI products, or uncovering the similarity of different models and provide some insights for the GenAI developers.



References：<br />
[1] https://www.sciencedirect.com/science/article/pii/S1738573319308587 <br />
[2] https://www.kaggle.com/datasets/yangsangtai/tiny-genimage?select=imagenet_ai_0424_wukong <br />
[3] https://github.com/GenImage-Dataset/GenImage?tab=readme-ov-file <br />
[4] Hendrycks, D., & Gimpel, K. (2016). Gaussian error linear units (gelus). arXiv preprint arXiv:1606.08415.
