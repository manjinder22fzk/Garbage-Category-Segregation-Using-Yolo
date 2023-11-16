# Garbage-Category-Segregation-Using-Yolo

## Introduction

In 2022, the government of British Columbia came up with the idea of Automatic Rubbish Segregation.  The aim of the project was to identify the waste products into "Organic" and "Non-Organic" so that the organic waste can be segregated from the non-organic ones in order to reduce the manual cost and effort of doing this and make the  whole  process faster. I was selected along with a team of 3 to work on this project as a research team from Langara College.

## Dataset 

Since no dataset was provided by the BC Government until we built a prototype, we had to manually collect the data by clicking pictures of the garbage from the roads and dustbin boxes of possible households. We collected around 6000 images of data over a span of 1 month . We wanted to have Balanced Dataset so around 3100 images were of Organic waste and 2900 images were of Non-Organic Waste .

The next was to label the data using LabelMe Software. We labeled the images using multiple rectangular boxes on each image. 

## Choosing the Model

We tried out  multiple CNN Models including : 
  1. Resnet
  2. Faster R-CNN
  3. Detectron
  4. Yolo V5, Yolo V7, Yolo V8
  5. SSD
  6. Alexnet

We obtained a higher accuracy using Yolo V7 but the model compilation was faster using Yolo V8 so there was a trade-off between speed and accuracy so we went ahead with Yolo V8. 

## Performance Metrics

Since it's a complex job to measure the performance of a model, we used the below metrics instead of using 1 : 
  1. Precision and Recall
  2. F1 Score
  3. Intersection Over Union
  4. Receiver Operating Characteristic (ROC) Curve and Area Under Curve (AUC)

We we able to achieve an F1 Score of 0.7. 

## Achievements 

We presented our solution at the Langara Research Symposium 2023 and won the first prize. Our model was finally selected by the college authorities to submit to BC government.

Note : I have not uploaded the best weights and dataset on github due to their heavy size. If you want the best trained models along with dataset check it out here - https://drive.google.com/drive/folders/17vJ72aru4Ba82HT6eHapGAR6LiOIOA5o


