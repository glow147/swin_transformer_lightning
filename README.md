# swin_transformer_lightning

<a href=https://github.com/microsoft/Swin-Transformer>Swin transformer Official Github</a>

Swin_transformer reimplements using pytorch_lightning and albumentations.

After 300 epochs training, It almost acquired <b>78%</b> accuracy from validation.

This result is lower <b>4%</b> than original code.

## Cause

1. I prepared resized training image ( original size -> 224, 224 ) for speed up training, valid dataset also same.
However, when we store to jpeg or the other format using PIL Image, it loose the information even we set quailty=100, subsample=0.
(I am not sure in cv2 has same issue.)

2. When I do validation task, I only apply normalize. but in official code, firstly, they resized image to 256 ( in 224 image size situation 
eq : int(256 / 224 * output_image_size)). Secondly, they did CenterCrop for focusing center.
This process results in a difference of about 2% in accuracy.

3. torch transform resize and albumentation resize have different value. Even I used same image and same interpolation.

4. setting weight decay to bias, It drop accuracy about 2%

## Solve

1. Use original training dataset.
2. Done.
3. I only did timm library. You can see in Figure 1.
4. Done.

## Result

Max Acc : 81.121

![ACC](https://user-images.githubusercontent.com/26201768/234764443-379fb38d-0bd5-4ef8-ad98-42956afc6a4d.png)Figure 1. Accuracy
