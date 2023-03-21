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
