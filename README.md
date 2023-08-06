# ST-RoomNet

This is the official implementation of ST-RoomNet: https://openaccess.thecvf.com/content/CVPR2023W/VOCVALC/html/Ibrahem_ST-RoomNet_Learning_Room_Layout_Estimation_From_Single_Image_Through_Unsupervised_CVPRW_2023_paper.html

The spatial transformer module is based on this repo: https://github.com/dantkz/spatial-transformer-tensorflow

We modified the spatial transformer module to work in Tensorflow2.x and added more features such as nearest neighbor interpolation in addition to the original bilinear and bicubic interpolations.

requirements:

opencv 4.4.1

tensorflow 2.9.1

If you use this code, please cite the paper as follows:

Ibrahem, H., Salem, A., & Kang, H. S. (2023). ST-RoomNet: Learning Room Layout Estimation From Single Image Through Unsupervised Spatial Transformations. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 3375-3383).
