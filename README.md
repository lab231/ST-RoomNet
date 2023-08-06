# ST-RoomNet: Learning Room Layout Estimation From Single Image Through Unsupervised Spatial Transformations

This is the official implementation of ST-RoomNet: https://openaccess.thecvf.com/content/CVPR2023W/VOCVALC/html/Ibrahem_ST-RoomNet_Learning_Room_Layout_Estimation_From_Single_Image_Through_Unsupervised_CVPRW_2023_paper.html

The spatial transformer module is based on this repo: https://github.com/dantkz/spatial-transformer-tensorflow

We modified the spatial transformer module to work in Tensorflow2.x and added more features such as nearest neighbor interpolation in addition to the original bilinear and bicubic interpolations.

requirements:

opencv 4.4.1

tensorflow 2.9.1

If you use this code, please cite the paper as follows:

@InProceedings{Ibrahem_2023_CVPR,<br>
    author    = {Ibrahem, Hatem and Salem, Ahmed and Kang, Hyun-Soo},<br>
    title     = {ST-RoomNet: Learning Room Layout Estimation From Single Image Through Unsupervised Spatial Transformations},<br>
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},<br>
    month     = {June},<br>
    year      = {2023},<br>
    pages     = {3375-3383}<br>
}
