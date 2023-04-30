# Chargrid-ocr (Pytorch)
![](network.png)

Implementation of 
[Chargrid-OCR: End-to-end Trainable Optical Character Recognition through Semantic Segmentation and Object Detection](https://arxiv.org/pdf/1909.04469.pdf)

It is a novel approach for optical character recognition (OCR) of printed documents. The proposed method, called Chargrid-OCR, combines instance segmentation and OCR into a single end-to-end trainable neural network. The network first segments the text regions in the document using a modified version of Mask R-CNN and then recognizes the characters in each segmented region using a convolutional neural network (CNN) with a novel Chargrid representation. The Chargrid representation is a grid-based encoding scheme that encodes each character in a grid cell and is designed to be robust to variations in character size and aspect ratio. 


