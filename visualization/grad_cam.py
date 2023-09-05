import argparse
import cv2
import numpy as np
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from pytorch_grad_cam import GradCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM, \
    FullGrad

from pytorch_grad_cam.utils.image import show_cam_on_image, \
    preprocess_image
from pytorch_grad_cam.ablation_layer import AblationLayerVit

def reshape_transform_swin(tensor, height=12, width=12):
    # print("before = ", tensor.shape) # (1, 144, 1024)
    result = tensor.reshape(tensor.size(0),
                            height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    # print('after = ', result.shape)

    return result # (1, 1024, 12, 12) match (384, 384)

    
def reshape_transform_trans_encoder(tensor):
    # print('before = ', tensor.shape) # (144, 1, 2048)
    tensor = tensor.permute(1, 0, 2) #  (1, 144, 2048)
    result = reshape_transform_swin(tensor) # (1, 2048, 12, 12)

    return result

# def reshape_transform_trans_decoder(tensor):
#     print('before = ', tensor.shape) # (6, 1, 2048)
#     tensor = tensor.permute(1, 0, 2) #  (1, 6, 2048)
#     # result = tensor.permute(0, 2, 1) # (1, 2048, 6)

#     result = tensor.view(1, 1, 2048, 6)

#     return result
    
methods = \
        {"gradcam": GradCAM,
         "scorecam": ScoreCAM,
         "gradcam++": GradCAMPlusPlus,
         "ablationcam": AblationCAM,
         "xgradcam": XGradCAM,
         "eigencam": EigenCAM,
         "eigengradcam": EigenGradCAM,
         "layercam": LayerCAM,
         "fullgrad": FullGrad}

def visualize(method, model, target_layers, img_paths, epoch, idx, save_folder, idx_check_layer):
    model.eval()

    # if args.use_cuda:
    model = model.cuda()
    if idx_check_layer == 1:
        if method == "ablationcam":
            cam = methods[method](model=model,
                                    target_layers=target_layers,
                                    use_cuda=True,
                                    reshape_transform=reshape_transform_swin,
                                    ablation_layer=AblationLayerVit())
        else:
            cam = methods[method](model=model,
                                    target_layers=target_layers,
                                    use_cuda=True,
                                    reshape_transform=reshape_transform_swin)
    elif idx_check_layer == 2:
        if method == "ablationcam":
            cam = methods[method](model=model,
                                    target_layers=target_layers,
                                    use_cuda=True,
                                    reshape_transform=reshape_transform_trans_encoder,
                                    ablation_layer=AblationLayerVit())
        else:
            cam = methods[method](model=model,
                                    target_layers=target_layers,
                                    use_cuda=True,
                                    reshape_transform=reshape_transform_trans_encoder
                                    )
    # else: # idx_check_layer == 3:
    #     if method == "ablationcam":
    #         cam = methods[method](model=model,
    #                                 target_layers=target_layers,
    #                                 use_cuda=True,
    #                                 reshape_transform=reshape_transform_trans_decoder,
    #                                 ablation_layer=AblationLayerVit())
    #     else:
    #         cam = methods[method](model=model,
    #                                 target_layers=target_layers,
    #                                 use_cuda=True,
    #                                 reshape_transform=reshape_transform_trans_decoder
    #                                 )
            
    plt.figure(figsize=(15, 15))
    i = 0
    for image_path in img_paths:
      # image_path = os.path.join(img_paths, image_file)
      # print(image_path)

      rgb_img = cv2.imread(image_path, 1)[:, :, ::-1]
      rgb_img = cv2.resize(rgb_img, (384, 384)) 
      rgb_img = np.float32(rgb_img) / 255
      input_tensor = preprocess_image(rgb_img, mean=[0.5, 0.5, 0.5],
                                      std=[0.5, 0.5, 0.5])
      
    #   print('img shape = ', input_tensor.shape)

    #   image = Image.open(image_path).convert("L")
    #   image = np.array(image)
    #   image = Image.fromarray(image)
    #   print(image.shape)
      
    
      # AblationCAM and ScoreCAM have batched implementations.
      # You can override the internal batch size for faster computation.
      cam.batch_size = 28

      grayscale_cam = cam(input_tensor=input_tensor,
                          targets=None,
                          eigen_smooth='store_true',
                          aug_smooth='store_true')
      
      # Here grayscale_cam has only one image in the batch
      grayscale_cam = grayscale_cam[0, :]

      cam_image = show_cam_on_image(rgb_img, grayscale_cam)
      ax = plt.subplot(8, 3, i +1)
      i+=1
      plt.imshow(cv2.cvtColor(cam_image, cv2.COLOR_BGR2RGB))

      print(image_path)
      e = image_path.split('/')[5:]
      name_img = '/'.join(e).split('/')[2:]
      name_img = '/'.join(name_img)
      print(name_img)
      plt.title(name_img)
      plt.axis("off")

    if idx_check_layer == 1:  
        save_path = f'visualize_{method}_epoch_{epoch}_idx_{idx}_swin.jpg' 
    elif idx_check_layer == 2:
        save_path =  f'visualize_{method}_epoch_{epoch}_idx_{idx}_transEncoder.jpg' 
    # elif idx_check_layer == 3:
    #     save_path = f'visualize_{method}_epoch_{epoch}_idx_{idx}_transDecoder_1.jpg' 
    # else:
    #     save_path = f'visualize_{method}_epoch_{epoch}_idx_{idx}_transDecoder_2.jpg' 

    save_img_visualize = os.path.join(save_folder, save_path)
    if not os.path.isdir(save_folder):
        os.makedirs(save_folder)
    figure = ax.get_figure()
    figure.savefig(save_img_visualize, bbox_inches='tight', pad_inches=0.1)  # save image

    plt.tight_layout()