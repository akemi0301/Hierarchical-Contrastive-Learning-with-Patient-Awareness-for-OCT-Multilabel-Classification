from sklearn.manifold import TSNE
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch

colors_per_class = None

def generate_distinct_colors(num_colors):
    colors = []
    for i in range(num_colors):
        r = (i * 23) % 256
        g = (i * 47) % 256
        b = (i * 71) % 256
        color_hex = "#{:02X}{:02X}{:02X}".format(r, g, b)
        colors.append(color_hex)
    return colors


def scale_to_01_range(x):
    value_range = (np.max(x) - np.min(x))

    starts_from_zero = x - np.min(x)

    return starts_from_zero / value_range

def get_features(dataloader, model, opt, method):
    # train_loader
    global colors_per_class
    print("=============================================")
    print(f"Visualizing METHOD: {method} ...")
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    features = None
    labels = []
    index_method = -1
    if opt.dataset == 'Prime':
        csv_path_train = './final_csvs_' + str(opt.patient_split) + '/complete_prime_recovery_trex'+'/full_prime_train.csv'
        index_method = 10 if method == "BCVA" else 12 #Index of CST
    elif opt.dataset == 'Prime_TREX_DME_Fixed' or opt.dataset == 'Prime_TREX_Alpha' \
            or opt.dataset == 'Patient_Split_2_Prime_TREX' or opt.dataset == 'Patient_Split_3_Prime_TREX':
        csv_path_train = './final_csvs_' + str(opt.patient_split) +'/datasets_combined/prime_trex_compressed.csv'
        index_method = 2 if method == "BCVA" else 3
    elif opt.dataset == "Competition":
        print("Dataset competition")
        csv_path_train = './final_competition_csv/Training_Unlabeled_Clinical_Data.csv'
        index_method = 1 if method == "BCVA" else 2
    else:
        raise ValueError(opt.dataset)
    
    dataframe = pd.read_csv(csv_path_train)
    total_label = list(dataframe[method].unique())
    total_label = list(map(lambda x: str(x), total_label))

    num_colors = dataframe[method].nunique()
    distinct_colors = generate_distinct_colors(num_colors)
    colors_per_class = {key: distinct_colors[idx] for idx, key in enumerate(total_label) }


    # image_paths = []
    print('Running the model inference')
    for idx, batch in tqdm(enumerate(dataloader)):
        if isinstance(batch[0][0], str):
            images = batch[1][0].to(device)
        else: images = batch[0][0].to(device)
        labels += batch[index_method]
        
        with torch.no_grad():
            output = model.forward(images)
        current_features = output.cpu().numpy()
        if features is not None:
            features = np.concatenate((features, current_features))
        else:
            features = current_features
        if idx == int(8000 / current_features.shape[0]): break
        # print("FEATURES SHAPE: ", features.shape)
        
    return features, labels

def visualize_tsne_points(tx, ty, labels, epoch, batch, method):
    # initialize matplotlib plot
    # print(labels)
    # print(labels.tolist())
    # print("GET LABEL ENDING")
    # print([i for i, l in enumerate(labels.tolist()) if str(l) == 40])
    # print("Get ax")
    # fig = plt.figure()
    fig = plt.figure(figsize=(12, 8))
    # print("get fix")
    ax = fig.add_subplot(111)

    # for every class, we'll add a scatter plot separately
    for label in tqdm(colors_per_class):
        # find the samples of the current class in the data
        # print(labels)
        indices = [i for i, l in enumerate(labels) if str(l.tolist()) == label]
        # print("INDICE:------", indices)
        if len(indices) == 0: continue
        # extract the coordinates of the points of this class only
        current_tx = np.take(tx, indices)
        current_ty = np.take(ty, indices)
        # print(current_tx)

        ax.scatter(current_tx, current_ty, c=colors_per_class[label], label=label)

    # build a legend using the labels we set previously
    # ax.legend(loc='best')

    # finally, show the plot
    # plt.show()
    print("Save figure")
    figure = ax.get_figure()
    save_path = f'./plot/method_{method}_epoch_{epoch}_batch_{batch}_tsne.jpg'
    figure.savefig(save_path, bbox_inches='tight', pad_inches=0.1)  # save image
    print("=============================================")
    

def tsne_visualize(dataloader, model, epoch, batch, opt):
    import os
    if not os.path.exists('plot'):
        os.makedirs('plot')
    method = None
    if opt.num_methods == 1:
        method = [opt.method1.upper()]
    elif opt.num_methods == 2:
        method = [opt.method1.upper(), opt.method2.upper()]
    else:
        return
    for i in range(opt.num_methods):
        print(f"METHOD {i+1}")
        feature_tsne, labels_tsne = get_features(dataloader, model, opt, method[i])
        tsne = TSNE(n_components=2).fit_transform(feature_tsne)
        tx, ty = tsne[:, 0], tsne[:, 1]
        tx = scale_to_01_range(tx)
        ty = scale_to_01_range(ty)
        visualize_tsne_points(tx, ty, labels_tsne, epoch, batch, method[i])


# 2 method thì visualize 2 lần
