# Implementation of Consistency Model
# https://arxiv.org/pdf/2303.01469.pdf


from typing import List
from tqdm import tqdm
import math
from torch.utils.data import ConcatDataset

import torch
from torch.utils.data import DataLoader

from torchvision.datasets import MNIST, CIFAR10, OxfordIIITPet
from torchvision import transforms
from torchvision.utils import save_image, make_grid

from consistency_models import ConsistencyModel, kerras_boundaries

from tqdm import tqdm
from sklearn.manifold import TSNE

from matplotlib import pyplot as plt
import numpy as np
import random 
import json


def scale_to_01_range(x):
    # compute the distribution range
    value_range = (np.max(x) - np.min(x))
 
    # move the distribution so that it starts from zero
    # by extracting the minimal value from all its values
    starts_from_zero = x - np.min(x)
 
    # make the distribution fit [0; 1] by dividing by its range
    return starts_from_zero / value_range


def calculate_gram_matrix(features):
    features = features.view(features.size(0),-1)
    G = torch.ones((features.shape[0],features.shape[0]), device=features.device)
    for i in range(features.shape[0]):
        for j in range(features.shape[0]):
            G[i,j] = torch.dot(features[i],features[j])/(torch.norm(features[i])*torch.norm(features[j]))
    return G

def calculate_effective_rank(G):
    U, S, Vh = torch.linalg.svd(G)
    S = S/torch.sum(S)
    sum = 0.0
    for i in range(len(S)):
        sum += S[i]*torch.log(S[i])
    return -1.0*(sum.item())
        
    
 



def sample(
    n_sample: int = 5,
    device="cuda:0",
    n_channels=1,
    name="mnist",
    weight_path = "./ct_mnist.pth",
    im_dim =32,
):
    model = ConsistencyModel(n_channels, D=256)
    model.to(device)

    model.load_state_dict(torch.load(weight_path))

    model.eval()

    arr_list = []


    for sample in tqdm(range(1, n_sample)):
        

        with torch.no_grad():
            # Sample 5 Steps
            x = torch.randn((1,n_channels,im_dim,im_dim)).to(device=device)
            xh = model.sample(
                x * 80.0,
                list(reversed([5.0, 10.0, 20.0, 40.0, 80.0])),
            )
            xh = (xh * 0.5 + 0.5).clamp(0, 1)
            grid = make_grid(xh, nrow=1)
            save_image(grid, f"./new_samples/ct_{name}_sample_5step_{sample}.png")

            # Sample 2 Steps
            xh = model.sample(
                torch.randn_like(x).to(device=device) * 80.0,
                list(reversed([2.0, 80.0])),
            )
            xh = (xh * 0.5 + 0.5).clamp(0, 1)
            grid = make_grid(xh, nrow=1)
            save_image(grid, f"./new_samples/ct_{name}_sample_2step_{sample}.png")

            arr_list.append(x)
    
    list_seeds = torch.cat(arr_list, dim=0)

    count = 0


    with torch.no_grad():
        lst_dict = {'0':[],'1':[], '2':[],'3':[],'4':[],'5':[]}
        for i in tqdm(range(len(list_seeds))):
            for j in range(len(list_seeds)):

                if i != j:
                    int_list = torch.cat([0.1*k*list_seeds[i].unsqueeze(0)+0.1*(10-k)*list_seeds[j].unsqueeze(0) for k in range(11)])
                    lst_dict['0'].append(int_list.view(-1,3*32*32).cpu())
                    xh,list_features = model.sample_intermediates(
                        int_list * 80.0,
                        list(reversed([5.0, 10.0, 20.0, 40.0, 80.0])),
                    )

                    xh = (xh * 0.5 + 0.5).clamp(0, 1)
                    grid = make_grid(xh, nrow=11)
                    save_image(grid, f"./new_interpol_samples/ct_{name}_sample_5step_{count}.png")
                    count += 1
                    for i in range(5):
                        lst_dict[str(i+1)].append(list_features[i].unsqueeze(0).view(-1,3*32*32).cpu())
        for i in range(6):
            lst_dict[str(i)] = torch.cat(lst_dict[str(i)], dim=0)
            tsne = TSNE(n_components=2).fit_transform(lst_dict[str(i)])
            tx = tsne[:, 0]
            ty = tsne[:, 1]
 
            tx = scale_to_01_range(tx)
            ty = scale_to_01_range(ty)

            plt.scatter(tx, ty)

            plt.xlabel('t-SNE 1')
            plt.ylabel('t-SNE 2')
            plt.title(f't-SNE visualization of the features of the {i+1} timestep towards image generation')
            plt.savefig(f'./new_tsne/tsne_samples_{i+1}.png')





        
def inverted_samples_class_random( n_sample: int = 5,
    device="cuda:0",
    n_channels=1,
    name="mnist",
    weight_path = "./ct_mnist.pth",
    im_dim =32,):
    model = ConsistencyModel(n_channels, D=256)
    model.to(device)

    model.load_state_dict(torch.load(weight_path))

    model.eval()

    arr_list = []

    test_dataset = CIFAR10(root='./data', train=False, transform=transforms.Compose([transforms.Resize((im_dim,im_dim)), transforms.ToTensor()]), download=True)

    classes = test_dataset.classes
    class_count = {}
    for item, index in tqdm(test_dataset):
        label = classes[index]
        if label not in class_count:
            class_count[label] = [item]
        class_count[label].append(item)

    
    N = math.ceil(math.sqrt((400 * (150**2 - 4) / 1000) + 4) - 1) + 1
    boundaries = kerras_boundaries(7.0, 0.002, N, 80.0).to(device)




    for sample in tqdm(range(1, n_sample)):

        label = random.choice(list(class_count.keys()))
        

        with torch.no_grad():
            # Sample 5 Steps
            x = random.choice(class_count[label]).unsqueeze(0).to(device=device)
            z = torch.randn_like(x).to(device=device)
            t = 80.0*torch.ones((x.shape[0], 1), device=device)
            t_0 = boundaries[80].view(-1,1)
            x = x + z*t_0[:,:,None,None]
            xh = model.sample(
                x.to(device=device) * 80.0,
                list(reversed([5.0, 10.0, 20.0, 40.0, 80.0])),
            )
            xh = (xh * 0.5 + 0.5).clamp(0, 1)
            grid = make_grid(xh, nrow=1)
            save_image(grid, f"./samples_class_random/ct_{name}_sample_5step_{sample}.png")

            # Sample 2 Steps
            xh = model.sample(
                torch.randn_like(x).to(device=device) * 80.0,
                list(reversed([2.0, 80.0])),
            )
            xh = (xh * 0.5 + 0.5).clamp(0, 1)
            grid = make_grid(xh, nrow=1)
            save_image(grid, f"./samples_class_random/ct_{name}_sample_2step_{sample}.png")

            arr_list.append(x)
    
    list_seeds = torch.cat(arr_list, dim=0)



    with torch.no_grad():
        count = 0
        for i in tqdm(range(len(list_seeds))):
            for j in range(len(list_seeds)):

                if i != j:
                    int_list = torch.cat([0.1*k*list_seeds[i].unsqueeze(0)+0.1*(10-k)*list_seeds[j].unsqueeze(0) for k in range(11)])
                    xh = model.sample(
                        int_list * 80.0,
                        list(reversed([5.0, 10.0, 20.0, 40.0, 80.0])),
                    )

                    xh = (xh * 0.5 + 0.5).clamp(0, 1)
                    grid = make_grid(xh, nrow=11)
                    save_image(grid, f"./interpol_samples_class_random/ct_{name}_sample_5step_{count}.png")
                    count += 1
       








            # save model

def sample_proximity(
    n_sample: int = 5,
    device="cuda:0",
    n_channels=1,
    name="mnist",
    weight_path = "./ct_mnist.pth",
    im_dim =32,
):
    model = ConsistencyModel(n_channels, D=256)
    model.to(device)

    model.load_state_dict(torch.load(weight_path))

    model.eval()

    arr_list = []

    x = torch.randn((1,n_channels,im_dim,im_dim)).to(device=device)






    for sample in tqdm(range(1, n_sample)):
        

        with torch.no_grad():
            # Sample 5 Steps
            x_add = sample*0.01*torch.randn_like(x).to(device=device)
            x_hat = x + x_add/torch.norm(x_add)
            xh = model.sample(
                x_hat * 80.0,
                list(reversed([5.0, 10.0, 20.0, 40.0, 80.0])),
            )
            xh = (xh * 0.5 + 0.5).clamp(0, 1)
            grid = make_grid(xh, nrow=1)
            save_image(grid, f"./close_samples/ct_{name}_sample_5step_{sample}.png")

            arr_list.append(x_hat)
    
    list_seeds = torch.cat(arr_list, dim=0)

    count = 0


    with torch.no_grad():
        for i in tqdm(range(len(list_seeds))):
            for j in range(len(list_seeds)):

                if i != j:
                    int_list = torch.cat([0.1*k*list_seeds[i].unsqueeze(0)+0.1*(10-k)*list_seeds[j].unsqueeze(0) for k in range(11)])
                    xh = model.sample(
                        int_list * 80.0,
                        list(reversed([5.0, 10.0, 20.0, 40.0, 80.0])),
                    )

                    xh = (xh * 0.5 + 0.5).clamp(0, 1)
                    grid = make_grid(xh, nrow=11)
                    save_image(grid, f"./close_interpol_samples/ct_{name}_sample_5step_{count}.png")
                    count += 1

def effective_rank_samples( n_sample: int = 5,
    device="cuda:0",
    n_channels=1,
    name="mnist",
    weight_path = "./ct_mnist.pth",
    im_dim =32,):
    model = ConsistencyModel(n_channels, D=256)
    model.to(device)

    model.load_state_dict(torch.load(weight_path))

    model.eval()

    arr_list = []
    feat_list = []

    test_dataset = CIFAR10(root='./data', train=False, transform=transforms.Compose([transforms.Resize((im_dim,im_dim)), transforms.ToTensor()]), download=True)
    test_dataset_3 = MNIST(root='./data', train=False, transform=transforms.Compose([transforms.Resize((im_dim,im_dim)),transforms.Grayscale(num_output_channels=3), transforms.ToTensor()]), download=True)
    test_dataset_2 = OxfordIIITPet(root='./data', split='test', transform=transforms.Compose([transforms.Resize((im_dim,im_dim)),transforms.Grayscale(num_output_channels=3), transforms.ToTensor()]), download=True)


    classes = test_dataset_2.classes
    class_count = {}
    for item, index in tqdm(test_dataset_2):
        label = classes[index]
        if label not in class_count:
            class_count[label] = [item]
        class_count[label].append(item)

    
    N = math.ceil(math.sqrt((400 * (150**2 - 4) / 1000) + 4) - 1) + 1
    boundaries = kerras_boundaries(7.0, 0.002, N, 80.0).to(device)




    for sample in tqdm(range(1, n_sample)):

        label = random.choice(list(class_count.keys()))
        

        with torch.no_grad():
            # Sample 5 Steps
            x = random.choice(class_count[label]).unsqueeze(0).to(device=device)
            z = torch.randn_like(x).to(device=device)
            t = 80.0*torch.ones((x.shape[0], 1), device=device)
            t_0 = boundaries[80].view(-1,1)
            x = x + z*t_0[:,:,None,None]
            xh,fh = model.sample_feature_intermediates(
                x.to(device=device) * 80.0,
                list(reversed([5.0, 10.0, 20.0, 40.0, 80.0])),
            )
            xh = (xh * 0.5 + 0.5).clamp(0, 1)
            grid = make_grid(xh, nrow=1)
            save_image(grid, f"./samples_class_random/ct_{name}_sample_5step_{sample}.png")

            # Sample 2 Steps
            xh = model.sample(
                torch.randn_like(x).to(device=device) * 80.0,
                list(reversed([2.0, 80.0])),
            )
            xh = (xh * 0.5 + 0.5).clamp(0, 1)
            grid = make_grid(xh, nrow=1)
            save_image(grid, f"./samples_class_random/ct_{name}_sample_2step_{sample}.png")

            arr_list.append(x)
            feat_list.append(fh)
    
    list_seeds = torch.cat(arr_list, dim=0)

    feat_list_2 = []

    feat_list_3 = []



    with torch.no_grad():
        count = 0
        for i in tqdm(range(len(list_seeds))):
            for j in range(len(list_seeds)):

                if i != j:
                    int_list = torch.cat([0.1*k*list_seeds[i].unsqueeze(0)+0.1*(10-k)*list_seeds[j].unsqueeze(0) for k in range(11)])
                    #int_list = torch.cat([0.1*k*torch.randn_like(list_seeds[i]).unsqueeze(0)+0.1*(10-k)*torch.randn_like(list_seeds[j]).unsqueeze(0) for k in range(11)])
                    xh,fh = model.sample_feature_intermediates(
                        int_list * 80.0,
                        list(reversed([5.0, 10.0, 20.0, 40.0, 80.0])),
                    )

                    xh = (xh * 0.5 + 0.5).clamp(0, 1)
                    grid = make_grid(xh, nrow=11)
                    save_image(grid, f"./interpol_samples_class_random/ct_{name}_sample_5step_{count}.png")
                    count += 1
                    feat_list_2.append(fh)

    
    with torch.no_grad():
        count = 0
        for i in tqdm(range(len(list_seeds))):
            for j in range(len(list_seeds)):

                if i != j:
                    #int_list = torch.cat([0.1*k*list_seeds[i].unsqueeze(0)+0.1*(10-k)*list_seeds[j].unsqueeze(0) for k in range(11)])
                    int_list = torch.cat([0.1*k*torch.randn_like(list_seeds[i]).unsqueeze(0)+0.1*(10-k)*torch.randn_like(list_seeds[j]).unsqueeze(0) for k in range(11)])
                    xh,fh = model.sample_feature_intermediates(
                        int_list * 80.0,
                        list(reversed([5.0, 10.0, 20.0, 40.0, 80.0])),
                    )

                    xh = (xh * 0.5 + 0.5).clamp(0, 1)
                    grid = make_grid(xh, nrow=11)
                    count += 1
                    feat_list_3.append(fh)



    feat_dict = {'0':[],'1':[], '2':[],'3':[],'4':[]}
    feat_dict_2 = {'0':[],'1':[], '2':[],'3':[],'4':[]}

    feat_dict_3 = {'0':[],'1':[], '2':[],'3':[],'4':[]}




  

    for i in range(5):
        feat_dict[str(i)] = [feat_list[j][i].unsqueeze(0) for j in range(len(feat_list))]
        feat_dict_2[str(i)] = [feat_list_2[j][i].unsqueeze(0) for j in range(len(feat_list_2))]
        feat_dict_3[str(i)] = [feat_list_3[j][i].unsqueeze(0) for j in range(len(feat_list_3))]

    rank_dict = {'80':0.0,'40':0.0, '20':0.0,'10':0.0,'5':0.0}
    rank_dict_2 = {'80':0.0,'40':0.0, '20':0.0,'10':0.0,'5':0.0}
    rank_dict_3 = {'80':0.0,'40':0.0, '20':0.0,'10':0.0,'5':0.0}

    for i in range(5):
        feat_dict[str(i)] = torch.cat(feat_dict[str(i)], dim=0)
        feat_dict_2[str(i)] = torch.cat(feat_dict_2[str(i)], dim=0)
        feat_dict_3[str(i)] = torch.cat(feat_dict_3[str(i)], dim=0)

        
        G = calculate_gram_matrix(feat_dict[str(i)])
        rank_dict[list(rank_dict.keys())[i]] = calculate_effective_rank(G)
        G = calculate_gram_matrix(feat_dict_2[str(i)])
        rank_dict_2[list(rank_dict_2.keys())[i]] = calculate_effective_rank(G)
        G = calculate_gram_matrix(feat_dict_3[str(i)])
        rank_dict_3[list(rank_dict_3.keys())[i]] = calculate_effective_rank(G)
    x = rank_dict.keys()
    y = rank_dict.values()

    plt.scatter(x, y,label="MNIST Dataset")
    
    plt.xlabel('Timesteps')
    plt.ylabel('Effective Rank')
    plt.title(f'Effective Rank of the features of the timesteps towards image generation')
    plt.savefig(f'./effective_rank.png')

    x_2 = rank_dict_2.keys()
    y_2 = rank_dict_2.values()

    plt.scatter(x_2, y_2,label="Interpolated Samples")

    x_3 = rank_dict_3.keys()
    y_3 = rank_dict_3.values()

    plt.scatter(x_3, y_3,label="Random Samples")

    
    plt.xlabel('Timesteps')
    plt.ylabel('Effective Rank')
    plt.title(f'Effective Rank of the samples')
    plt.legend()

    plt.savefig(f'./effective_rank_intermediates.png')


    details = {'name':'oxford','weight_path': f'{weight_path}', 'samples' : rank_dict, 'interpolated_samples': rank_dict_2, 'random_samples': rank_dict_3}  
  
    with open('./result.txt', 'w') as convert_file:
     convert_file.write(json.dumps(details))




    



                   



if __name__ == "__main__":
    # train()
    effective_rank_samples(n_sample=10, n_channels=3, name="simp_bias",weight_path="./ct_mnist_oxf.pth", im_dim=32)
