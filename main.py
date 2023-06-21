# Implementation of Consistency Model
# https://arxiv.org/pdf/2303.01469.pdf


from typing import List
from tqdm import tqdm
import math

import torch
from torch.utils.data import DataLoader,ConcatDataset,Subset

from torchvision.datasets import MNIST, CIFAR10,OxfordIIITPet
from torchvision import transforms
from torchvision.utils import save_image, make_grid

from consistency_models import ConsistencyModel, kerras_boundaries


def mnist_dl():
    tf = transforms.Compose(
        [
            transforms.Pad(2),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5)),
        ]
    )

    dataset = MNIST(
        "./data",
        train=True,
        download=True,
        transform=tf,
    )

    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=20)

    return dataloader


def cifar10_dl():
    tf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    dataset = CIFAR10(
        "./data",
        train=True,
        download=True,
        transform=tf,
    )

    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=20)

    return dataloader

def oxfordiiitpet_dl():
    tf = transforms.Compose(
        [
            transforms.Resize((64,64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5,0.5,0.5)),

        ]
    )

    dataset = OxfordIIITPet('./data', split='trainval', download=True, transform=tf)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)

    return dataloader


def cifar_mnist_dl(nsamples=5000,use_all=True):
    tf_1 = transforms.Compose(
        [
            transforms.Resize((32,32)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
        ]
    )

    tf_2 = transforms.Compose(
        [
            transforms.Resize((32,32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
        ]
    )

    dataset = CIFAR10(
        "./data",
        train=True,
        download=True,
        transform=tf_2,
    )

    dataset_2 = MNIST(
        "./data",
        train=True,
        download=True,
        transform=tf_1,
    )

    if use_all == False:
        dataset_2 = Subset(dataset_2, list(range(nsamples)))




    dataset_act = ConcatDataset([dataset, dataset_2])

    dataloader = DataLoader(dataset_act, batch_size=16, shuffle=True, num_workers=0)

    return dataloader


def cifar_oxfordiiitpet_dl(n_samples=7000):

    """
    OxfordIIITPet has only 7000 samples!
    
    """
    tf_1 = transforms.Compose(
        [
            transforms.Resize((32,32)),
            #transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
        ]
    )

    tf_2 = transforms.Compose(
        [
            transforms.Resize((32,32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
        ]
    )

    dataset = CIFAR10(
        "./data",
        train=True,
        download=True,
        transform=tf_2,
    )

    dataset_2 = OxfordIIITPet(
        "./data",
        split='trainval',
        download=True,
        transform=tf_1,
    )

    dataset = Subset(dataset, list(range(n_samples)))





    dataset_act = ConcatDataset([dataset, dataset_2])

    dataloader = DataLoader(dataset_act, batch_size=128, shuffle=True, num_workers=0)

    return dataloader

def mnist_oxfordiiitpet_dl(n_samples=7000):
    tf_1 = transforms.Compose(
        [
            transforms.Resize((32,32)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
        ]
    )

    tf_2 = transforms.Compose(
        [
            transforms.Resize((32,32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
        ]
    )

    dataset = MNIST(
        "./data",
        train=True,
        download=True,
        transform=tf_1,
    )

    dataset_2 = OxfordIIITPet(
        "./data",
        split='trainval',
        download=True,
        transform=tf_2,
    )

    dataset = Subset(dataset, list(range(n_samples)))





    dataset_act = ConcatDataset([dataset, dataset_2])

    dataloader = DataLoader(dataset_act, batch_size=128, shuffle=True, num_workers=0)

    return dataloader

def cifar_nooverlap_oxfordiiitpet_dl(n_samples=7000):

    tf_1 = transforms.Compose(
        [
            transforms.Resize((32,32)),
            #transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
        ]
    )

    tf_2 = transforms.Compose(
        [
            transforms.Resize((32,32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
        ]
    )

    dataset = CIFAR10(
        "./data",
        train=True,
        download=True,
        transform=tf_2,
    )

    dataset_2 = OxfordIIITPet(
        "./data",
        split='trainval',
        download=True,
        transform=tf_1,
    )

    #dataset = Subset(dataset, list(range(n_samples)))

    classes = dataset.classes
    tensor_list = []
    index_list = []
    for item, index in tqdm(dataset):
        label = classes[index]
        if label == "dog" or label == "cat" or label == "bird" or label == "horse" or label == "frog" or label == "deer":
            continue
        else:
            tensor_list.append(item.unsqueeze(0))
            index_list.append(index.unsqueeze(0))

    data = torch.cat(tensor_list, dim=0)
    ids = torch.cat(index_list, dim=0)

    dataset = torch.utils.data.TensorDataset(data, ids)

    dataset = Subset(dataset, list(range(n_samples)))




    

    





    dataset_act = ConcatDataset([dataset, dataset_2])

    dataloader = DataLoader(dataset_act, batch_size=128, shuffle=True, num_workers=0)

    return dataloader



def train(
    n_epoch: int = 100,
    device="cuda:0",
    dataloader=mnist_dl(),
    n_channels=1,
    name="mnist",
):
    model = ConsistencyModel(n_channels, D=256)
    model.to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Define \theta_{-}, which is EMA of the params
    ema_model = ConsistencyModel(n_channels, D=256)
    ema_model.to(device)
    ema_model.load_state_dict(model.state_dict())

    for epoch in range(1, n_epoch):
        N = math.ceil(math.sqrt((epoch * (150**2 - 4) / n_epoch) + 4) - 1) + 1
        boundaries = kerras_boundaries(7.0, 0.002, N, 80.0).to(device)

        pbar = tqdm(dataloader)
        loss_ema = None
        model.train()
        for x, _ in pbar:
            optim.zero_grad()
            x = x.to(device)

            z = torch.randn_like(x)
            t = torch.randint(0, N - 1, (x.shape[0], 1), device=device)
            t_0 = boundaries[t]
            t_1 = boundaries[t + 1]

            loss = model.loss(x, z, t_0, t_1, ema_model=ema_model)

            loss.backward()
            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.9 * loss_ema + 0.1 * loss.item()

            optim.step()
            with torch.no_grad():
                mu = math.exp(2 * math.log(0.95) / N)
                # update \theta_{-}
                for p, ema_p in zip(model.parameters(), ema_model.parameters()):
                    ema_p.mul_(mu).add_(p, alpha=1 - mu)

            pbar.set_description(f"loss: {loss_ema:.10f}, mu: {mu:.10f}")

        model.eval()
        with torch.no_grad():
            # Sample 5 Steps
            xh = model.sample(
                torch.randn_like(x).to(device=device) * 80.0,
                list(reversed([5.0, 10.0, 20.0, 40.0, 80.0])),
            )
            xh = (xh * 0.5 + 0.5).clamp(0, 1)
            grid = make_grid(xh, nrow=4)
            save_image(grid, f"./contents/ct_{name}_sample_5step_{epoch}.png")

            # Sample 2 Steps
            xh = model.sample(
                torch.randn_like(x).to(device=device) * 80.0,
                list(reversed([2.0, 80.0])),
            )
            xh = (xh * 0.5 + 0.5).clamp(0, 1)
            grid = make_grid(xh, nrow=4)
            save_image(grid, f"./contents/ct_{name}_sample_2step_{epoch}.png")

            # save model
            torch.save(model.state_dict(), f"./ct_{name}.pth")


if __name__ == "__main__":
    # train()
    train(dataloader=cifar_mnist_dl(), n_channels=3, name="cifar_mnist")
