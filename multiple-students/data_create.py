# Copyright (c) 2018, Curious AI Ltd. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

import torchvision.transforms as transforms
import torch.utils.data as data

import loader_cifar as cifar
import loader_cifar_zca as cifar_zca
import loader_svhn as svhn

"""
数据制作
"""

def cifar10(label_num, boundary, batch_size, num_workers):
    """
    根据标记率制作数据
    :param label_rate: 数据标记率
    :param boundary: 切割边界
    :param batch_size:
    :return:
    """
    dataloader = cifar.CIFAR10
    num_classes = 10
    data_dir = './dataset'

    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                     std=[0.2023, 0.1994, 0.2010])
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=2),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    # 固定均分batch_size
    batch_size_label = batch_size // 2
    batch_size_unlabel = batch_size // 2

    labelset = dataloader(root=data_dir, split='label', download=False,
                          transform=transform_train, label_num=label_num, boundary=boundary)  # default：download=True
    unlabelset = dataloader(root=data_dir, split='unlabel', download=False,
                            transform=transform_train, label_num=label_num, boundary=boundary)
    label_loader = data.DataLoader(labelset,
                                   batch_size=batch_size_label,
                                   shuffle=True,
                                   num_workers=num_workers,
                                   pin_memory=True)
    unlabel_loader = data.DataLoader(unlabelset,
                                     batch_size=batch_size_unlabel,
                                     shuffle=True,
                                     num_workers=num_workers,
                                     pin_memory=True)

    print("Batch size (label): ", batch_size_label)
    print("Batch size (unlabel): ", batch_size_unlabel)

    validset = dataloader(root=data_dir, split='valid', download=False,
                          transform=transform_test, boundary=boundary)
    val_loader = data.DataLoader(validset,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=num_workers,
                                 pin_memory=True)

    testset = dataloader(root=data_dir, split='test', download=False, transform=transform_test)
    test_loader = data.DataLoader(testset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=num_workers,
                                  pin_memory=True)

    return label_loader, unlabel_loader, val_loader, test_loader


def cifar10_zca(label_num, boundary, batch_size, num_workers):
    """
    根据标记率制作数据
    :param label_rate: 数据标记率
    :param boundary: 切割边界
    :param batch_size:
    :return:
    """
    dataloader = cifar_zca.CIFAR10
    num_classes = 10
    data_dir = './dataset'

    transform_train = transforms.Compose([
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    # 固定均分batch_size
    batch_size_label = batch_size // 2
    batch_size_unlabel = batch_size // 2

    labelset = dataloader(root=data_dir, split='label', download=False,
                          transform=transform_train, label_num=label_num, boundary=boundary)  # default：download=True
    unlabelset = dataloader(root=data_dir, split='unlabel', download=False,
                            transform=transform_train, label_num=label_num, boundary=boundary)
    label_loader = data.DataLoader(labelset,
                                   batch_size=batch_size_label,
                                   shuffle=True,
                                   num_workers=num_workers,
                                   pin_memory=True)
    unlabel_loader = data.DataLoader(unlabelset,
                                     batch_size=batch_size_unlabel,
                                     shuffle=True,
                                     num_workers=num_workers,
                                     pin_memory=True)

    print("Batch size (label): ", batch_size_label)
    print("Batch size (unlabel): ", batch_size_unlabel)

    validset = dataloader(root=data_dir, split='valid', download=False,
                          transform=transform_test, boundary=boundary)
    val_loader = data.DataLoader(validset,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=num_workers,
                                 pin_memory=True)

    testset = dataloader(root=data_dir, split='test', download=False, transform=transform_test)
    test_loader = data.DataLoader(testset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=num_workers,
                                  pin_memory=True)

    return label_loader, unlabel_loader, val_loader, test_loader


def svhn(label_num, boundary, batch_size, num_workers):
    """
    根据标记率制作数据
    :param label_rate: 数据标记率
    :param boundary: 切割边界
    :param batch_size:
    :return:
    """
    dataloader = svhn.SVHN
    num_classes = 10
    data_dir = './dataset'

    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=2),
        transforms.ToTensor(),
        normalize,
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    # 固定均分batch_size
    batch_size_label = batch_size // 2
    batch_size_unlabel = batch_size // 2

    labelset = dataloader(root=data_dir, split='label', download=False,
                          transform=transform_train, label_num=label_num, boundary=boundary)  # default：download=True
    unlabelset = dataloader(root=data_dir, split='unlabel', download=False,
                            transform=transform_train, label_num=label_num, boundary=boundary)
    label_loader = data.DataLoader(labelset,
                                   batch_size=batch_size_label,
                                   shuffle=True,
                                   num_workers=num_workers,
                                   pin_memory=True)
    unlabel_loader = data.DataLoader(unlabelset,
                                     batch_size=batch_size_unlabel,
                                     shuffle=True,
                                     num_workers=num_workers,
                                     pin_memory=True)

    print("Batch size (label): ", batch_size_label)
    print("Batch size (unlabel): ", batch_size_unlabel)

    validset = dataloader(root=data_dir, split='valid', download=False,
                          transform=transform_test, boundary=boundary)
    val_loader = data.DataLoader(validset,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=num_workers,
                                 pin_memory=True)

    testset = dataloader(root=data_dir, split='test', download=False, transform=transform_test)
    test_loader = data.DataLoader(testset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=num_workers,
                                  pin_memory=True)

    return label_loader, unlabel_loader, val_loader, test_loader

def cifar100(label_num, boundary, batch_size, num_workers):
    """
    根据标记率制作数据
    :param label_rate: 数据标记率
    :param boundary: 切割边界
    :param batch_size:
    :return:
    """
    dataloader = cifar.CIFAR100
    num_classes = 100
    data_dir = './dataset'

    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                     std=[0.2023, 0.1994, 0.2010])
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=2),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    # 固定均分batch_size
    batch_size_label = batch_size // 2
    batch_size_unlabel = batch_size // 2

    labelset = dataloader(root=data_dir, split='label', download=False,
                          transform=transform_train, label_num=label_num, boundary=boundary)  # default：download=True
    unlabelset = dataloader(root=data_dir, split='unlabel', download=False,
                            transform=transform_train, label_num=label_num, boundary=boundary)
    label_loader = data.DataLoader(labelset,
                                   batch_size=batch_size_label,
                                   shuffle=True,
                                   num_workers=num_workers,
                                   pin_memory=True)
    unlabel_loader = data.DataLoader(unlabelset,
                                     batch_size=batch_size_unlabel,
                                     shuffle=True,
                                     num_workers=num_workers,
                                     pin_memory=True)

    print("Batch size (label): ", batch_size_label)
    print("Batch size (unlabel): ", batch_size_unlabel)

    validset = dataloader(root=data_dir, split='valid', download=False,
                          transform=transform_test, boundary=boundary)
    val_loader = data.DataLoader(validset,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=num_workers,
                                 pin_memory=True)

    testset = dataloader(root=data_dir, split='test', download=False, transform=transform_test)
    test_loader = data.DataLoader(testset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=num_workers,
                                  pin_memory=True)

    return label_loader, unlabel_loader, val_loader, test_loader




def imagenet():
    channel_stats = dict(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
    train_transformation = data.TransformTwice(transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ]))
    eval_transformation = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ])

    return {
        'train_transformation': train_transformation,
        'eval_transformation': eval_transformation,
        'datadir': 'data-local/images/ilsvrc2012/',
        'num_classes': 1000
    }





