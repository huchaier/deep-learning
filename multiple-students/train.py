# this code is modified from the pytorch code: https://github.com/CSAILVision/places365
# JH Kim
#  

import argparse
import os
import shutil

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import numpy as np
import matplotlib.pyplot as plt

import bisect

import data_create
from methods import validate, train_sup, train_pi, train_mt, train_mtp
from network.architectures import CNN as net13
from network.architectures import resnet18

parser = argparse.ArgumentParser(description='PyTorch Semi-supervised learning Training')
parser.add_argument('--arch', '-a', metavar='ARCH', default='net13',
                    help='model architecture: ' + ' (default: wideresnet)', choices=['net13', 'resnet18'])  # 网络结构选择
parser.add_argument('--model', '-m', metavar='MODEL', default='mt',
                    help='model: ' + ' (default: pso_mt)',
                    choices=['baseline', 'pi', 'mt', 'mt+', 'ds_mt', 'ms', 'd-ds_mt'])  # 算法选择
parser.add_argument('--optim', '-o', metavar='OPTIM', default='sgd',
                    help='optimizer: ' + ' (default: adam)', choices=['adam', 'sgd'])  # 反传算法
parser.add_argument('--dataset', '-d', metavar='DATASET', default='cifar10',
                    help='dataset: ' + ' (default: cifar10)', choices=['cifar10', 'cifar10_zca', 'svhn'])  # 数据集
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 0)')  # 工作者
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')  # 总训练批次
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=60, type=int,
                    metavar='N', help='mini-batch size (default: 256)')  # batch-size
parser.add_argument('--drop_rate', default=0.5, type=float, help='dropRatio')  # drop概率
parser.add_argument('--lr', '--learning-rate', default=0.003, type=float,
                    metavar='LR', help='initial learning rate')  # 学习率
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')  # 动量
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')  # 权重衰减，L2正则化系数
parser.add_argument('--weight_l1', '--l1', default=1e-3, type=float,
                    metavar='W1', help='l1 regularization (default: 1e-3)')  # L1正则化
parser.add_argument('--print-freq', '-p', default=100, type=int,
                    metavar='N', help='print frequency (default: 10)')  # 打印进程
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')  # 最近的ckpt保存路径
parser.add_argument('--num_classes', default=10, type=int, help='number of classes in the model')  # 类别数
parser.add_argument('--ckpt', default='./checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: ckpt)')  # ckp保存路径
parser.add_argument('--boundary', default=0, type=int, help='different label/unlabel division [0,9]')  # 标签划分
parser.add_argument('--gpu', default=0, type=str, help='cuda_visible_devices')  # cuda
parser.add_argument('--label_num', default=4000, type=int, help='rate of label data')  # 标记样本数
args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

best_prec1 = 0  # 最佳验证准确率
best_test_prec1 = 0  # 测试准确率
# is_best = False

acc1_stu1_tr, losses_stu1_tr, losses_stu1_cl_tr = [], [], []  # 学生1训练top1，总误差， 一致性误差
acc1_stu2_tr, losses_stu2_tr, losses_stu2_cl_tr = [], [], []  # 学生2训练top1，总误差， 一致性误差
acc1_t_tr, losses_t_tr = [], []  # 教师（模型）准确率、损失
acc1_t_val, losses_t_val = [], []  # 验证准确率、损失
acc1_t_test, losses_t_test = [], []  # 测试准确率、损失

learning_rate, weights_cl = [], []  # 学习率、一致性权重


def main():
    global args, best_prec1, best_test_prec1
    global acc1_stu1_tr, losses_stu1_tr, losses_stu1_cl_tr
    global acc1_stu2_tr, losses_stu2_tr, losses_stu2_cl_tr
    global acc1_t_tr, losses_t_tr
    global acc1_t_val, losses_t_val
    global acc1_t_test, losses_t_test
    global learning_rate, weights_cl
    args = parser.parse_args()

    # 保存训练loss
    loss_dict = {
        # 教师（模型）分类、一致性、总损失
        "loss_class": [],
        "loss_cl": [],
        "loss_total": [],
        # 学生1分类、一致、总损失
        "loss_class_1": [],
        "loss_cl_1": [],
        "loss_total_1": [],
        # 学生2分类、一致、总损失
        "loss_class_2": [],
        "loss_cl_2": [],
        "loss_total_2": [],
        # 验证、测试损失
        "val_loss": [],
        "test_loss": []
    }

    # 网络模型选择
    if args.arch == 'net13':
        print("Model: %s" % args.arch)
        student = net13(num_classes=args.num_classes, dropRatio=args.drop_rate)
        if args.model == 'ds_mt':
            student2 = net13(args.num_classes, dropRatio=args.drop_rate, isL2=True)
        if args.model == 'ms':
            student1 = net13(num_classes=args.num_classes, dropRatio=args.drop_rate)
            student2 = net13(num_classes=args.num_classes, dropRatio=args.drop_rate)
    elif args.arch == 'resnet18':
        print("Model: %s" % args.arch)
        student = resnet18(num_classes=args.num_classes)
        if args.model == 'ds_mt':
            student2 = resnet18(args.num_classes, isL2=True)
        if args.model == 'ms':
            student1 = resnet18(num_classes=args.num_classes)
            student2 = resnet18(num_classes=args.num_classes)
    else:
        assert (False)

    # 算法参数初始化
    if args.model == 'mt':
        import copy
        teacher = copy.deepcopy(student)
        teacher_model = torch.nn.DataParallel(teacher).cuda()  # 多GPU并行

    if args.model == 'mt+':
        import copy
        student1 = copy.deepcopy(student)
        student2 = copy.deepcopy(student)
        student1_model = torch.nn.DataParallel(student1).cuda()  # 多GPU并行
        student2_model = torch.nn.DataParallel(student2).cuda()  # 多GPU并行
        # student2_model = torch.nn.DataParallel(student2).cuda()  # 多GPU并行

    if args.model == 'ds_mt' or args.model == 'd-ds_mt':
        import copy
        student1 = copy.deepcopy(student)
        student1_model = torch.nn.DataParallel(student1).cuda()  # 多GPU并行
        student2_model = torch.nn.DataParallel(student2).cuda()  # 多GPU并行

    if args.model == 'ms':
        student1_model = torch.nn.DataParallel(student1).cuda()  # 多GPU并行
        student2_model = torch.nn.DataParallel(student2).cuda()  # 多GPU并行

    # if args.model == 'pso_mt+':
    #     student1_model = torch.nn.DataParallel(student1).cuda()  # 多GPU并行
    #     student2_model = torch.nn.DataParallel(student2).cuda()  # 多GPU并行
    #     # 保存最优参数
    #     best_his_param = {
    #         'local_loss': 999,
    #         'his_loss': 999,
    #     }

    student_model = torch.nn.DataParallel(student).cuda()

    # 检查点恢复
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            loss_dict = checkpoint['loss_dict']
            best_prec1 = checkpoint['best_prec1']
            student_model.load_state_dict(checkpoint['student_state_dict'])

            if args.model == 'mt':
                teacher_model.load_state_dict(checkpoint['teacher_state_dict'])
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(args.resume, checkpoint['epoch']))
            elif args.model != 'baseline' and args.model != 'pi':
                student1_model.load_state_dict(checkpoint['student1_state_dict'])
                student2_model.load_state_dict(checkpoint['student2_state_dict'])
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # 优化器
    if args.optim == 'sgd' or args.optim == 'adam':
        pass
    else:
        print('Not Implemented Optimizer')
        assert (False)

    # 反传算法优化器
    if args.optim == 'adam':
        print('Using Adam optimizer')
        optimizer = torch.optim.Adam(student_model.parameters(), args.lr,
                                     betas=(0.9, 0.999),
                                     weight_decay=args.weight_decay)
        if args.model != 'mt' and args.model != 'baseline' and args.model != 'pi':
            student1_optimizer = torch.optim.Adam(student1_model.parameters(), args.lr,
                                                  betas=(0.9, 0.999),
                                                  weight_decay=args.weight_decay)
            student2_optimizer = torch.optim.Adam(student2_model.parameters(), args.lr,
                                                  betas=(0.9, 0.999),
                                                  weight_decay=args.weight_decay)

    elif args.optim == 'sgd':
        print('Using SGD optimizer')
        optimizer = torch.optim.SGD(student_model.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
        if args.model != 'mt' and args.model != 'baseline' and args.model != 'pi':
            student1_optimizer = torch.optim.SGD(student1_model.parameters(), args.lr,
                                                 momentum=args.momentum,
                                                 weight_decay=args.weight_decay)
            student2_optimizer = torch.optim.SGD(student2_model.parameters(), args.lr,
                                                 momentum=args.momentum,
                                                 weight_decay=args.weight_decay)

    # 保存点路径设置
    ckpt_dir = args.ckpt + '/' + args.dataset + '_' + str(
        args.label_num) + '_' + args.arch + '_' + args.model + '_' + args.optim
    ckpt_dir = ckpt_dir + '_e%d' % (args.epochs)

    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    cudnn.benchmark = True

    # 数据导入
    label_loader, unlabel_loader, val_loader, test_loader = \
        data_create.__dict__[args.dataset](label_num=args.label_num, boundary=args.boundary,
                                           batch_size=args.batch_size, num_workers=args.workers)

    # 损失函数
    criterion = nn.CrossEntropyLoss(reduction='sum').cuda()
    criterion_mse = nn.MSELoss(reduction='sum').cuda()
    criterion_kl = nn.KLDivLoss(reduction='sum').cuda()
    criterion_l1 = nn.L1Loss(reduction='sum').cuda()

    criterions = (criterion, criterion_mse, criterion_kl, criterion_l1)

    # 训练
    for epoch in range(args.start_epoch, args.epochs):
        # 修改学习率
        if args.optim == 'adam':
            print('Learning rate schedule for Adam')
            lr = adjust_learning_rate_adam(optimizer, epoch)
            if args.model != 'mt' and args.model != 'baseline' and args.model != 'pi':
                _ = adjust_learning_rate_adam(student1_optimizer, epoch)
                _ = adjust_learning_rate_adam(student2_optimizer, epoch)

        elif args.optim == 'sgd':
            print('Learning rate schedule for SGD')
            lr = adjust_learning_rate(optimizer, epoch)
            if args.model != 'mt' and args.model != 'baseline' and args.model != 'pi':
                _ = adjust_learning_rate(student1_optimizer, epoch)
                _ = adjust_learning_rate(student2_optimizer, epoch)

        # train for one epoch
        if args.model == 'baseline':
            print('Supervised Training')
            for i in range(10):  # baseline repeat 10 times since small number of training set
                anser_dict = train_sup(label_loader, student_model, criterions, optimizer, epoch, args)
                weight_cl = 0.0
                anser_dict['weight_cl'] = weight_cl
            # 添加损失，用于绘图
            loss_dict['loss_total'].append(anser_dict['total_loss'])
        elif args.model == 'pi':
            print('Pi model')
            anser_dict = train_pi(label_loader, unlabel_loader, student_model, criterions, optimizer, epoch, args)
            loss_dict['loss_class'].append(anser_dict['class_loss'])
            loss_dict['loss_cl'].append(anser_dict['pi_loss'])
            loss_dict['loss_total'].append(anser_dict['total_loss'])

        elif args.model == 'mt':
            print('Mean Teacher model')
            anser_dict = train_mt(label_loader, unlabel_loader, teacher_model, student_model, criterions, optimizer,
                                  epoch, args)
            loss_dict['loss_class'].append(anser_dict['class_loss'])
            loss_dict['loss_cl'].append(anser_dict['cl_loss'])
            loss_dict['loss_total'].append(anser_dict['total_loss'])

        elif args.model == 'mt+':
            print('Mean Teacher Plus Student model')
            # (学生1top1, 学生1分类损失，学生1一致性损失，教师top1，一致性权重）
            anser_dict = train_mtp(label_loader, unlabel_loader, student_model, student1_model, student2_model,
                                   criterions, student1_optimizer, student2_optimizer, epoch, args)
            loss_dict['loss_class'].append(anser_dict['ema_class_loss'])
            loss_dict['loss_class_1'].append(anser_dict['1_class_loss'])
            loss_dict['loss_cl_1'].append(anser_dict['1_cl_loss'])
            loss_dict['loss_total_1'].append(anser_dict['1_total_loss'])
            loss_dict['loss_class_2'].append(anser_dict['2_class_loss'])
            loss_dict['loss_cl_2'].append(anser_dict['2_cl_loss'])
            loss_dict['loss_total_2'].append(anser_dict['2_total_loss'])

        elif args.model == 'ds_mt':
            print('Dual Student with Mean Teacher Model')
            # (学生1top1, 学生1分类损失，学生1一致性损失，教师top1，一致性权重）
            anser_dict = train_mtp(label_loader, unlabel_loader, student_model, student1_model, student2_model,
                                   criterions, student1_optimizer, student2_optimizer, epoch, args, c_flag=True)
            loss_dict['loss_class'].append(anser_dict['ema_class_loss'])
            loss_dict['loss_class_1'].append(anser_dict['1_class_loss'])
            loss_dict['loss_cl_1'].append(anser_dict['1_cl_loss'])
            loss_dict['loss_total_1'].append(anser_dict['1_total_loss'])
            loss_dict['loss_class_2'].append(anser_dict['2_class_loss'])
            loss_dict['loss_cl_2'].append(anser_dict['2_cl_loss'])
            loss_dict['loss_total_2'].append(anser_dict['2_total_loss'])

        elif args.model == 'd-ds_mt':
            print('Dual Student with Double Mean Teacher Model')
            # (学生1top1, 学生1分类损失，学生1一致性损失，教师top1，一致性权重）
            anser_dict = train_mtp(label_loader, unlabel_loader, student_model, student1_model, student2_model,
                                   criterions, student1_optimizer, student2_optimizer, epoch, args)
            loss_dict['loss_class'].append(anser_dict['ema_class_loss'])
            loss_dict['loss_class_1'].append(anser_dict['1_class_loss'])
            loss_dict['loss_cl_1'].append(anser_dict['1_cl_loss'])
            loss_dict['loss_total_1'].append(anser_dict['1_total_loss'])
            loss_dict['loss_class_2'].append(anser_dict['2_class_loss'])
            loss_dict['loss_cl_2'].append(anser_dict['2_cl_loss'])
            loss_dict['loss_total_2'].append(anser_dict['2_total_loss'])

        elif args.model == 'ms':
            print('Multiple Student Model')
            # (学生1top1, 学生1分类损失，学生1一致性损失，教师top1，一致性权重）
            anser_dict = train_mtp(label_loader, unlabel_loader, student_model, student1_model, student2_model,
                                   criterions, student1_optimizer, student2_optimizer, epoch, args)
            loss_dict['loss_class'].append(anser_dict['ema_class_loss'])
            loss_dict['loss_class_1'].append(anser_dict['1_class_loss'])
            loss_dict['loss_cl_1'].append(anser_dict['1_cl_loss'])
            loss_dict['loss_total_1'].append(anser_dict['1_total_loss'])
            loss_dict['loss_class_2'].append(anser_dict['2_class_loss'])
            loss_dict['loss_cl_2'].append(anser_dict['2_cl_loss'])
            loss_dict['loss_total_2'].append(anser_dict['2_total_loss'])
        # elif args.model == 'pso_mt+':
        #     print('pso_mt+ model')
        #     # (学生1top1, 学生1分类损失，学生1一致性损失，教师top1，一致性权重）
        #     prec1_1_tr, loss_1_tr, loss_1_cl_tr, prec1_t_tr, weight_cl  = train_psomt_pul(
        #         label_loader, unlabel_loader, student_model, student1_model, student2_model,
        #         criterions, student1_optimizer, student2_optimizer, epoch, args, best_his_param)
        else:
            print("Not Implemented ", args.model)
            assert (False)

        # 教师的验证和测试
        if args.model == 'mt':
            prec1_t_val, loss_t_val = validate(val_loader, teacher_model, criterions, args, 'valid')
            prec1_t_test, loss_t_test = validate(test_loader, teacher_model, criterions, args, 'test')
        else:
            prec1_t_val, loss_t_val = validate(val_loader, student_model, criterions, args, 'valid')
            prec1_t_test, loss_t_test = validate(test_loader, student_model, criterions, args, 'test')

        loss_dict['val_loss'].append(loss_t_val)
        loss_dict['test_loss'].append(loss_t_test)

        # 添加训练结果，保存到checkpoint中
        if args.model == 'baseline' or args.model == 'pi':
            acc1_stu1_tr.append(anser_dict['top1'])
            losses_stu1_tr.append(anser_dict['total_loss'])
            acc1_t_val.append(prec1_t_val)
            losses_t_val.append(loss_t_val)
            acc1_t_test.append(prec1_t_test)
            losses_t_test.append(loss_t_test)
        elif args.model == 'mt':
            # 学生训练
            acc1_stu1_tr.append(anser_dict['top1_1'])
            acc1_t_tr.append(anser_dict['top1_t'])
            losses_stu1_tr.append(anser_dict['top1_t'])
            losses_stu1_cl_tr.append(anser_dict['total_loss'])
            # 验证
            acc1_t_val.append(prec1_t_val)
            losses_t_val.append(loss_t_val)
            # 测试
            acc1_t_test.append(prec1_t_test)
            losses_t_test.append(loss_t_test)
        else:
            # 学生1训练
            acc1_stu1_tr.append(anser_dict['top1_1'])
            acc1_stu2_tr.append(anser_dict['top1_2'])
            acc1_t_tr.append(anser_dict['top1_t'])
            losses_stu1_tr.append(anser_dict['1_total_loss'])
            losses_stu2_tr.append(anser_dict['2_total_loss'])
            losses_t_tr.append(anser_dict['ema_class_loss'])
            losses_stu1_cl_tr.append(anser_dict['1_cl_loss'])
            acc1_t_val.append(prec1_t_val)
            losses_t_val.append(loss_t_val)
            acc1_t_test.append(prec1_t_test)
            losses_t_test.append(loss_t_test)
        weights_cl.append(anser_dict['weight_cl'])
        learning_rate.append(lr)

        # remember best prec@1 and save checkpoint
        if args.model == 'baseline' or args.model == 'pi':
            # 根据测试准确率保存
            is_best = prec1_t_test > best_prec1
            if is_best:
                best_prec1 = prec1_t_test
            print("Best test precision: %.3f" % best_prec1)
            dict_checkpoint = {
                'epoch': epoch + 1,
                'loss_dict': loss_dict,
                'student_state_dict': student_model.state_dict(),
                'best_prec1': best_prec1,
                'acc1_tr': acc1_stu1_tr,
                'losses_tr': losses_stu1_tr,
                'acc1_val': acc1_t_val,
                'losses_val': loss_t_val,
                'acc1_test': acc1_t_test,
                'losses_test': losses_t_test,
                'weights_cl': weights_cl,
                'learning_rate': learning_rate,
            }
        elif args.model == 'mt':
            is_best = prec1_t_test > best_prec1
            if is_best:
                best_prec1 = prec1_t_test
            print("Best test precision: %.3f" % best_prec1)
            dict_checkpoint = {
                'epoch': epoch + 1,
                'loss_dict': loss_dict,
                'student_state_dict': student_model.state_dict(),
                'teacher_state_dict': teacher_model.state_dict(),
                'best_prec1': best_prec1,
                'acc1_1_tr': acc1_stu1_tr,
                'losses_1_tr': losses_stu1_tr,
                'acc1_t_tr': acc1_t_tr,
                'losses_t_tr': losses_t_tr,
                'acc1_val': acc1_t_val,
                'losses_val': loss_t_val,
                'acc1_test': acc1_t_test,
                'losses_test': losses_t_test,
                'weights_cl': weights_cl,
                'learning_rate': learning_rate,
            }
        else:
            is_best = prec1_t_test > best_prec1
            if is_best:
                best_prec1 = prec1_t_test
            print("Best test precision: %.3f" % best_prec1)
            dict_checkpoint = {
                'epoch': epoch + 1,
                'loss_dict': loss_dict,
                'teacher_state_dict': student_model.state_dict(),  # 教师模型参数
                'student1_state_dict': student1.state_dict(),  # 学生1模型参数
                'student2_state_dict': student2.state_dict(),  # 学生2模型参数

                'best_prec1': best_prec1,
                'acc1_1_tr': acc1_stu1_tr,
                'losses_1_tr': losses_stu1_tr,
                'acc1_2_tr': acc1_stu2_tr,
                'losses_2_tr': losses_stu2_tr,
                'acc1_t_tr': acc1_t_tr,
                'acc1_t_val': acc1_t_val,
                'loss_t_val': loss_t_val,
                'acc1_t_test': acc1_t_test,
                'loss_t_test': losses_t_test,
                'weights_cl': weights_cl,
                'learning_rate': learning_rate,
            }

        save_checkpoint(dict_checkpoint, is_best, args.arch.lower() + str(args.boundary), dirname=ckpt_dir)

    # 回执模型损失图并保存
    save_loss(loss_dict['loss_class'], loss_dict['loss_cl'], loss_dict['loss_total'], name='model detail loss')
    save_loss(loss_dict['loss_class_1'], loss_dict['loss_cl_1'], loss_dict['loss_total_1'], name='Student1 loss')
    save_loss(loss_dict['loss_class_2'], loss_dict['loss_cl_2'], loss_dict['loss_total_2'], name='Student2 loss')
    save_loss(loss_dict['loss_total'], loss_dict['val_loss'], loss_dict['test_loss'], f_flag=True,
              name='train/val/test_loss')


def save_loss(data1, data2, data3, f_flag=False, name='loss'):
    x = range(0, len(data1))
    y1 = np.array(data1)
    y2 = np.array(data2)
    y3 = np.array(data3)
    # plt.subplot(2, 1, 1)  # 用于窗口切割，当前窗口为2行1列，图片位于第二位（中间）
    plt.scatter(x, y1, c='y', marker='^')
    plt.scatter(x, y2, c='g', marker='x')
    plt.scatter(x, y3, c='r', marker='.')
    if f_flag:
        plt.legend(["train", "val", "test"])
    else:
        plt.legend(["class", "cl", "total"])
    plt.title('{} vs. epoch'.format(name))
    plt.savefig("{}.jpg".format(name))
    # plt.show()


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', dirname='/checkpoint'):
    fpath = os.path.join(dirname, filename + '_latest.pth.tar')
    print('fpath= ', fpath)
    torch.save(state, fpath)
    if is_best:
        bpath = os.path.join(dirname, filename + '_best.pth.tar')
        # 文件移动
        shutil.copyfile(fpath, bpath)


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 at [150, 225, 300] epochs"""

    boundary = [args.epochs // 2, args.epochs // 4 * 3, args.epochs]
    lr = args.lr * 0.1 ** int(bisect.bisect_left(boundary, epoch))
    # print(epoch, lr, bisect.bisect_left(boundary, epoch))
    # lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


def adjust_learning_rate_adam(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 5 at [240] epochs"""

    boundary = [args.epochs // 5 * 4]
    lr = args.lr * 0.2 ** int(bisect.bisect_left(boundary, epoch))
    # print(epoch, lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


if __name__ == '__main__':
    main()
