import time
import math
import torch
import torch.nn.functional as F
from learning_fun import linear_rampup, cosine_rampdown


def train_sup(label_loader, model, criterions, optimizer, epoch, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses_class = AverageMeter()
    losses_total = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    criterion, _, _, criterion_l1 = criterions

    end = time.time()

    label_iter = iter(label_loader)
    for i in range(len(label_iter)):
        # lr = adjust_learning_rate(optimizer, epoch, i, len(label_iter))

        input, target, _ = next(label_iter)
        # measure data loading time
        data_time.update(time.time() - end)
        sl = input.shape
        batch_size = sl[0]

        # patch for torch.__version__ > 0.4.1 
        # target = target.cuda(async=True)
        target = target.cuda()

        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        # compute output
        output = model(input_var)

        loss_ce = criterion(output, target_var) / float(batch_size)

        reg_l1 = cal_reg_l1(model, criterion_l1)

        loss = loss_ce + args.weight_l1 * reg_l1

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses_class.update(loss_ce.item(), input.size(0))
        losses_total.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(label_iter), batch_time=batch_time, data_time=data_time,
                loss=losses_class, top1=top1, top5=top5))

    return {
        "top1": top1.avg,
        "class_loss": losses_class.avg,
        "total_loss": losses_total.avg
    }


def train_pi(label_loader, unlabel_loader, model, criterions, optimizer, epoch, args, weight_pi=20.0):
    batch_time = AverageMeter()
    data_time = AverageMeter()

    losses_class = AverageMeter()
    losses_pi = AverageMeter()
    losses_total = AverageMeter()

    top1 = AverageMeter()
    top5 = AverageMeter()
    weights_cl = AverageMeter()

    # switch to train mode
    model.train()

    criterion, criterion_mse, _, criterion_l1 = criterions

    end = time.time()

    label_iter = iter(label_loader)
    unlabel_iter = iter(unlabel_loader)
    len_iter = len(unlabel_iter)
    for i in range(len_iter):
        # set weights for the consistency loss
        # lr = adjust_learning_rate(optimizer, epoch, i, len(len_iter))
        weight_cl = cal_consistency_weight(epoch * len_iter + i, end_ep=(args.epochs // 2) * len_iter, end_w=1.0)

        try:
            input, target, input1 = next(label_iter)
        except StopIteration:
            label_iter = iter(label_loader)
            input, target, input1 = next(label_iter)
        input_ul, _, input1_ul = next(unlabel_iter)
        sl = input.shape
        su = input_ul.shape
        batch_size = sl[0] + su[0]
        # measure data loading time
        data_time.update(time.time() - end)

        # patch for torch.__version__ > 0.4.1
        # target = target.cuda(async=True)
        target = target.cuda()

        input_var = torch.autograd.Variable(input)
        input1_var = torch.autograd.Variable(input1)
        input_ul_var = torch.autograd.Variable(input_ul)
        input1_ul_var = torch.autograd.Variable(input1_ul)
        input_concat_var = torch.cat([input_var, input_ul_var])
        input1_concat_var = torch.cat([input1_var, input1_ul_var])

        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_concat_var)
        with torch.no_grad():
            output1 = model(input1_concat_var)

        output_label = output[:sl[0]]
        # pred = F.softmax(output, 1) # consistency loss on logit is better
        # pred1 = F.softmax(output1, 1)
        loss_ce = criterion(output_label, target_var) / float(sl[0])
        loss_pi = criterion_mse(output, output1) / float(args.num_classes * batch_size)

        reg_l1 = cal_reg_l1(model, criterion_l1)

        loss = loss_ce + args.weight_l1 * reg_l1 + weight_cl * weight_pi * loss_pi

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output_label.data, target, topk=(1, 5))
        losses_class.update(loss_ce.item(), input.size(0))
        losses_pi.update(loss_pi.item(), input.size(0))
        losses_total.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))
        weights_cl.update(weight_cl, input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'LossPi {loss_pi.val:.4f} ({loss_pi.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len_iter, batch_time=batch_time,
                data_time=data_time, loss=losses_class, loss_pi=losses_pi,
                top1=top1, top5=top5))

    return {
        "top1": top1.avg,
        "top5": top5.avg,
        "class_loss": losses_class.avg,
        "pi_loss": losses_pi.avg,
        "total_loss": losses_total.avg,
        "weight_cl": weights_cl.avg
    }


def train_mt(label_loader, unlabel_loader, teacher_model, student_model, criterions, optimizer, epoch, args,
             ema_const=0.95, weight_mt=8.0):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses_class = AverageMeter()
    losses_cl = AverageMeter()
    losses_total = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    top1_t = AverageMeter()
    top5_t = AverageMeter()
    weights_cl = AverageMeter()

    # switch to train mode
    teacher_model.train()
    student_model.train()

    criterion, criterion_mse, _, criterion_l1 = criterions

    end = time.time()

    label_iter = iter(label_loader)
    unlabel_iter = iter(unlabel_loader)
    len_iter = len(unlabel_iter)
    for i in range(len_iter):
        # set weights for the consistency loss
        global_step = epoch * len_iter + i
        weight_cl = cal_consistency_weight(global_step, end_ep=(args.epochs // 2) * len_iter, end_w=1.0)
        # lr = adjust_learning_rate(optimizer, epoch, i, len(len_iter))

        try:
            input, target, input1 = next(label_iter)
        except StopIteration:
            label_iter = iter(label_loader)
            input, target, input1 = next(label_iter)
        input_ul, _, input1_ul = next(unlabel_iter)
        sl = input.shape
        su = input_ul.shape
        batch_size = sl[0] + su[0]
        # measure data loading time
        data_time.update(time.time() - end)
        # target = target.cuda(async=True)
        target = target.cuda()
        input_var = torch.autograd.Variable(input)
        input1_var = torch.autograd.Variable(input1)
        input_ul_var = torch.autograd.Variable(input_ul)
        input1_ul_var = torch.autograd.Variable(input1_ul)
        input_concat_var = torch.cat([input_var, input_ul_var])
        input1_concat_var = torch.cat([input1_var, input1_ul_var])

        target_var = torch.autograd.Variable(target)

        # compute output
        output = student_model(input_concat_var)
        with torch.no_grad():
            output1 = teacher_model(input1_concat_var)

        output_label = output[:sl[0]]
        output1_label = output1[:sl[0]]

        loss_ce = criterion(output_label, target_var) / float(sl[0])  # 分类损失

        # loss_ema = criterion(output1_label, target_var) /float(sl[0])
        loss_cl = criterion_mse(output, output1) / float(args.num_classes * batch_size)  # 一致性损失

        reg_l1 = cal_reg_l1(student_model, criterion_l1)

        loss = loss_ce + args.weight_l1 * reg_l1 + weight_cl * weight_mt * loss_cl

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output_label.data, target, topk=(1, 5))
        prec1_t, prec5_t = accuracy(output1_label.data, target, topk=(1, 5))
        losses_class.update(loss_ce.item(), input.size(0))
        losses_cl.update(loss_cl.item(), input.size(0))
        losses_total.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))
        top1_t.update(prec1_t.item(), input.size(0))
        top5_t.update(prec5_t.item(), input.size(0))
        weights_cl.update(weight_cl, input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        update_ema_variables(teacher_model, student_model, ema_const, global_step)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss_cla {loss.val:.4f} ({loss.avg:.4f})\t'
                  'LossCL {loss_cl.val:.4f} ({loss_cl.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'PrecT@1 {top1_t.val:.3f} ({top1_t.avg:.3f})\t'
                  'PrecT@5 {top5_t.val:.3f} ({top5_t.avg:.3f})'.format(
                epoch, i, len_iter, loss=losses_class, loss_cl=losses_cl,
                top1=top1, top1_t=top1_t, top5_t=top5_t))

    return {
        "top1_1": top1.avg,
        "top1_t": top1_t.avg,
        "class_loss": losses_class.avg,
        "cl_loss": losses_cl.avg,
        "total_loss": losses_total.avg,
        "weight_cl": weights_cl.avg
    }


def train_mtp(
        label_loader, unlabel_loader, teacher_model, student1_model, student2_model, criterions,
        student1_optimizer, student2_optimizer, epoch, args, c_flag=False, ema_const=0.95, weight_mt=8.0):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses_class_1 = AverageMeter()
    losses_cl_1 = AverageMeter()
    losses_total_1 = AverageMeter()
    losses_class_2 = AverageMeter()
    losses_cl_2 = AverageMeter()
    losses_total_2 = AverageMeter()
    losses_class_ema = AverageMeter()
    top1_1 = AverageMeter()
    top5_1 = AverageMeter()
    top1_2 = AverageMeter()
    top5_2 = AverageMeter()
    top1_t = AverageMeter()
    top5_t = AverageMeter()
    weights_cl = AverageMeter()

    # switch to train mode
    teacher_model.train()
    student1_model.train()
    student2_model.train()

    criterion, criterion_mse, _, criterion_l1 = criterions

    end = time.time()

    label_iter = iter(label_loader)
    unlabel_iter = iter(unlabel_loader)
    len_iter = len(unlabel_iter)
    for i in range(len_iter):
        # set weights for the consistency loss
        global_step = epoch * len_iter + i
        weight_cl = cal_consistency_weight(global_step, end_ep=(args.epochs // 2) * len_iter, end_w=1.0)

        try:
            input1, target, input_ema = next(label_iter)
            input2 = input1.clone()
        except StopIteration:
            label_iter = iter(label_loader)
            input1, target, input_ema = next(label_iter)

        input1_ul, _, input_ema_ul = next(unlabel_iter)
        input2_ul = input1_ul.clone()

        sl = input1.shape
        su = input1_ul.shape
        batch_size = sl[0] + su[0]
        # measure data loading time
        data_time.update(time.time() - end)
        # target = target.cuda(async=True)
        target = target.cuda()

        input1_var = torch.autograd.Variable(input1)
        input2_var = torch.autograd.Variable(input2)
        input_ema_var = torch.autograd.Variable(input_ema)
        input1_ul_var = torch.autograd.Variable(input1_ul)
        input2_ul_var = torch.autograd.Variable(input2_ul)
        input_ema_ul_var = torch.autograd.Variable(input_ema_ul)
        input1_concat_var = torch.cat([input1_var, input1_ul_var])
        input2_concat_var = torch.cat([input2_var, input2_ul_var])
        input_ema_concat_var = torch.cat([input_ema_var, input_ema_ul_var])

        target_var = torch.autograd.Variable(target)

        # compute output
        output1 = student1_model(input1_concat_var)
        if c_flag:
            output2_1, output2_2 = student2_model(input2_concat_var)
        else:
            output2_1 = student2_model(input2_concat_var)

        with torch.no_grad():
            output_ema = teacher_model(input_ema_concat_var)

        output1_label = output1[:sl[0]]
        output2_label = output2_1[:sl[0]]
        output_ema_label = output_ema[:sl[0]]
        # pred = F.softmax(output, 1)
        # pred1 = F.softmax(output1, 1)
        # 分类损失
        loss_ce1 = criterion(output1_label, target_var) / float(sl[0])
        loss_ce2 = criterion(output2_label, target_var) / float(sl[0])
        loss_ema = criterion(output_ema_label, target_var) / float(sl[0])

        # 参数计算
        total_loss = loss_ce1 + loss_ce2
        wk1 = (total_loss - loss_ce1) / total_loss
        wk2 = (total_loss - loss_ce2) / total_loss

        # 一致性损失
        loss_cl1 = criterion_mse(output1, output_ema) / float(args.num_classes * batch_size)
        if c_flag:
            loss_cl2 = criterion_mse(output2_1, output2_2) / float(args.num_classes * batch_size)

        else:
            loss_cl2 = criterion_mse(output2_1, output_ema) / float(args.num_classes * batch_size)

        # L1正则化
        reg_l1 = cal_reg_l1(student1_model, criterion_l1)
        reg_l2 = cal_reg_l1(student2_model, criterion_l1)

        loss1 = loss_ce1 + args.weight_l1 * reg_l1 + weight_cl * weight_mt * loss_cl1
        loss2 = loss_ce2 + args.weight_l1 * reg_l2 + weight_cl * weight_mt * loss_cl2

        # measure accuracy and record loss
        prec1_1, prec5_1 = accuracy(output1_label.data, target, topk=(1, 5))
        prec1_2, prec5_2 = accuracy(output2_label.data, target, topk=(1, 5))
        prec1_t, prec5_t = accuracy(output_ema_label.data, target, topk=(1, 5))
        losses_class_ema.update(loss_ema.item(), input1.size(0))

        losses_class_1.update(loss_ce1.item(), input1.size(0))
        losses_cl_1.update(loss_cl1.item(), input1.size(0))
        losses_total_1.update(loss1.item(), input1.size(0))

        losses_class_2.update(loss_ce2.item(), input2.size(0))
        losses_cl_2.update(loss_cl2.item(), input2.size(0))
        losses_total_2.update(loss2.item(), input1.size(0))

        top1_1.update(prec1_1.item(), input1.size(0))
        top5_1.update(prec5_1.item(), input1.size(0))
        top1_2.update(prec1_2.item(), input2.size(0))
        top5_2.update(prec5_2.item(), input2.size(0))
        top1_t.update(prec1_t.item(), input_ema.size(0))
        top5_t.update(prec5_t.item(), input_ema.size(0))
        weights_cl.update(weight_cl, input_ema.size(0))

        # compute gradient and do SGD step
        student1_optimizer.zero_grad()
        loss1.backward()
        student1_optimizer.step()

        student2_optimizer.zero_grad()
        loss2.backward()
        student2_optimizer.step()

        update_mtp_variables(teacher_model, student1_model, student2_model, ema_const, global_step, wk1, wk2)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss_ema {loss_ema.val:.4f} ({loss_ema.avg:.4f})\t'
                  'Loss_1 {loss_1.val:.4f} ({loss_1.avg:.4f})\t'
                  'Loss_2 {loss_2.val:.4f} ({loss_2.avg:.4f})\t'
                  'LossCL_1 {loss_cl_1.val:.4f} ({loss_cl_1.avg:.4f})\t'
                  'LossCL_2 {loss_cl_2.val:.4f} ({loss_cl_2.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'PrecT@1 {top1_t.val:.3f} ({top1_t.avg:.3f})\t'
                  'PrecT@5 {top5_t.val:.3f} ({top5_t.avg:.3f})'.format(
                epoch, i, len_iter, loss_ema=losses_class_ema, loss_1=losses_class_1, loss_2=losses_class_1,
                loss_cl_1=losses_cl_1, loss_cl_2=losses_cl_2, top1=top1_1, top1_t=top1_t, top5_t=top5_t))
    return {
        'top1_1': top1_1.avg,
        'top1_2': top1_2.avg,
        'top1_t': top1_t.avg,
        '1_class_loss': losses_class_1.avg,
        '2_class_loss': losses_class_2.avg,
        'ema_class_loss': losses_class_ema.avg,
        '1_cl_loss': losses_cl_1.avg,
        '2_cl_loss': losses_cl_2.avg,
        '1_total_loss': losses_total_1.avg,
        '2_total_loss': losses_total_2.avg,
        'weight_cl': weights_cl.avg
    }


def validate(val_loader, model, criterions, args, mode='valid'):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    criterion, criterion_mse, _, _ = criterions

    end = time.time()
    with torch.no_grad():
        for i, (input, target, _) in enumerate(val_loader):
            sl = input.shape
            batch_size = sl[0]
            # target = target.cuda(async=True)
            target = target.cuda()
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)

            # compute output
            output = model(input_var)
            softmax = torch.nn.LogSoftmax(dim=1)(output)
            loss = criterion(output, target_var) / float(batch_size)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                if mode == 'test':
                    print('Test: [{0}/{1}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                          'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                        i, len(val_loader), batch_time=batch_time, loss=losses,
                        top1=top1, top5=top5))
                else:
                    print('Valid: [{0}/{1}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                          'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                        i, len(val_loader), batch_time=batch_time, loss=losses,
                        top1=top1, top5=top5))

    print(' ****** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.3f} '
          .format(top1=top1, top5=top5, loss=losses))

    return top1.avg, losses.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def cal_consistency_weight(epoch, init_ep=0, end_ep=150, init_w=0.0, end_w=20.0):
    """Sets the weights for the consistency loss"""
    if epoch > end_ep:
        weight_cl = end_w
    elif epoch < init_ep:
        weight_cl = init_w
    else:
        T = float(epoch - init_ep) / float(end_ep - init_ep)
        # weight_mse = T * (end_w - init_w) + init_w #linear
        weight_cl = (math.exp(-5.0 * (1.0 - T) * (1.0 - T))) * (end_w - init_w) + init_w  # exp
    # print('Consistency weight: %f'%weight_cl)
    # print('weight_cl=', weight_cl)
    return weight_cl


def cal_reg_l1(model, criterion_l1):
    reg_loss = 0
    np = 0
    for param in model.parameters():
        reg_loss += criterion_l1(param, torch.zeros_like(param))
        np += param.nelement()
    reg_loss = reg_loss / np
    return reg_loss


def update_ema_variables(teacher_model, student_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1.0 - 1.0 / float(global_step + 1), alpha)
    for param_t, param in zip(teacher_model.parameters(), student_model.parameters()):
        param_t.data.mul_(alpha).add_(1 - alpha, param)


def update_mtp_variables(teacher_model, student1_model, student2_model, alpha, global_step, wk1, wk2):
    # Use the true average until the exponential average is more correct
    alpha = min(1.0 - 1.0 / float(global_step + 1), alpha)
    for param_t, param1, param2 in zip(teacher_model.parameters(), student1_model.parameters(),
                                       student2_model.parameters()):
        tem_param1 = param1.data.mul(wk1)
        tem_param2 = param2.data.mul(wk2)
        tem_param_t = tem_param1 + tem_param2

        param_t.data.mul_(alpha).add_(1 - alpha, tem_param_t)
