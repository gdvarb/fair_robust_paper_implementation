import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def pgd_attack(model,
                  X,
                  y,
                  epsilon,
                  clip_max,
                  clip_min,
                  num_steps,
                  step_size, weight=None):

    device = X.device

    if weight is not None:
        # new_eps will be a tensor with shape [batch_size, 1, 1, 1]
        new_eps = (epsilon * weight).view(weight.shape[0], 1, 1, 1)
    else:
        new_eps = epsilon

    imageArray = X.detach().cpu().numpy()
    X_random = np.random.uniform(-epsilon, epsilon, X.shape)
    imageArray = np.clip(imageArray + X_random, 0, 1.0)

    X_pgd = torch.tensor(imageArray).to(device).float()
    X_pgd.requires_grad = True

    for i in range(num_steps):
        pred = model(X_pgd)
        loss = nn.CrossEntropyLoss()(pred, y)
        loss.backward()

        eta = step_size * X_pgd.grad.data.sign()
        X_pgd = X_pgd + eta

        eta = torch.clamp(X_pgd.data - X.data, -new_eps, new_eps)
        X_pgd = X.data + eta
        X_pgd = torch.clamp(X_pgd, clip_min, clip_max)

        X_pgd = X_pgd.detach()
        X_pgd.requires_grad_()
        X_pgd.retain_grad()

    return X_pgd



def in_class(predict, label):

    probs = torch.zeros(10)
    for i in range(10):
        in_class_id = torch.tensor(label == i, dtype= torch.float)
        correct_predict = torch.tensor(predict == label, dtype= torch.float)
        in_class_correct_predict = (correct_predict) * (in_class_id)
        acc = torch.sum(in_class_correct_predict).item() / torch.sum(in_class_id).item()
        probs[i] = acc

    return probs

def evaluate(model, test_loader, configs, device, mode = 'Test'):

    print('Doing evaluation mode ' + mode)
    model.eval()

    correct = 0
    correct_adv = 0

    all_label = []
    all_pred = []
    all_pred_adv = []

    for batch_idx, (data, target) in enumerate(test_loader):

        data, target = torch.tensor(data).to(device), torch.tensor(target).to(device)
        all_label.append(target)

        ## clean test
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        add = pred.eq(target.view_as(pred)).sum().item()
        correct += add
        model.zero_grad()
        all_pred.append(pred)

        ## adv test
        x_adv = pgd_attack(model, X = data, y = target, **configs)
        output1 = model(x_adv)
        pred1 = output1.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        add1 = pred1.eq(target.view_as(pred1)).sum().item()
        correct_adv += add1
        all_pred_adv.append(pred1)

    all_label = torch.cat(all_label).flatten()
    all_pred = torch.cat(all_pred).flatten()
    all_pred_adv = torch.cat(all_pred_adv).flatten()

    acc = in_class(all_pred, all_label)
    acc_adv = in_class(all_pred_adv, all_label)

    total_clean_error = 1- correct / len(test_loader.dataset)
    total_bndy_error = correct / len(test_loader.dataset) - correct_adv / len(test_loader.dataset)

    class_clean_error = 1 - acc
    class_bndy_error = acc - acc_adv

    return class_clean_error, class_bndy_error, total_clean_error, total_bndy_error


def frl_train_remargin(h_net, ds_train, ds_valid, optimizer, now_epoch, configs, configs1, device, delta1, rate2, beta):
    print('train epoch ' + str(now_epoch), flush=True)

    ## given model, get the validation performance and gamma
    class_clean_error, class_bndy_error, total_clean_error, total_bndy_error = \
        evaluate(h_net, ds_valid, configs1, device, mode='Validation')

    # CIFAR 10
    num_classes = 10  
    if not hasattr(frl_train_remargin, "epsilons"):
        frl_train_remargin.epsilons = torch.ones(num_classes, device=device) * configs['epsilon']
    epsilons = frl_train_remargin.epsilons
    
    #updates the per-class adversarial margin epsilon
    for i in range(num_classes):
        # increasing the margin for classes with higher boundary error
        epsilons[i] = epsilons[i] * torch.exp(rate2 * (class_bndy_error[i] - delta1[i]))
        epsilons[i] = torch.clamp(epsilons[i], max=16/255)
    frl_train_remargin.epsilons = epsilons

    h_net.train()
    for batch_idx, (data, target) in enumerate(ds_train):
        data, target = torch.tensor(data).to(device), torch.tensor(target).to(device)
        sample_eps = epsilons[target]
        
        #generate adversarial examples based on sample_eps
        x_adv = pgd_attack(h_net, data, target,**configs,weight=sample_eps)
        
        optimizer.zero_grad()
        
        # Loss Computation
        loss_natural = nn.CrossEntropyLoss()(h_net(data), target)
        loss_bndy_vec = nn.KLDivLoss(reduction='none')(F.log_softmax(h_net(x_adv), dim=1), F.softmax(h_net(data), dim=1))
        loss_bndy = torch.sum(loss_bndy_vec, 1)
        loss = torch.mean(loss_natural) + beta * torch.mean(loss_bndy)
        loss.backward()
        optimizer.step()


