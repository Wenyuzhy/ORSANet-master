import warnings

warnings.filterwarnings("ignore")
import torch.utils.data as data
from torchvision import transforms

import argparse
import torchvision.datasets as datasets
from sklearn.metrics import f1_score
from time import time
from utils import *
from models.sam import SAM
from models.net import ORSA
from torchsampler import ImbalancedDatasetSampler

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='rafdb', help='dataset [rafdb, affectnet7, affectnet8, occlu-FER]')
    parser.add_argument('--batch_size', type=int, default=20, help='Batch size.')
    parser.add_argument('--val_batch_size', type=int, default=50, help='Batch size for validation.')
    parser.add_argument('--modeltype', type=str, default='large', help='small or base or large')
    parser.add_argument('--optimizer', type=str, default="adam", help='Optimizer, adam or sgd.')
    parser.add_argument('--lr', type=float, default=0.00001, help='Initial learning rate for sgd.')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum for sgd')
    parser.add_argument('--workers', default=2, type=int, help='Number of data loading workers (default: 4)')
    parser.add_argument('--epochs', type=int, default=400, help='Total training epochs.')
    parser.add_argument('--gpu', type=str, default='1', help='assign multi-gpus by comma concat')
    parser.add_argument('--use_drae', type=int, default=5, help='use DRAEloss or not')
    parser.add_argument('--weight_drae', type=int, default=0.1, help='weight of DRAEloss')
    return parser.parse_args()


def run_training():
    args = parse_args()
    torch.manual_seed(123)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    print("Work on GPU: ", os.environ['CUDA_VISIBLE_DEVICES'])

    data_transforms_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(scale=(0.02, 0.1)),
    ])

    data_transforms_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    if args.dataset == "rafdb":
        datapath = './data/raf/'
        traindir = os.path.join(datapath, 'train')
        valdir = os.path.join(datapath, 'valid')
        num_classes = 7
        train_dataset = datasets.ImageFolder(traindir, transform=data_transforms_train)
        val_dataset = datasets.ImageFolder(valdir, transform=data_transforms_val)
        model = ORSA(img_size=224, num_classes=num_classes, type=args.modeltype)

    elif args.dataset == "affectnet7":
        datapath = './data/AffectNet_7/'
        num_classes = 7
        traindir = os.path.join(datapath, 'train')
        valdir = os.path.join(datapath, 'valid')
        train_dataset = datasets.ImageFolder(traindir, transform=data_transforms_train)
        val_dataset = datasets.ImageFolder(valdir, transform=data_transforms_val)
        model = ORSA(img_size=224, num_classes=num_classes, type=args.modeltype)

    elif args.dataset == "affectnet8":
        datapath = './data/AffectNet_8/'
        num_classes = 8
        traindir = os.path.join(datapath, 'train')
        valdir = os.path.join(datapath, 'valid')
        train_dataset = datasets.ImageFolder(traindir, transform=data_transforms_train)
        val_dataset = datasets.ImageFolder(valdir, transform=data_transforms_val)
        model = ORSA(img_size=224, num_classes=num_classes, type=args.modeltype)

    elif args.dataset == "occlu-FER":
        datapath = './data/Occlu-FER/'
        num_classes = 8
        traindir = os.path.join(datapath, 'train')
        valdir = os.path.join(datapath, 'valid')
        train_dataset = datasets.ImageFolder(traindir, transform=data_transforms_train)
        val_dataset = datasets.ImageFolder(valdir, transform=data_transforms_val)
        model = ORSA(img_size=224, num_classes=num_classes, type=args.modeltype)

    else:
        return print('dataset name is not correct')

    print('Train set size:', train_dataset.__len__())
    print('Validation set size:', val_dataset.__len__())
    
    if args.dataset == 'raf':
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               num_workers=args.workers,
                                               shuffle=True,
                                               pin_memory=True)
    else:
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                               sampler=ImbalancedDatasetSampler(train_dataset),
                                               batch_size=args.batch_size,
                                               num_workers=args.workers,
                                               shuffle=False,
                                               pin_memory=True)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                            batch_size=args.val_batch_size,
                                            num_workers=args.workers,
                                            shuffle=False,
                                            pin_memory=True)

    print("batch_size:", args.batch_size)

    if args.optimizer == 'adamw':
        base_optimizer = torch.optim.AdamW
    elif args.optimizer == 'adam':
        base_optimizer = torch.optim.Adam
    elif args.optimizer == 'sgd':
        base_optimizer = torch.optim.SGD
    else:
        raise ValueError("Optimizer not supported.")

    optimizer = SAM(model.parameters(), base_optimizer, lr=args.lr, rho=0.05, adaptive=False,)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)

    model = torch.nn.DataParallel(model)
    model = model.cuda()
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    print('Total Parameters: %.3fM' % parameters)

    CE_criterion = torch.nn.CrossEntropyLoss()
    lsce_criterion = LabelSmoothingCrossEntropy(smoothing=0.2)
    DRAELoss_criterion = DRAELoss()

    best_acc = 0
    for i in range(1, args.epochs + 1):
        train_loss = 0.0
        correct_sum = 0
        iter_cnt = 0
        start_time = time()
        model.train()
        for batch_i, (imgs, targets) in enumerate(train_loader):
            iter_cnt += 1
            optimizer.zero_grad()
            imgs = imgs.cuda()
            outputs, features = model(imgs)
            targets = targets.cuda()

            CE_loss = CE_criterion(outputs, targets)
            lsce_loss = lsce_criterion(outputs, targets)
            DRAE_loss = DRAELoss_criterion(outputs, targets)
            
            loss = CE_loss
            if(i > args.use_drae):
                loss += DRAE_loss * args.weight_drae
            else:
                loss += 2 * lsce_loss
            loss.backward()
            optimizer.first_step(zero_grad=True)

            # second forward-backward pass
            outputs, features = model(imgs)
            CE_loss = CE_criterion(outputs, targets)
            lsce_loss = lsce_criterion(outputs, targets)
            DRAE_loss = DRAELoss_criterion(outputs, targets)
            
            loss = CE_loss
            if(i > args.use_drae):
                loss += DRAE_loss * args.weight_drae
            else:
                loss += 2 * lsce_loss
            loss.backward() # make sure to do a full forward pass
            optimizer.second_step(zero_grad=True)

            train_loss += loss
            _, predicts = torch.max(outputs, 1)
            correct_num = torch.eq(predicts, targets).sum()
            correct_sum += correct_num

        train_acc = correct_sum.float() / float(train_dataset.__len__())
        train_loss = train_loss / iter_cnt
        elapsed = (time() - start_time) / 60

        print('[Epoch %d] Train time:%.2f, Training accuracy:%.4f. Loss: %.3f LR:%.6f' %
              (i, elapsed, train_acc, train_loss, optimizer.param_groups[0]["lr"]))

        scheduler.step()

        pre_labels = []
        gt_labels = []
        with torch.no_grad():
            val_loss = 0.0
            iter_cnt = 0
            bingo_cnt = 0
            model.eval()
            for batch_i, (imgs, targets) in enumerate(val_loader):
                outputs, features = model(imgs.cuda())
                targets = targets.cuda()

                CE_loss = CE_criterion(outputs, targets)
                loss = CE_loss

                val_loss += loss
                iter_cnt += 1
                _, predicts = torch.max(outputs, 1)
                correct_or_not = torch.eq(predicts, targets)
                bingo_cnt += correct_or_not.sum().cpu()
                pre_labels += predicts.cpu().tolist()
                gt_labels += targets.cpu().tolist()

            val_loss = val_loss / iter_cnt
            val_acc = bingo_cnt.float() / float(val_dataset.__len__())
            val_acc = np.around(val_acc.numpy(), 4)
            f1 = f1_score(pre_labels, gt_labels, average='macro')
            total_socre = 0.67 * f1 + 0.33 * val_acc

            print("[Epoch %d] Validation accuracy:%.4f, Loss:%.3f, f1 %4f, score %4f" % (
            i, val_acc, val_loss, f1, total_socre))

            if val_acc > 0.90 and val_acc > best_acc:
                torch.save({'iter': i,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(), },
                           os.path.join('./checkpoint', "epoch" + str(i) + "_acc" + str(val_acc) + ".pth"))
                print('Model saved.')
            if val_acc > best_acc:
                best_acc = val_acc
                print("best_acc:" + str(best_acc))


if __name__ == "__main__":
    run_training()
