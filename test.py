import warnings
warnings.filterwarnings("ignore")
import torch.utils.data as data
from torchvision import transforms
import os
import argparse
import torchvision.datasets as datasets

from utils import *
from models.net import ORSA, ORSA_N
from sklearn.metrics import confusion_matrix


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='rafdb', help='dataset [rafdb, affectnet7, affectnet8, occlu-fer]')
    parser.add_argument('-c', '--checkpoint', type=str, default=None, help='Pytorch checkpoint file path')
    parser.add_argument('--batch_size', type=int, default=100, help='Batch size.')
    parser.add_argument('--modeltype', type=str, default='large', help='small or base or large')
    parser.add_argument('--workers', type=int, default=2, help='Number of data loading workers (default: 4)')
    parser.add_argument('--gpu', type=str, default='0', help='assign multi-gpus by comma concat')
    parser.add_argument('-p', '--plot_cm', action="store_true", help="Ploting confusion matrix.")
    return parser.parse_args()

def test():
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    print("Work on GPU: ", os.environ['CUDA_VISIBLE_DEVICES'])

    data_transforms_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # transforms.RandomErasing(p=1, scale=(0.1, 0.1), ratio=(1, 1), value=0)
    ])

    if args.dataset == "rafdb":
        datapath = './data/raf/valid'
        num_classes = 7
        test_dataset = datasets.ImageFolder(datapath, transform=data_transforms_test)
        model = ORSA(img_size=224, num_classes=num_classes, type=args.modeltype)

    elif args.dataset == "affectnet7":
        datapath = './data/AffectNet_7/valid'
        num_classes = 7
        test_dataset = datasets.ImageFolder(datapath, transform=data_transforms_test)
        model = ORSA(img_size=224, num_classes=num_classes, type=args.modeltype)

    elif args.dataset == "affectnet8":
        datapath = './data/AffectNet_8/valid'
        num_classes = 8
        test_dataset = datasets.ImageFolder(datapath, transform=data_transforms_test)
        model = ORSA(img_size=224, num_classes=num_classes, type=args.modeltype)

    elif args.dataset == "occlu-fer":
        datapath = './data/Occlu-FER/valid'
        num_classes = 8
        test_dataset = datasets.ImageFolder(datapath, transform=data_transforms_test)
        model = ORSA(img_size=224, num_classes=num_classes, type=args.modeltype)

    test_size = test_dataset.__len__()
    print('Test set size:', test_size)

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                             batch_size=args.batch_size,
                                             num_workers=args.workers,
                                             shuffle=False,
                                             pin_memory=True)


    print("Loading pretrained weights...", args.checkpoint)
    checkpoint = torch.load(args.checkpoint)
    checkpoint = checkpoint["model_state_dict"]
    model = load_pretrained_weights(model, checkpoint)
    model = model.cuda()

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    print('Total Parameters: %.3fM' % parameters)

    pre_labels = []
    gt_labels = []
    with torch.no_grad():
        bingo_cnt = 0
        model.eval()
        for batch_i, (imgs, targets) in enumerate(test_loader):
            outputs, features = model(imgs.cuda())
            targets = targets.cuda()
            _, predicts = torch.max(outputs, 1)
            correct_or_not = torch.eq(predicts, targets)
            bingo_cnt += correct_or_not.sum().cpu()
            pre_labels += predicts.cpu().tolist()
            gt_labels += targets.cpu().tolist()

        acc = bingo_cnt.float() / float(test_size)
        acc = np.around(acc.numpy(), 4)
        print(f"Test accuracy: {acc:.4f}.")
        cm = confusion_matrix(gt_labels, pre_labels)
        print(cm)

        if args.dataset == "rafdb":
            labels_name = ['SU', 'FE', 'DI', 'HA', 'SA', 'AN', "NE"]
        elif args.dataset == "affectnet7":
            labels_name = ['NE', 'HA', 'SA', 'SU', 'FE', 'DI', "AN"]
        elif args.dataset == "affectnet8":
            labels_name = ['NE', 'HA', 'SA', 'SU', 'FE', 'DI', "AN", "CO"]
        elif args.dataset == "occlu-fer":
            labels_name = ['AN', 'DI', 'FE', 'HA', 'NE', "SA", "SU"]

        all_features = np.concatenate(all_features, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        sample_size = 2000 
        if len(all_features) > sample_size:
            indices = np.random.choice(len(all_features), sample_size, replace=False)
            all_features = all_features[indices]
            all_labels = all_labels[indices]
        
        dataset_names = {
            'rafdb': 'RAF-DB',
            'affectnet7': 'AffectNet-7',
            'affectnet8': 'AffectNet-8',
            'occlu-fer': 'Occlu-FER'
        }

        plot_confusion_matrix(cm, labels_name, dataset_names[args.dataset], acc)

        plot_tsne(
            all_features, 
            all_labels, 
            labels_name=labels_name, 
            dataset_name=dataset_names[args.dataset],
            save_path=f'tsne_{args.dataset}.png'
        )

if __name__ == "__main__":                    
    test()
