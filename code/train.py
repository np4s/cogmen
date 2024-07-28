import argparse
import time
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from model import COGMEN
from dataloader import get_IEMOCAP_loaders
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
from comet_ml import Experiment
from comet_ml.integration.pytorch import log_model

experiment = Experiment(
  api_key="",
  project_name="cogmen",
  workspace="np4s"
)

def train_or_eval_model(model, loss_f, dataloader, epoch=0, train_flag=False, optimizer=None, cuda_flag=False, args=None,
                              test_label=False):
    losses, preds, labels, masks = [], [], [], []

    if train_flag:
        model.train()
    else:
        model.eval()

    step = 0
    for data in dataloader:
        if train_flag:
            optimizer.zero_grad()

        textf, visuf, acouf, qmask, umask, label = [d.cuda() for d in data[:-1]] if cuda_flag else data[:-1]
        lengths = [(umask[j] == 1).nonzero().tolist()[-1][0] + 1 for j in range(len(umask))]
        
        # handles = []
        # if train_flag == True and args.modulation == True and step % args.tau == 0:
        #     for module in MODAL_GEN:
        #         handles.append(module.register_forward_hook(hook))
        
        log_prob = model(textf, lengths, qmask, acouf=acouf, visuf= visuf)
        label = torch.cat([label[j][:lengths[j]] for j in range(len(label))])
        loss = loss_f(log_prob, label)
        preds.append(torch.argmax(log_prob, 1).cpu().numpy())
        labels.append(label.cpu().numpy())
        losses.append(loss.item())
        
        if train_flag == True:
            loss.backward()

            # if args.modulation == True:

            #     modulation(model, log_prob, textf, qmask, umask, acouf, visuf, labels_, step, args)
            #     for handle in handles:
            #         handle.remove()

            optimizer.step()
            step += 1

    if preds != []:
        preds = np.concatenate(preds)
        labels = np.concatenate(labels)
    else:
        return [], [], float('nan'), float('nan'), [], [], float('nan')

    labels = np.array(labels)
    preds = np.array(preds)
    avg_loss = round(np.sum(losses) / len(losses), 4)
    avg_accuracy = round(accuracy_score(labels, preds) * 100, 2)
    avg_fscore = round(f1_score(labels, preds, average='weighted') * 100, 2)

    return avg_loss, avg_accuracy, labels, preds, avg_fscore

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--no_cuda', action='store_true', default=False, help='does not use GPU')
    parser.add_argument('--dataset', default='IEMOCAP', help='dataset to train and test')
    parser.add_argument('--data_dir', type=str, default='./data/iemocap/IEMOCAP_features.pkl', help='dataset dir')
    parser.add_argument('--modals', default='avl', help='modals to fusion: avl')
    parser.add_argument('--windowp', type=int, default=2, help='context window size for constructing edges in graph model for past utterances')
    parser.add_argument('--windowf', type=int, default=2, help='context window size for constructing edges in graph model for future utterances')
    parser.add_argument('--seqcontext-nlayer', type=int, default=2, help='number of layers in seqcontext')
    parser.add_argument('--gnn_nhead', type=int, default=2, help='number of heads in GNN')
    parser.add_argument('--epochs', type=int, default=60, metavar='E', help='number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, metavar='BS', help='batch size')
    parser.add_argument('--valid_rate', type=float, default=0.1, metavar='valid_rate', help='valid rate, 0.0/0.1')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR', help='learning rate')
    parser.add_argument('--l2', type=float, default=0.00001, metavar='L2', help='L2 regularization weight')
    parser.add_argument('--dropout', type=float, default=0.1, metavar='dropout', help='dropout rate')
    parser.add_argument('--class_weight', action='store_true', default=False, help='use class weights')
    parser.add_argument('--output', type=str, default='./outputs', help='saved model dir')
    parser.add_argument('--tensorboard', action='store_true', default=False, help='Enables tensorboard log')
    parser.add_argument('--test_label', action='store_true', default=False, help='whether do test only')
    parser.add_argument('--test_modal', default='l', help='whether do test only')
    parser.add_argument('--load_model', type=str, default='./outputs', help='trained model dir')
    parser.add_argument('--name', type=str, default='demo', help='Experiment name')
    parser.add_argument('--log_dir', type=str, default='log/', help='tensorboard save path')
    parser.add_argument('--beta', type=float, default=1, help='')
    parser.add_argument('--gamma', type=float, default=1, help='')
    parser.add_argument('--tau', type=float, default=1, help='')
    parser.add_argument('--modulation', action='store_true', default=False, help='Enables grad modulation')
    args = parser.parse_args()

    # log all the args
    print(args)
    experiment.log_parameters(vars(args))
    
    cuda_flag = torch.cuda.is_available() and not args.no_cuda
    
    if args.tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=args.log_dir)
    else:
        writer = None

    n_epochs = args.epochs
    batch_size = args.batch_size
    feat2dim = {'IS10': 1582, '3DCNN': 512, 'textCNN': 100, 'bert': 768, 'denseface': 342, 'MELD_text': 600, 'MELD_audio': 300}
    D_audio = feat2dim['IS10'] if args.dataset == 'IEMOCAP' else feat2dim['MELD_audio']
    D_visual = feat2dim['denseface']
    D_text = feat2dim['textCNN'] if args.dataset == 'IEMOCAP' else feat2dim['MELD_text']
    
    D_e = 100
    D_h = 100
    
    n_speakers, n_classes, class_weights, target_names = -1, -1, None, None
    if args.dataset == 'IEMOCAP':
        n_speakers, n_classes = 2, 6
        target_names = ['hap', 'sad', 'neu', 'ang', 'exc', 'fru']
        class_weights = torch.FloatTensor([1 / 0.086747,
                                           1 / 0.144406,
                                           1 / 0.227883,
                                           1 / 0.160585,
                                           1 / 0.127711,
                                           1 / 0.252668])
    
    model = COGMEN(args.modals, D_audio, D_visual, D_text, D_e, D_h, n_speakers, args.windowp, args.windowf, args.seqcontext_nlayer, args.gnn_nhead, n_classes, args.dropout, args.no_cuda)
    
    print('Running on the {} features........'.format(args.modals))

    if cuda_flag:
        print('Running on GPU')
        class_weights = class_weights.cuda()
        model.cuda()
    else:
        print('Running on CPU')


    loss_f = nn.NLLLoss(class_weights if args.class_weight else None)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)

    if args.dataset == 'IEMOCAP':
        train_loader, valid_loader, test_loader = get_IEMOCAP_loaders(data_path=args.data_dir,
                                                                      valid_rate=args.valid_rate,
                                                                      batch_size=batch_size,
                                                                      num_workers=0)
    else:
        train_loader, valid_loader, test_loader = None, None, None
        print("There is no such dataset")
    
    if args.test_label == False:
        best_fscore = None
        counter = 0

        for e in range(n_epochs):
            start_time = time.time()
            train_loss, train_acc, _, _, train_fscore = train_or_eval_model(model=model,
                                                                                loss_f=loss_f,
                                                                                dataloader=train_loader,
                                                                                epoch=e,
                                                                                train_flag=True,
                                                                                optimizer=optimizer,
                                                                                cuda_flag=cuda_flag,
                                                                                args=args)
            
            
            end_time = time.time()
            train_time = round(end_time-start_time, 2)

            start_time = time.time()
            with torch.no_grad():
                valid_loss, valid_acc, _, _, valid_fscore = train_or_eval_model(model=model,
                                                                                    loss_f=loss_f,
                                                                                    dataloader=valid_loader,
                                                                                    epoch=e,
                                                                                    cuda_flag=cuda_flag,
                                                                                    args=args)
                
            end_time = time.time()
            valid_time = round(end_time-start_time, 2)

            if args.tensorboard:
                writer.add_scalar('val/accuracy', valid_acc, e)
                writer.add_scalar('val/fscore', valid_fscore, e)
                writer.add_scalar('val/loss', valid_loss, e)
                writer.add_scalar('train/accuracy', train_acc, e)
                writer.add_scalar('train/fscore', train_fscore, e)
                writer.add_scalar('train/loss', train_loss, e)


            print('epoch: {}, train_loss: {}, train_acc: {}, train_fscore: {}, valid_loss: {}, valid_acc: {}, valid_fscore: {}, train_time: {} sec, valid_time: {} sec'. \
                    format(e, train_loss, train_acc, train_fscore, valid_loss, valid_acc, valid_fscore, train_time, valid_time))
            
            if best_fscore == None:
                best_fscore = valid_fscore
            elif valid_fscore > best_fscore:
                best_fscore = valid_fscore
                counter = 0
                path = os.path.join(args.output, args.dataset, args.modals)
                if not os.path.isdir(path): os.makedirs(path)
                torch.save(model.state_dict(), os.path.join(path, args.name+'.pth'))
            else:
                counter += 1
                if counter >= 10:
                    print("Early stopping")
                    break

        if args.tensorboard:
            writer.close()
            
            
    if args.test_label == True:
        model.load_state_dict(torch.load(args.load_model))
    else:
        model.load_state_dict(torch.load(os.path.join(args.output, args.dataset, args.modals, args.name+'.pth')))
    with torch.no_grad():
        test_loss, test_acc, test_label, test_pred, test_fscore = train_or_eval_model(model=model,
                                                                                            loss_f=loss_f,
                                                                                            dataloader=test_loader,
                                                                                            train_flag=True,
                                                                                            cuda_flag=cuda_flag,
                                                                                            args=args,
                                                                                            test_label=False)
    print('Test performance..')
    print('Loss {}, accuracy {}'.format(test_loss, test_acc))
    print(classification_report(test_label, test_pred, digits=4))
    print(confusion_matrix(test_label, test_pred))
    log_model(experiment, model, model_name=args.name, overwrite=True)