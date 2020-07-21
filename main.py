from model import MSCNN
import torch.nn as nn
import argparse
import math
from tqdm import tqdm
from utils import *
import torch
import time
import Optim
import os


def evaluate(data, X, Y, model, evaluateL2, evaluateL1, batch_size):
    model.eval()
    total_loss = 0
    total_loss_l1 = 0
    n_samples = 0
    predict = None
    test = None

    for X, Y in data.get_batches(X, Y, batch_size, False):
        output = model(X)
        if predict is None:
            predict = output
            test = Y
        else:
            predict = torch.cat((predict, output))
            test = torch.cat((test, Y))

        scale = data.scale.expand(output.size(0), data.m)
        total_loss += evaluateL2(output * scale, Y * scale).item()
        total_loss_l1 += evaluateL1(output * scale, Y * scale).item()
        n_samples += (output.size(0) * data.m)
    rse = math.sqrt(total_loss / n_samples) / data.rse
    rae = (total_loss_l1 / n_samples) / data.rae

    predict = predict.data.cpu().numpy()
    Ytest = test.data.cpu().numpy()

    sigma_p = (predict).std(axis=0)
    sigma_g = (Ytest).std(axis=0)
    mean_p = predict.mean(axis=0)
    mean_g = Ytest.mean(axis=0)
    index = (sigma_g != 0)
    correlation = ((predict - mean_p) * (Ytest - mean_g)).mean(axis=0) / (sigma_p * sigma_g)
    correlation = (correlation[index]).mean()

    return rse, rae, correlation, predict, Ytest




def train(data, X, Y, model, criterion, optim, batch_size):
    model.train()
    total_loss = 0
    n_samples = 0
    for X, Y in tqdm(data.get_batches(X, Y, batch_size, True)):
        if X.size(0)<=8:
            continue
        model.zero_grad()
        output = model(X)
        scale = data.scale.expand(output.size(0), data.m)
        loss = criterion(output * scale, Y * scale)
        loss.backward(retain_graph=True)
        grad_norm = optim.step()
        total_loss += loss.item()
        n_samples += (output.size(0) * data.m)
    return total_loss / n_samples


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Time series forecasting')
    parser.add_argument('--data', type=str, default='./data/solar_AL.txt',
                        help='the path of the dataset ')
    parser.add_argument('--horizon', type=int, default=3,
                        help='horizon of data')
    parser.add_argument('--cuda', type=bool,default=True,
                        help='running on the GPU ')
    parser.add_argument('--steps', type=int, default=24,
                        help='time steps in one day')
    parser.add_argument('--days', type=int, default=7,
                        help='How many days of data to predict')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='data size of each batch')
    parser.add_argument('--gpu', type=str, default='0',
                        help='which GPU to be use')
    parser.add_argument('--skip', type=int, default=10,
                        help='how many steps will be skipped in skip-CNN')
    parser.add_argument('--C_nums', type=int, default=100,
                        help='the number of kernel in CNN')
    parser.add_argument('--C_steps', type=int, default=6,
                        help='The time span of the convolution kernel')
    parser.add_argument('--save', type=str, default='./MSCNN_model.pt',
                        help='The path to save the model')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--normalize', type=int, default=2)
    parser.add_argument('--seed', type=int, default=54321)
    args = parser.parse_args()


    window = int(args.steps*args.days)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device_ids = range(torch.cuda.device_count())

    # Set the random seed manually for reproducibility.
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
        else:
            torch.cuda.manual_seed(args.seed)
    Data = Data_utility(args.data, 0.6, 0.2, args.cuda, args.horizon, window,args.days, args.steps, args.normalize)
    model = MSCNN(window, args.steps, args.skip, Data.train[0].shape[1], args.C_nums, args.C_steps, args.cuda)
    criterion = nn.MSELoss(size_average=False)
    evaluateL2 = nn.MSELoss(size_average=False)
    evaluateL1 = nn.L1Loss(size_average=False)
    best_val = np.inf
    if args.cuda:
        model.cuda()
        criterion = criterion.cuda()
        evaluateL1 = evaluateL1.cuda()
        evaluateL2 = evaluateL2.cuda()
    if args.cuda:
        model = torch.nn.DataParallel(model)
    optim = Optim.Optim(
        model.parameters(), 'adam', 0.001, 10,
    )

    try:
        print('begin training')
        train_start_time=time.time()
        for epoch in range(1, args.epochs + 1):
            epoch_start_time = time.time()

            train_loss = train(Data, Data.train[0], Data.train[1], model, criterion, optim, args.batch_size)
            val_loss, val_rae, val_corr, _, _ = evaluate(Data, Data.valid[0], Data.valid[1], model, evaluateL2, evaluateL1,
                                                   args.batch_size);
            print(
                '| end of epoch {:3d} | time: {:5.2f}s | train_loss {:5.4f} | valid rse {:5.4f} | valid rae {:5.4f} | valid corr  {:5.4f}'.format(
                    epoch, (time.time() - epoch_start_time), train_loss, val_loss, val_rae, val_corr))

            # Save the model if the validation loss is the best we've seen so far.
            if val_loss < best_val:
                with open(args.save, 'wb') as f:
                    torch.save(model, f)
                best_val = val_loss
            if epoch % 5 == 0:
                test_acc, test_rae, test_corr, _, _ = evaluate(Data, Data.test[0], Data.test[1], model, evaluateL2,
                                                         evaluateL1,args.batch_size);
                print("test rse {:5.4f} | test rae {:5.4f} | test corr {:5.4f}".format(test_acc, test_rae, test_corr))
        print('traning_time:'+str(time.time()-train_start_time))
        training_time = str(time.time()-train_start_time)
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

    # Load the best saved model.
    with open(args.save, 'rb') as f:
        model = torch.load(f)
    test_start_time = time.time()
    test_acc, test_rae, test_corr,test_predict, test_ytest = evaluate(Data, Data.test[0], Data.test[1], model, evaluateL2, evaluateL1,
                                             args.batch_size);
    print("test rse {:5.4f} | test rae {:5.4f} | test corr {:5.4f}".format(test_acc, test_rae, test_corr))
    print('test_time:' + str(time.time() - test_start_time))
    test_time = str(time.time() - test_start_time)

    #Save the result
    np.save(args.data.replace('data','output').replace('.txt',str(args.horizon)+'_predict.npy'),test_predict)
    np.save(args.data.replace('data', 'output').replace('.txt', str(args.horizon) + '_ytest.npy'), test_ytest)
    with open(args.data.replace('data','time').replace('.txt',str(args.horizon)+'.txt'), 'w') as f:
        f.write('traing_time=' + training_time)
        f.write('test_time=' + test_time)
    with open(args.data.replace('data', 'result').replace('.txt', str(args.horizon)+'.txt'), 'w') as f:
        f.write('rse=' + str(test_acc)+'\n')
        f.write('rae=' + str(test_rae) + '\n')
        f.write('corr=' + str(test_corr) + '\n')









