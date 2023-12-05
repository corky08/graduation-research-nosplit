import torch
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
from temperature_scaling import ModelWithTemperature
from isotonic import isotonic
from cal_curve import cal_curve

def discretize(proba):
    threshold = torch.Tensor([0.5]) # 0か1かを分ける閾値を0.5に設定
    discretized = (proba >= threshold).int() # 閾値未満で0、以上で1に変換
    return discretized

def train(dataloader, model, criterion, optimizer, epoch):
    y_axis_list = []
    with tqdm(range(epoch)) as pbar_epoch:
        for e in pbar_epoch:
            loss_sum = 0
            pbar_epoch.set_description("[Epoch %d]" % (e+1))
            with tqdm(enumerate(dataloader), total=len(dataloader), leave=False) as pbar_loss:
                torch.manual_seed(0)
                for _, (batch, label) in pbar_loss: #エポックのループの内側で、さらにデータローダーによるループ

                    optimizer.zero_grad()

                    t_p = model(batch)

                    label = label.unsqueeze(1) #損失関数に代入するために次元を調節する処理(気にしなくて大丈夫です)
                    loss = criterion(t_p,label)
                    loss_sum += loss
                    loss.backward()

                    optimizer.step()

            loss_avg = loss_sum / len(dataloader)
            y_axis_list.append(loss_avg.detach().numpy())#プロット用のy軸方向リストに損失の値を代入

            if (e+1) % 10 == 0:#10エポック毎に損失の値を表示
                print("epoch: %d  loss: %f" % (e+1 ,float(loss_avg.detach())))

        x_axis_list = [num for num in range(epoch)]#損失プロット用x軸方向リスト

    # 損失の描画
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.plot(x_axis_list,y_axis_list)
    plt.show()

def test(dataloader, model, loss_fn):
    size = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        torch.manual_seed(0)
        for X, y in dataloader:
            y = y.unsqueeze(1)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            pred = torch.sigmoid(pred)
            correct += (discretize(pred) == y).type(torch.float).sum().item()
    test_loss /= size
    correct /= len(dataloader.dataset)
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

def KL(test_loader, model):
    treat_size = 0
    KL_sum = 0

    with torch.no_grad():
        for X, y in test_loader:
            y = torch.flatten(y)
            prob = torch.flatten(torch.sigmoid(model(X)))
            treat_size += y.sum().item()
            KL_sum += ((np.log(prob+1e-12)-np.log(1-prob+1e-12))*y.numpy()).sum().item()

    # logits_list = []
    # labels_list = []
    # scaled_list = []
    # scaled_model = ModelWithTemperature(model)
    # scaled_model.set_temperature(test_loader)
    # scaled_model.eval()
    # with torch.no_grad():
    #     for input, label in test_loader:
    #         input = input
    #         logits = torch.flatten(torch.sigmoid(model(input)))
    #         scaled = torch.flatten(torch.sigmoid(scaled_model(input)))
    #         logits_list.append(logits)
    #         labels_list.append(label)
    #         scaled_list.append(scaled)
    #     logits = torch.cat(logits_list)
    #     labels = torch.cat(labels_list)
    #     scaled = torch.cat(scaled_list)

    # with torch.no_grad():
    #     for X, y in test_loader:
    #         y = y.unsqueeze(1)
    #         prob = torch.sigmoid(scaled_model(X))
    #         treat_size += y.sum().item()
    #         KL_sum += ((np.log(prob.numpy()+1e-12)-np.log(1-prob.numpy()+1e-12))*y.numpy()).sum().item()
    # cal_curve(logits, scaled, labels)

    # iso_reg = isotonic(test_loader, model)
    # with torch.no_grad():
    #     for X, y in test_loader:
    #         y = torch.flatten(y)
    #         prob = torch.flatten(torch.sigmoid(model(X)))
    #         cal = iso_reg.predict(prob)
    #         treat_size += y.sum().item()
    #         KL_sum += ((np.log(cal+1e-12)-np.log(1-cal+1e-12))*y.numpy()).sum().item()

    return KL_sum / treat_size

def nonRCT_KL(pros_loader, test_loader, in_dim, pros_model, model):
    treat_size = 0
    KL_sum = 0

    with torch.no_grad():
        for X, y in test_loader:
            y = torch.flatten(y)
            pros = torch.flatten(torch.sigmoid(pros_model(X[:, :in_dim])))
            prob = torch.flatten(torch.sigmoid(model(X)))
            treat_size += y.sum().item()
            KL_sum += (((np.log(prob+1e-12)-np.log(1-prob+1e-12))-(np.log(pros+1e-12)-np.log(1-pros+1e-12)))*y.numpy()).sum().item()

    # logits_list = []
    # labels_list = []
    # scaled_list = []
    # pros_scaled = ModelWithTemperature(pros_model)
    # pros_scaled.set_temperature(pros_loader)
    # scaled_model = ModelWithTemperature(model)
    # scaled_model.set_temperature(test_loader)
    # pros_scaled.eval()
    # scaled_model.eval()
    # with torch.no_grad():
    #     for input, label in test_loader:
    #         input = input
    #         logits = torch.flatten(torch.sigmoid(model(input)))
    #         scaled = torch.flatten(torch.sigmoid(scaled_model(input)))
    #         logits_list.append(logits)
    #         labels_list.append(label)
    #         scaled_list.append(scaled)
    #     logits = torch.cat(logits_list)
    #     labels = torch.cat(labels_list)
    #     scaled = torch.cat(scaled_list)

    # with torch.no_grad():
    #     for X, y in test_loader:
    #         y = y.unsqueeze(1)
    #         pros = torch.sigmoid(pros_scaled(X[:, :in_dim]))
    #         prob = torch.sigmoid(scaled_model(X))
    #         treat_size += y.sum().item()
    #         KL_sum += (((np.log(prob+1e-12)-np.log(1-prob+1e-12))-(np.log(pros+1e-12)-np.log(1-pros+1e-12)))*y.numpy()).sum().item()
    # cal_curve(logits, scaled, labels)

    # iso_reg = isotonic(test_loader, model)
    # pros_iso_reg = isotonic(pros_loader, pros_model)
    # with torch.no_grad():
    #     for X, y in test_loader:
    #         y = torch.flatten(y)
    #         prob = torch.flatten(torch.sigmoid(model(X)))
    #         pros = torch.flatten(torch.sigmoid(pros_model(X)))
    #         pros = pros_iso_reg.predict(pros)
    #         cal = iso_reg.predict(prob)
    #         treat_size += y.sum().item()
    #         KL_sum += (((np.log(prob+1e-12)-np.log(1-prob+1e-12))-(np.log(pros+1e-12)-np.log(1-pros+1e-12)))*y.numpy()).sum().item()

    return KL_sum / treat_size