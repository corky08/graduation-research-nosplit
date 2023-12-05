import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

def cal_curve(ori_prob, scaled, label):
    ori_prob_true, ori_prob_pred = calibration_curve(y_true=label, y_prob=ori_prob, n_bins=20)
    sca_prob_true, sca_prob_pred = calibration_curve(y_true=label, y_prob=scaled, n_bins=20)

    fig, ax1 = plt.subplots()
    ax1.plot(ori_prob_pred, ori_prob_true, marker='s', label='calibration plot', color='skyblue') # キャリプレーションプロットを作成
    ax1.plot(sca_prob_pred, sca_prob_true, marker='v', label='calibration plot', color='green') # キャリプレーションプロットを作成
    ax1.plot([0, 1], [0, 1], linestyle='--', label='ideal', color='limegreen') # 45度線をプロット
    ax1.legend(bbox_to_anchor=(1.12, 1), loc='upper left')
    # ax2 = ax1.twinx() # 2軸を追加
    # ax2.hist(prob, bins=20, histtype='step', color='orangered') # スコアのヒストグラムも併せてプロット
    plt.show()