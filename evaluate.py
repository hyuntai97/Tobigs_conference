import torch
import numpy as np
import pandas as pd
import os
import joblib
import matplotlib.pyplot as plt

def MAPEval(y_pred, y_true):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def inference(model, target_feature, target_len, input_window, data_root, data_name, savedir, logdir, device):
    og_data_path = os.path.join(data_root, data_name)
    df = pd.read_csv(og_data_path)

    scaler = joblib.load(os.path.join(logdir, 'scaler.pkl'))

    pred = np.array(df.iloc[-(input_window + target_len):-target_len, :][target_feature])
    pred = pred.reshape(-1,1)
    pred_std = scaler.transform(pred)

    predict = model.predict(torch.tensor(pred_std).to(device).float(), target_len)

    predict = scaler.inverse_transform(predict.reshape(-1,1))
    real = df[target_feature].to_numpy()

    ax = plt.plot(range(100), real[-100:], label='real')
    ax = plt.plot(range(100-target_len, 100), predict, label='predict')
    ax.set_title('test set')
    ax.legend()








    # print('\t2. Feature Importance: GAIN')
    # ax = lgb.plot_importance(model, max_num_features=20, importance_type='gain')
    # ax.set(title=f'Feature Importance (gain)',
    #       xlabel='Feature Importance',
    #       ylabel='Features')
    # ax.figure.savefig(f'{TRAIN_DIR}feature_importance_gain.png', dpi=100, bbox_inches='tight')
    # plt.clf()






    

