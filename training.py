# train.py
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import gc

from preprocessing import Preprocessor
from custom_split import PurgedGroupTimeSeriesSplit
from model import create_ae_mlp

# Weighted average from Donate et al. formula
def weighted_average(a):
    w = []
    n = len(a)
    for j in range(1, n + 1):
        j = 2 if j == 1 else j
        w.append(1 / (2**(n + 1 - j)))
    return np.average(a, weights=w)

def main():
    TEST = False
    # Example usage
    data_path = '../train.csv'
    
    print("Loading & preprocessing data...")
    prep = Preprocessor(frac_d=0.5, frac_thresh=1e-5)
    df = prep.load_data(path=data_path, test=TEST)
    X, y, date, weight, sw, features = prep.prep_data(df)

    resp_cols = ['resp','resp_1','resp_2','resp_3','resp_4']
    del df

    # Hyperparams
    params = {
        'num_columns': len(features),
        'num_labels' : 5,
        'hidden_units': [96, 96, 896, 448, 448, 256],
        'dropout_rates': [
            0.03527936123679956, 
            0.038424974585075086, 
            0.42409238408801436, 
            0.10431484318345882, 
            0.49230389137187497, 
            0.32024444956111164, 
            0.2716856145683449, 
            0.4379233941604448
        ],
        'ls': 0,
        'lr': 1e-3
    }

    # Train
    n_splits = 5
    group_gap = 31
    gkf = PurgedGroupTimeSeriesSplit(n_splits=n_splits, group_gap=group_gap)
    scores = []
    batch_size = 4096

    print("Starting cross-validation training...")
    for fold, (tr, te) in enumerate(gkf.split(X, y, date)):
        ckp_path = f'Model_{fold}.hdf5'
        model = create_ae_mlp(**params)
        ckp = ModelCheckpoint(
            ckp_path, monitor='val_action_AUC', verbose=0, 
            save_best_only=True, save_weights_only=True, mode='max'
        )
        es = EarlyStopping(
            monitor='val_action_AUC', min_delta=1e-4, patience=10, 
            mode='max', baseline=None, restore_best_weights=True, verbose=0
        )

        history = model.fit(
            X[tr], [X[tr], y[tr], y[tr]],
            validation_data=(X[te], [X[te], y[te], y[te]]),
            sample_weight=sw[tr],
            epochs=100,
            batch_size=batch_size,
            callbacks=[ckp, es],
            verbose=0
        )

        # Best score
        best_score = max(history.history['val_action_AUC'])
        print(f"Fold {fold} ROC AUC: {best_score}")
        scores.append(best_score)

        K.clear_session()
        del model
        gc.collect()

    print("All folds done!")
    print("Scores:", scores)
    print("Weighted Average CV Score:", weighted_average(scores))

if __name__ == "__main__":
    main()
