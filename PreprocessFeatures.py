import os

import numpy as np
import pandas as pd
from tqdm import tqdm

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, MinMaxScaler

import librosa

import joblib

def construct_melspectogram_images(audio, sr, file_name, model_res=(224, 224), img_dpi=120):
    S = librosa.feature.melspectrogram(y=audio, sr=sr)
            
    S_dB = librosa.power_to_db(S, ref=np.max)
    
    fig = plt.figure(figsize=(model_res[0]/img_dpi, model_res[1]/img_dpi))
    img = librosa.display.specshow(S_dB, sr=sr)
    plt.savefig(fname=os.path.join(os.path.join('Data', f'{type}_img'), file_name+'.png'), transparent=True, dpi=img_dpi, 
                format='png', pad_inches=0, bbox_inches='tight')

    plt.close()
    return S_dB    

def extract_audio_features(construct_spectograms=True, n_mfcc=20, model_res=(224, 224), img_dpi=120):
    for type in ['train', 'test']:
        columns = ['name']
        for i in range(20):
            columns.append(f'mfcc{i+1}')
        for i in range(18):
            columns.append(f'rolloff{i+1}')
        columns += ['rms', 'centr', 'cross_rate']

        features = pd.DataFrame(columns=columns)

        for i, file in enumerate(tqdm(os.listdir(os.path.join('Data', f'{type}_audio')))):
            name, _ = os.path.splitext(os.fsdecode(file))
            
            path = os.path.join('Data', f'{type}_audio', name + '.wav')
            audio, sr = librosa.load(path=path)
            audio, trim_range = librosa.effects.trim(y=audio)

            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
            mfcc = np.mean(mfcc, axis=1)

            rms = librosa.feature.rms(y=audio)
            rms = np.mean(rms)

            centr = librosa.feature.spectral_centroid(y=audio, sr=sr)
            centr = np.mean(centr)

            cross_rate = librosa.feature.zero_crossing_rate(y=audio)
            cross_rate = np.mean(cross_rate)
        
            spec_rolloff = np.array([librosa.feature.spectral_rolloff(y=audio, sr=sr, roll_percent=i) for i in np.arange(0.05, 0.95, 0.05)])
            spec_rolloff = np.mean(spec_rolloff, axis=-1).flatten()
            
            features.loc[i] = np.concat([[name], mfcc, spec_rolloff, [rms], [centr], [cross_rate]])

            if construct_spectograms:
                construct_melspectogram_images(audio, sr, name, model_res, img_dpi)

        if type == 'train':
            std_scaler, min_max_scaler = StandardScaler(), MinMaxScaler()
            std_scalers, min_max_scalers = [], []
            
            for col in features.columns[1:]:
                features[col] = min_max_scaler.fit_transform(features[col].values.reshape(-1, 1))
                features[col] = std_scaler.fit_transform(features[col].values.reshape(-1, 1))
                std_scalers.append(std_scaler)
                min_max_scalers.append(min_max_scaler)

            joblib.dump(min_max_scalers, os.path.join('Utils', 'min_max_scalers.gz'))
            joblib.dump(std_scalers, os.path.join('Utils', 'std_scalers.gz'))
        
        elif type == 'test':
            # min_max_scalers = joblib.load(os.path.join('Utils', 'min_max_scalers.gz'))
            # std_scalers = joblib.load(os.path.join('Utils', 'std_scalers.gz'))
            for scaler_id, col in enumerate(features.columns[1:]):
                # min_max_scaler = min_max_scalers[scaler_id]
                # std_scaler = std_scalers[scaler_id]

                min_max_scaler = MinMaxScaler()
                std_scaler = StandardScaler()

                features[col] = std_scaler.fit_transform(min_max_scaler.fit_transform(features[col].values.reshape(-1, 1)))
        
        features.to_csv(os.path.join('Utils', f'{type}_features.csv'), sep=',', index=False)

if __name__ == '__main__':
    img_dpi = 120
    model_res = (224, 224)
    
    extract_audio_features(construct_spectograms=False, model_res=model_res, img_dpi=img_dpi)