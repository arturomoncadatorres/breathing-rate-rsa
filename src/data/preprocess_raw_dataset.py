# -*- coding: utf-8 -*-
import pathlib
import click
import logging
import librosa
import pickle
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

import numpy as np
import wfdb

import src.signals.rr as rr
import src.signals.nn as nn
import src.signals.respiration as resp


#%%
@click.command()
@click.argument('input_filepath', type=click.Path(exists=False))
@click.argument('output_filepath', type=click.Path(exists=False))
def main(input_filepath='./data/raw/', output_filepath='./data/processed/', T_int=0.2, win_lengths_m=[1,3,5]):
    """ 
    Run data scripts to pre-process data files and
    save them as .pkl in output_filepath.
    
    Parameters
    ----------
    input_filepath: string or pathlib.Path
        Location of the data files (as downloaded using download_raw_dataset.py)
    output_filepath: string or pathlib.Path
        Location where the resulting dataset files will be saved (as .pkl)
    T_int: float [s]
        Interpolation interval. Defaults to 0.2
    win_lengths_m: list
        List with each element being the length of the windows
        (in minutes) for segmenting the signals. It defaults
        to [1, 3, 5], as in the original paper. 
        Notice that long lists will result in very large .pkl files.
        
    Returns
    -------
    None. Files are written in output_filepath
        
    References
    ----------
    - A. Schafer, K. W. Kratkly, "Estimation of Breathing Rate from Respiratory 
    Sinus Arrhythmia: Comparison of Various Methods", 2008.
    """    
    logger = logging.getLogger(__name__)
    logger.info('Pre-processing raw dataset...')
    
    # Convert paths to pathlib.Paths for ease of use.
    input_filepath = pathlib.Path(input_filepath)
    output_filepath = pathlib.Path(output_filepath)
    
    # Read the RECORDS file to fetch participants IDs
    # We will use these IDs to loop through all the data files.
    logger.info('\tReading participants IDs...')
    
    with open(str(input_filepath/'RECORDS')) as f:
        participants_ids = f.read().splitlines()
        
    logger.info('\t\tDONE!')
    
    
    # Loop and pre-process data of all participants.
    # We will store all the required data in a dictionary.
    for participant_id in participants_ids:
        
        logger.info('\tProcessing participant ' + participant_id + '...')
        
        # Initialize dictionary where the dataset will be stored.
        dataset = {}
        
        # Extract important parameters from the recording
        logger.info('\t\tExtracting parameters...')
        record = wfdb.rdrecord(str(input_filepath/participant_id))
        n_samples = record.sig_len
        fs = record.fs
        Ts = 1/fs

        # Get R peak annotations
        logger.info('\t\tGetting R peak annotations...')
        annotation = wfdb.rdann(str(input_filepath/participant_id), 'ecg')
        r_peaks_samples = annotation.sample
        r_peaks_t = r_peaks_samples * Ts # [samples] --> [s]
        
        # Compute NN intervals and perform interpolation to obtain 
        # NN curve.
        logger.info('\t\tComputing NN intervals..')
        nn_x, nn_y = rr.calculate_nn(r_peaks_t)
        last_time_sample = n_samples * Ts
        logger.info('\t\tInterpolating NN intervals...')
        nn_interp_x, nn_interp_y = nn.interpolate(nn_x, nn_y, from_time=0, to_time=last_time_sample, T_int=T_int)
        
        # Smooth respiration signal.
        logger.info('\t\tSmoothing respiration signal...')
        respiration = record.p_signal[:,0]
        respiration_smooth_x, respiration_smooth_y = resp.smooth(respiration, nn_interp_x, from_time=0, to_time=last_time_sample, Ts=Ts, T_int=T_int)
        
        # Trim signals into smaller segments
        # Notice that for the respiration_smooth signal, T_int its
        # practically its sampling period (and not the original Ts).
        respiration = record.p_signal[:,0]
        ecg = record.p_signal[:,1]
        for win_length_m in win_lengths_m:
            logger.info(f'\t\tTrimming signals with window length = {win_length_m} min...')    
            win_length_s = win_length_m * 60 # [min] --> [s]
            win_length_samples = round(win_length_s * fs)
            win_length_samples_resp_smooth = round(win_length_s * (1/T_int))
            
            ecg_trimmed = list(librosa.util.frame(ecg, frame_length=win_length_samples, hop_length=win_length_samples, axis=0))
            dataset['ecg_' + str(win_length_m)] = ecg_trimmed
            
            respiration_trimmed = list(librosa.util.frame(respiration, frame_length=win_length_samples, hop_length=win_length_samples, axis=0))
            dataset['respiration_' + str(win_length_m)] = respiration_trimmed
            
            respiration_smooth_x_trimmed = list(librosa.util.frame(respiration_smooth_x, frame_length=win_length_samples_resp_smooth, hop_length=win_length_samples_resp_smooth, axis=0))
            respiration_smooth_y_trimmed = list(librosa.util.frame(respiration_smooth_y, frame_length=win_length_samples_resp_smooth, hop_length=win_length_samples_resp_smooth, axis=0))
            dataset['respiration_smooth_x_trimmed_' + str(win_length_m)] = respiration_smooth_x_trimmed
            dataset['respiration_smooth_y_trimmed_' + str(win_length_m)] = respiration_smooth_y_trimmed
        
        
        # Pack output and save in .pkl.
        logger.info('\t\tPacking the rest of the output...')    
        dataset['id'] = participant_id
        dataset['fs'] = record.fs
        dataset['Ts'] = Ts
        dataset['n_samples'] = record.sig_len
        dataset['t'] = np.arange(0, n_samples*Ts, Ts)
        dataset['T_int'] = T_int
        
        dataset['respiration'] = respiration
        dataset['ecg'] = ecg
        dataset['annotations'] = annotation.symbol
        dataset['respiration_smooth_x'] = respiration_smooth_x
        dataset['respiration_smooth_y'] = respiration_smooth_y
        
        logger.info('\t\tSaving .pkl file...')    
        with open(str(output_filepath/(participant_id + '.pkl')), 'wb') as f:
            pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
        
    return None



if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
