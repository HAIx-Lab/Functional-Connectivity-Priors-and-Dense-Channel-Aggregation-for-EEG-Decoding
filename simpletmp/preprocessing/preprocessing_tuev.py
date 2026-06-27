import mne
import numpy as np
import os
import pickle
from tqdm import tqdm
import concurrent.futures
from functools import partial
import shutil

# Suppress MNE info messages to keep the parallel progress bars clean
mne.set_log_level('WARNING')

"""
https://github.com/Abhishaike/EEG_Event_Classification
"""

def BuildEvents(signals, times, EventData):
    [numEvents, z] = EventData.shape  # numEvents is equal to # of rows of the .rec file
    fs = 200.0
    [numChan, numPoints] = signals.shape
    features = np.zeros([numEvents, numChan, int(fs) * 5])
    offending_channel = np.zeros([numEvents, 1])  # channel that had the detected thing
    labels = np.zeros([numEvents, 1])
    offset = signals.shape[1]
    signals = np.concatenate([signals, signals, signals], axis=1)
    for i in range(numEvents):  # for each event
        chan = int(EventData[i, 0])  # chan is channel
        start = np.where((times) >= EventData[i, 1])[0][0]
        end = np.where((times) >= EventData[i, 2])[0][0]
        features[i, :] = signals[
            :, offset + start - 2 * int(fs) : offset + end + 2 * int(fs)
        ]
        offending_channel[i, :] = int(chan)
        labels[i, :] = int(EventData[i, 3])
    return [features, offending_channel, labels]


def convert_signals(signals, Rawdata):
    signal_names = {
        k: v
        for (k, v) in zip(
            Rawdata.info["ch_names"], list(range(len(Rawdata.info["ch_names"])))
        )
    }
    new_signals = np.vstack(
        (
            signals[signal_names["EEG FP1-REF"]] - signals[signal_names["EEG F7-REF"]],  # 0
            (signals[signal_names["EEG F7-REF"]] - signals[signal_names["EEG T3-REF"]]),  # 1
            (signals[signal_names["EEG T3-REF"]] - signals[signal_names["EEG T5-REF"]]),  # 2
            (signals[signal_names["EEG T5-REF"]] - signals[signal_names["EEG O1-REF"]]),  # 3
            (signals[signal_names["EEG FP2-REF"]] - signals[signal_names["EEG F8-REF"]]),  # 4
            (signals[signal_names["EEG F8-REF"]] - signals[signal_names["EEG T4-REF"]]),  # 5
            (signals[signal_names["EEG T4-REF"]] - signals[signal_names["EEG T6-REF"]]),  # 6
            (signals[signal_names["EEG T6-REF"]] - signals[signal_names["EEG O2-REF"]]),  # 7
            (signals[signal_names["EEG FP1-REF"]] - signals[signal_names["EEG F3-REF"]]),  # 14
            (signals[signal_names["EEG F3-REF"]] - signals[signal_names["EEG C3-REF"]]),  # 15
            (signals[signal_names["EEG C3-REF"]] - signals[signal_names["EEG P3-REF"]]),  # 16
            (signals[signal_names["EEG P3-REF"]] - signals[signal_names["EEG O1-REF"]]),  # 17
            (signals[signal_names["EEG FP2-REF"]] - signals[signal_names["EEG F4-REF"]]),  # 18
            (signals[signal_names["EEG F4-REF"]] - signals[signal_names["EEG C4-REF"]]),  # 19
            (signals[signal_names["EEG C4-REF"]] - signals[signal_names["EEG P4-REF"]]),  # 20
            (signals[signal_names["EEG P4-REF"]] - signals[signal_names["EEG O2-REF"]]),
        )
    )  # 21
    return new_signals


def readEDF(fileName):
    Rawdata = mne.io.read_raw_edf(fileName, preload=True)
    Rawdata.resample(200)
    Rawdata.filter(l_freq=0.3, h_freq=75)
    Rawdata.notch_filter((60))

    _, times = Rawdata[:]
    signals = Rawdata.get_data(units='uV')
    RecFile = fileName[0:-3] + "rec"
    eventData = np.genfromtxt(RecFile, delimiter=",")
    Rawdata.close()
    return [signals, times, eventData, Rawdata]


def save_pickle(object, filename):
    with open(filename, "wb") as f:
        pickle.dump(object, f)


def process_single_edf(file_path, out_dir):
    """Worker function to process a single EDF file and save its events."""
    fname = os.path.basename(file_path)
    try:
        [signals, times, event, Rawdata] = readEDF(file_path)
        signals = convert_signals(signals, Rawdata)
    except (ValueError, KeyError):
        return f"Skipped {fname}: formatting error"
    
    signals, offending_channels, labels = BuildEvents(signals, times, event)

    for idx, (signal, offending_channel, label) in enumerate(
        zip(signals, offending_channels, labels)
    ):
        sample = {
            "signal": signal,
            "offending_channel": offending_channel,
            "label": label,
        }
        save_pickle(
            sample,
            os.path.join(out_dir, fname.split(".")[0] + "-" + str(idx) + ".pkl"),
        )
    return True


def gather_edf_files(base_dir):
    """Helper to collect all .edf file paths recursively."""
    edf_files = []
    for dirName, subdirList, fileList in os.walk(base_dir):
        for fname in fileList:
            if fname.endswith(".edf"):
                edf_files.append(os.path.join(dirName, fname))
    return edf_files


def copy_worker(args):
    """Worker function to natively copy files."""
    src, dst = args
    shutil.copy(src, dst)


if __name__ == '__main__':
    """
    TUEV dataset is downloaded from https://isip.piconepress.com/projects/tuh_eeg/html/downloads.shtml
    """

    root = os.path.expanduser("~/simpletmp/data/TUEV/v2.0.1/edf")
    target = os.path.expanduser('~/simpletmp/processed_data/TUEV')

    train_out_dir = os.path.join(target, "processed_train")
    eval_out_dir = os.path.join(target, "processed_eval")

    os.makedirs(train_out_dir, exist_ok=True)
    os.makedirs(eval_out_dir, exist_ok=True)

    # --- PROCESS TRAIN FILES ---
    BaseDirTrain = os.path.join(root, "train")
    train_edf_files = gather_edf_files(BaseDirTrain)
    
    print(f"Processing {len(train_edf_files)} train files using 16 cores...")
    with concurrent.futures.ProcessPoolExecutor(max_workers=16) as executor:
        func_train = partial(process_single_edf, out_dir=train_out_dir)
        list(tqdm(executor.map(func_train, train_edf_files), total=len(train_edf_files)))

    # --- PROCESS EVAL FILES ---
    BaseDirEval = os.path.join(root, "eval")
    eval_edf_files = gather_edf_files(BaseDirEval)
    
    print(f"Processing {len(eval_edf_files)} eval files using 16 cores...")
    with concurrent.futures.ProcessPoolExecutor(max_workers=16) as executor:
        func_eval = partial(process_single_edf, out_dir=eval_out_dir)
        list(tqdm(executor.map(func_eval, eval_edf_files), total=len(eval_edf_files)))


    # --- TRANSFER TO TRAIN, EVAL, AND TEST ---
    print("Splitting into train, eval, and test directories...")
    processed_root = os.path.expanduser('~/simpletmp/processed_data/TUEV')

    train_files = os.listdir(os.path.join(processed_root, "processed_train"))
    train_val_sub = list(set([f.split("_")[0] for f in train_files]))
    
    test_files = os.listdir(os.path.join(processed_root, "processed_eval"))

    train_val_sub.sort(key=lambda x: x)
    train_sub = train_val_sub[: int(len(train_val_sub) * 0.8)]
    val_sub = train_val_sub[int(len(train_val_sub) * 0.8) :]

    val_files_split = [f for f in train_files if f.split("_")[0] in val_sub]
    train_files_split = [f for f in train_files if f.split("_")[0] in train_sub]

    final_train_dir = os.path.join(processed_root, 'processed', 'processed_train')
    final_eval_dir = os.path.join(processed_root, 'processed', 'processed_eval')
    final_test_dir = os.path.join(processed_root, 'processed', 'processed_test')

    os.makedirs(final_train_dir, exist_ok=True)
    os.makedirs(final_eval_dir, exist_ok=True)
    os.makedirs(final_test_dir, exist_ok=True)

    # Build copy task lists
    copy_tasks = []
    for file in train_files_split:
        copy_tasks.append((os.path.join(processed_root, 'processed_train', file), os.path.join(final_train_dir, file)))
    for file in val_files_split:
        copy_tasks.append((os.path.join(processed_root, 'processed_train', file), os.path.join(final_eval_dir, file)))
    for file in test_files:
        copy_tasks.append((os.path.join(processed_root, 'processed_eval', file), os.path.join(final_test_dir, file)))

    print(f"Copying {len(copy_tasks)} files using 16 cores...")
    with concurrent.futures.ProcessPoolExecutor(max_workers=16) as executor:
        list(tqdm(executor.map(copy_worker, copy_tasks), total=len(copy_tasks)))

    print('Done!')
