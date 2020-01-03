import traceback
import os
import json
import pickle
from io import StringIO
from dataclasses import dataclass

import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm_notebook, tqdm

from emotions import Recognizer
from age_predictor import VideoPredictor, GenderPredictor, AgePredictor


@dataclass
class ResultsConfig:
    dataset_dir: str
    campaign_name: str
    video_length: dict
    exec_from_notebook: bool = True
    backup_results: bool = True
    dump_on_complete: bool = True
    target_fps: float = 30
    verbose: int = 2  # 2 - full, 1 - main, 0 - none


class ResultsProcessor:
    def __init__(self, config: ResultsConfig):
        self.config = config
        self.recognizer = None
        self.age_predictor = None
        self.gender_predictor = None
        self.results, self.results_bak = {}, {}
        self.metadata = {}
        self.labels = ['Neutral', 'Happiness', 'Sadness', 'Surprise', 'Fear', 'Disgust', 'Anger', 'Contempt']
        self.progress = tqdm_notebook if self.config.exec_from_notebook else tqdm

    def console(self, *args, level=1):
        if self.config.verbose >= level:
            print(*args)

    def process_results(self):
        if self.recognizer is None:
            self.console('Recognizer init.')
            self.recognizer = Recognizer()
        for dir in self.progress(os.listdir(self.config.dataset_dir)):
            if not os.path.isfile(self.config.dataset_dir + '/' + dir + '/raw_result.csv'):
                try:
                    res = self.recognizer.recognize_video(self.config.dataset_dir + '/{}/face_record_mjpeg.mp4'.format(dir))
                    with open(self.config.dataset_dir + '/' + dir + '/raw_result.csv', 'w') as f:
                        print(res, file=f)
                    self.results[dir] = pd.read_csv(StringIO(res))
                except Exception:
                    traceback.print_exc()
        self.console('Done!')
        if self.config.backup_results:
            self.results_bak = self.results.copy()
        if self.config.dump_on_complete:
            self.dump_results()
        self.console('Processing ended.')

    def load_processed(self):
        with open('processed_results/'+self.config.campaign_name + '/id_lst.json') as f:
            id_lst = json.load(f)
            self.results = self.load_results(id_lst)
            self.metadata = self.load_metadata(id_lst)
        self.console('Loaded successfully!')

    def load_metadata(self, id_lst):
        metadata = {}
        for video_id in self.progress(id_lst):
            try:
                with open(self.config.dataset_dir + '/' + video_id + '/metadata.json') as f:
                    metadata[video_id] = json.load(f)
            except:
                pass

        return metadata

    def load_results(self, id_lst):
        results = {}
        for video_id in self.progress(id_lst):
            try:
                results[video_id] = pd.read_csv(self.config.dataset_dir + '/' + video_id + '/raw_result.csv')
            except:
                pass
        return results

    def dump_results(self):
        dirname = 'processed_results/'+self.config.campaign_name
        if not os.path.isdir(dirname):
            os.mkdir(dirname)
        with open(dirname + '/id_lst.json', 'w') as f:
            json.dump(list(self.results.keys()), f)
        with open(dirname + '/results.pickle', 'wb') as f:
            pickle.dump(self.results, f)
        with open(dirname + '/metadata.pickle', 'wb') as f:
            pickle.dump(self.metadata, f)
        self.console('Dumped successfully')

    def interpolate_results(self, df, num_frames):
        t = [np.nan] * len(self.labels)
        tt = []
        for i in range(num_frames):
            tt.append(t)

        tt = pd.DataFrame(tt, columns=self.labels)
        for i in range(len(df)):
            tt.iloc[int(i * (num_frames / len(df)))] = df.iloc[i]
        failed = False
        try:
            column = df.columns[0]
            if np.isnan(tt[column].iloc[0]):
                i = 1
                while (i < len(tt)) and np.isnan(tt[column].iloc[i]):
                    i += 1
                if i == len(tt):
                    self.console('kek1', level=2)
                    raise Exception()
                tt.iloc[0] = tt.iloc[i]

            if np.isnan(tt[column].iloc[-1]):
                i = -2
                while (i > -len(tt)) and np.isnan(tt[column].iloc[i]):
                    i -= 1
                if i == -len(tt):
                    self.console('kek2', level=2)
                    raise Exception()
                tt.iloc[-1] = tt.iloc[i]

            for column in tt.columns:
                tt[column].interpolate(method='akima', inplace=True)
        except:
            traceback.print_exc()
            failed = True
        return tt, failed

    def fix_results(self):
        keys = list(self.results.keys())
        failed_counter = 0
        for key in self.progress(keys):
            self.results[key].columns = self.labels
            if key in self.metadata:
                if self.metadata[key]['index'] in self.config.video_length:
                    self.results[key], failed = self.interpolate_results(self.results[key], int(self.config.video_length[self.metadata[key]['index']] * self.config.target_fps))
                    if failed:
                        failed_counter += 1
                        del self.results[key]
                else:
                    self.console(self.metadata[key]['index'], 'not in video_length')
                    del self.results[key]
            else:
                del self.results[key]
                failed_counter += 1
        self.console(failed_counter, 'failed')

    def predict_gender(self, num_frames=1):
        if self.gender_predictor is None:
            self.gender_predictor = VideoPredictor(GenderPredictor(), reshape=True, face_size=96)
        gender_data = {}

        for dir in self.progress(os.listdir(self.config.dataset_dir)):
            try:
                gender_data[dir] = self.gender_predictor.predict(self.config.dataset_dir + f'/{dir}/face_record_mjpeg.mp4', num_frames=num_frames)
                self.console(dir, gender_data[dir])
            except Exception:
                traceback.print_exc()
        for key in gender_data.keys():
            gender_data[key] = gender_data[key][1]
        return gender_data

    def predict_age(self, num_frames=1):
        predictor = VideoPredictor(AgePredictor())
        age_data = {}

        for dir in tqdm_notebook(os.listdir(self.config.dataset_dir)):
            try:
                age_data[dir] = predictor.predict(self.config.dataset_dir + f'/{dir}/face_record_mjpeg.mp4', num_frames=num_frames)
            except Exception:
                traceback.print_exc()
        for key in age_data.keys():
            age_data[key] = age_data[key][1]
        return age_data


def separate_results(results, metadata, cat_info: dict):
    """
    :param results: {video_id (result_id): Object (pd.DataFrame, dict, etc)}
    :param cat_info: {video_identifier (not result id):  campaign_name}
    :return: {campaign_name: {video_id (result_id): Object} }
    """
    sep_res = {campaign: {} for campaign in set(cat_info.values())}
    sep_met = {campaign: {} for campaign in set(cat_info.values())}
    not_in_metadata_counter, not_in_cat_info_counter = 0, 0
    for result_id, result in results.items():
        if result_id not in metadata:
            print('Key', result_id, 'not in metadata!')
            not_in_metadata_counter += 1
        elif metadata[result_id]['index'] not in cat_info:
            print('Key', result_id, 'not in cat_info!')
            not_in_cat_info_counter += 1
        else:
            key = cat_info[metadata[result_id]['index']]
            sep_res[key][result_id] = result
            sep_met[key][result_id] = metadata[result_id]
    print('Done! "Not in metadata" fails:', not_in_metadata_counter, '"Not in cat_info fails:', not_in_cat_info_counter)
    return sep_res, sep_met
