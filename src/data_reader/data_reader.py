import wfdb
import pandas as pd
import os
import numpy as np


class PTB_XL_Reader():
    def __init__(self, path: str):
        self.database_path = path
        self.num_records = None

    def get_csv(self):
        return pd.read_csv(self.database_path + 'ptbxl_database.csv', index_col='ecg_id')

    def get_csv_row(self, row):
        num_skipped_rows = 1 + (row - 1)
        return pd.read_csv(self.database_path + 'ptbxl_database.csv', skiprows=num_skipped_rows, nrows=1, header=None)


    def get_record(self, row: int, **kwargs):
        ''' 
        kwargs Parameters: 
            - freq: can be low (fs = 100 Hz) or high (fs = 500 Hz)
                - options: ['low', 'high']
                - default is high
        '''
        Y = self.get_csv_row(row)
        HR_FILENAME_COLUMN = 27
        LR_FILE_NAME_COLUMN = 26
        freq = kwargs.get('freq', 'high')

        filename = ''
        if freq == 'high':
            filename = Y.loc[0, HR_FILENAME_COLUMN]
        elif freq == 'low':
            filename = Y.loc[0, LR_FILE_NAME_COLUMN]
        else:
            raise ValueError("freq can take value 'low' or 'high'")


        record = wfdb.rdrecord(self.database_path + filename)
        return record.p_signal, record.fs

    def get_num_records(self) -> int:
        if self.num_records is None:
            Y = pd.read_csv(self.database_path + 'ptbxl_database.csv', index_col='ecg_id')
            self.num_records = Y.shape[0]

        return self.num_records
    
    def get_all_raw_voltages(self, **kwargs):
        data, sampling_freq = self.get_record(1)

        num_recordings, num_leads = data.shape
        X = np.zeros((self.num_records, num_recordings, num_leads), dtype=np.float32)

        for i in range(1, self.num_records):
            data, sampling_freq = self.get_record(i)
            if data.shape[0] >= num_recordings:
                X[i] = data[:num_recordings, :]
            else:
                # Pad with zeros if the signal is unexpectedly short
                X[i, :data.shape[0], :] = data
        
        return X, sampling_freq

class MIT_BIH_Reader():
    def __init__(self, path: str):
        self.database_path = path

    def get_record(self, record_id: int):
        '''
            record_id should be an integer between 100 and 234 inclusive.
        '''
        record = wfdb.rdrecord(os.path.join(self.database_path, str(record_id)))
        return record.p_signal, record.fs


class CHAPMAN_Reader():
    def __init__(self, path: str):
        self.database_path = path

    def get_record(self, rel_path: str):
        '''
            rel_path should be of the form WFDBRecords/dd/ddd/JSddddd where each
            d represents a digit. See the Records file in the Chapman directory
            for specifics on valid file directories.
        '''
        if not rel_path.startswith('WFDBRecords'):
            print("CHAPMAN_Reader::get_record: ensure rel_path starts with WFDBRecords")
            exit(1)

        record = wfdb.rdrecord(os.path.join(self.database_path, rel_path))
        return record.p_signal, record.fs


if __name__ == "__main__":
    reader = PTB_XL_Reader()
    record = reader.get_record(4)       # shape is 5000, 12
                                        # 500 Hz * 10 s = 5000 recordings
                                        # 12 leads

