import numpy as np
import pickle
import pandas as pd
import csv
import datetime

from pathlib import Path
from matplotlib import pyplot as plt
from statsmodels.tsa.stattools import adfuller

from service.ThingsBoardClientLight import ThingsBoardClientLight
from settings.Authentication import Authentication
from settings.Settings import Settings

'''
Class for data manipulation and analysis tasks 
'''


class DataService:

    def __init__(self):
        self.auth = Authentication()
        self.settings = Settings()
        self.tc = ThingsBoardClientLight(self.auth.URL, self.auth.USERNAME, self.auth.PASSWORD)

    # creates a complete curve from the beginning of a data set, flags start and end
    @staticmethod
    def create_full_curve(df, vdd_key, test=False, current=False):
        if not current:
            full_curve = df.loc[:df[vdd_key].idxmin()].copy()
        else:
            full_curve = df.copy()
        # mark the start and end of the curve
        full_curve['start'] = 0
        full_curve['end'] = 0

        # if test is set, the data set does not get a label for the end
        if not test:
            full_curve.at[full_curve.index[-1], 'end'] = 1
        full_curve.at[full_curve.index[0], 'start'] = 1

        return full_curve

    # creates a complete curve from the beginning of a data set, flags start and end, fills up dataframe
    @staticmethod
    def create_full_curve_thesis(df, vdd_key, batch_size=1000, test=False):

        # create `full_curve`
        full_curve = df.loc[:df[vdd_key].idxmin()].copy()
        to_add = batch_size - len(full_curve)
        rest_of_batch_size = df.iloc[-to_add:].copy()
        rest_of_batch_size['data_BatV'] = df[vdd_key].loc[df[vdd_key].idxmin()]
        rest_of_batch_size['rm_data_BatV'] = df[vdd_key].loc[df[vdd_key].idxmin()]
        rest_of_batch_size['start'] = 0
        rest_of_batch_size['end'] = 1

        # fill the start of the curve with 0
        full_curve['start'] = 0
        full_curve['end'] = 0
        if not test:
            full_curve = pd.concat([full_curve, rest_of_batch_size], ignore_index=True)
        full_curve.at[full_curve.index[0], 'start'] = 1

        return full_curve

    # restructure data to time windows
    @staticmethod
    def restructure_data_for_lstm_production(X, window):
        X_ = []
        idx_range = range(len(X))
        for idx in idx_range:
            window_data = X[idx:idx + window]
            if len(window_data) == window:
                X_.append(window_data)
        X_ = np.array(X_)
        return X_

    # function to convert telemetry data saved as pickle to pandas dataframe
    def pickle_to_pandas_dataframe(self, data_name, key_list, missing_values='lag_impute', raw=False):
        list_dataframes = []
        file = open(f'../../data/sensor_data/{data_name}', 'rb')
        data = pickle.load(file)

        # extract dataframe to dataframe list, In this section the extracted timeseries data for each key in key_list is
        # converted into separate DataFrames, specific data cleansing is performed, and at the end all
        # DataFrames are merged into a single DataFrame pd_merge.
        for key in key_list:
            if key in data.keys():
                df = pd.DataFrame(data[key])
                df = df.rename(columns={'value': key})
                try:
                    df[key] = pd.to_numeric(df[key].str.replace('V', ''), downcast='float', errors='raise')
                    df[key] = df[key].astype('float64').round(2)
                    df.loc[df[key] > 1000, key] = df[key] / 1000
                except ValueError as e:
                    print(f'Key: {key} kann nicht to numeric geparst werden \n\n{e}\n\n{data_name}')
                df['ts'] = pd.to_datetime(df['ts'], unit='ms')
                list_dataframes.append(df)
            else:
                print(f'Key: {key}  IS NOT A VALID KEY')

        # merge dataframe list into one dataframe
        pd_merge = self.merge_dataframes(list_dataframes)

        if not raw:
            # resample
            pd_merge = pd_merge.resample(rule='H').mean().round(2)
            pd_merge = pd_merge.applymap(lambda x: x / 1000 if x > 1000 else x)

            # rolling-means_dataframe
            pd_merge_rolling = self.rolling_means_to_dataframe(pd_merge)

            # impute dataframe
            pd_merge_rolling_impute = self.impute_dataframe(pd_merge_rolling, missing_values)

            # create a column with the sensor name
            pd_merge_rolling_impute = pd_merge_rolling_impute.assign(sensor_id=data_name[:36])
            pd_merge_rolling_impute[data_name[:36]] = 1

            return pd_merge_rolling_impute
        else:
            return pd_merge

    # TODO A function that extracts individual curves from a Pandas DataFrame
    def fetch_current_cycle_thesis(self, device_id, key_list, missing_values='lag_impute'):
        data_name = f'{device_id}_Thesis_RAW'
        df = self.pickle_to_pandas_dataframe(data_name=data_name, key_list=key_list, missing_values=missing_values)
        # Find the index of the last value that differs at least by 1 from the previous value this point describes
        # the increase of the curve
        last_index = df.loc[(df[key_list[0]][::-1].diff() < 0) & (abs(df[key_list[0]][::-1].diff()) > 1)][key_list[0]][
                     ::-1].idxmax()
        df = df[last_index:]

        # Find the index of the last value that differs at least by 1 from the previous value this point describes
        # the fall of the curve
        last_index = df.loc[(df[key_list[0]][::-1].diff() > 0) & (abs(df[key_list[0]][::-1].diff()) > 1)][key_list[0]][
                     ::-1].idxmax()

        df = df[last_index:]
        df = df.drop(df.index[0:20])

        #  return of the data frame without the found outliers
        return df

    # Function to calculate rolling means for given DataFrame
    @staticmethod
    def rolling_means_to_dataframe(dataframe):
        df_rm = pd.DataFrame()
        for key in dataframe.keys():
            rm = f'rm_{key}'
            try:
                df_rm[rm] = dataframe[key].rolling(window=9, min_periods=1).mean()
            except pandas.errors.DataError as e:
                print(e)
        # merge rolling means_dataframe to merged_dataframe
        dataframe_rolling = pd.merge(dataframe, df_rm, on='ts', how='inner')
        return dataframe_rolling

    # gives an overview of a sensor and its requested measured values
    def preview(self, sensor_id, key_list, file_exists, fetch=False, missing_values='lag_impute'):

        vdd_key = None
        # if fetch is set the data will be loaded from the api
        if fetch:
            self.tc.fetch_timeseries_to_pickle(sensor_id, key_list)
        df = self.pickle_to_pandas_dataframe(f'{sensor_id}_RAW', key_list, missing_values)

        print(df.head())

        # get Sensor_type
        file = open(f'../../data/sensor_data/{sensor_id}_RAW', 'rb')
        data = pickle.load(file)
        sensor_type = data['sensor_type']
        sensor_name = data['sensor_name']
        print(data.keys())
        file.close()

        for e in df.keys():
            if e == 'data_vdd':
                vdd_key = 'data_vdd'
            elif e == 'data_BAT_V':
                vdd_key = 'data_BAT_V'
            elif e == 'data_BatV':
                vdd_key = 'data_BatV'
            elif e == 'data_battery':
                vdd_key = 'data_battery'
            elif e == 'data_batt':
                vdd_key = 'data_batt'
            elif e == 'status_battery_level':
                vdd_key = 'status_battery_level'
            elif e == 'data_analog_in_8':
                vdd_key = 'data_analog_in_8'

            if vdd_key is None:
                print('NO VDD KEY')
                return None

        # plots the sensor data and saves it in an image file
        self.draw_sensor(df.index, df[vdd_key], sensor_id, sensor_name)
        self.draw_sensor(df.index, df[vdd_key], sensor_id, sensor_name, scatter=False)

        # creates a CSV file containing a preview of the data for the specified sensor. If file_exists is True, the CSV
        # file will be updated. If file_exists False, a new CSV file is created and the headers are added.
        first_row = False
        sensor_preview_file = Path('../../data/sensor_preview/sensor_preview_file.csv')
        if not sensor_preview_file.is_file():
            first_row = True
        with open('../../data/sensor_preview/sensor_preview_file.csv', mode='a+') as sensor_preview_file:
            writer = csv.writer(sensor_preview_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            if first_row or file_exists:
                if file_exists:
                    writer.writerow([f'Updated: {str(datetime.datetime.now())}'])
                writer.writerow(['Sensor_id',
                                 'Sensor_Name',
                                 'Type',
                                 'Anzahl Datensätze',
                                 'vdd_min',
                                 'vdd_max',
                                 'Zuletzt gesendet am',
                                 'letzter Wert'])
            writer.writerow([sensor_id,
                             sensor_name,
                             sensor_type,
                             len(df.index),
                             df[vdd_key].min(),
                             df[vdd_key].max(),
                             df.index[-1],
                             df[vdd_key][-1]])
        sensor_preview_file.close()

    # plots the sensor data and saves it as an image file
    @staticmethod
    def draw_sensor(x_axis, y_axis, sensor_id, sensor_name, scatter=True, fontsize=14):
        if scatter:
            plt.scatter(x_axis, y_axis, alpha=.3)
            extension = 'scatter'
        else:
            plt.plot(x_axis, y_axis)
            extension = 'plot'
        plt.title(sensor_name)
        plt.xlabel('Zeit', fontsize=fontsize)
        plt.ylabel('Spannung in Volt', fontsize=fontsize)
        plt.xticks(fontsize=fontsize, rotation=45)
        plt.yticks(fontsize=fontsize)
        plt.legend(fontsize=fontsize)
        plt.tight_layout()
        plt.savefig(f'../../data/sensor_preview/png/{sensor_id}_{extension}', pad_inches=2)
        plt.show()

    # Function to merge dataframes by timestamp
    @staticmethod
    def merge_dataframes(list_dataframes):
        df = pd.DataFrame()
        pd_merge = pd.DataFrame()
        if len(list_dataframes) == 0:
            return df
        elif len(list_dataframes) == 1:
            df.set_index('ts', inplace=True)
            df = df[::-1]
            return df
        elif len(list_dataframes) >= 2:
            pd_merge = pd.merge(list_dataframes[0], list_dataframes[1], on='ts', how='inner')
            if len(list_dataframes) >= 3:
                for df_in_list in list_dataframes[2:]:
                    pd_merge = pd.merge(pd_merge, df_in_list, on='ts', how='inner')
        # set index and invert dataframe
        pd_merge.set_index('ts', inplace=True)
        pd_merge = pd_merge[::-1]
        return pd_merge

    # Function to impute missing values in dataframe
    @staticmethod
    def impute_dataframe(dataframe, missing_values):
        if missing_values == 'drop_na':
            dataframe = dataframe.dropna()
        elif missing_values == 'lag_impute':
            dataframe = dataframe.fillna(method='ffill')
        elif missing_values == 'rm_impute':
            for key in dataframe.keys():
                dataframe = dataframe.fillna(dataframe.rolling(9, min_periods=1).mean())
        else:
            print('Dataframe not imputed')
        return dataframe

    # read sensor data from a CSV file located at ../../data/Sensoren.csv and use the preview method of the same
    # class to preview each sensor's data.
    def preview_from_csv(self, fetch=True, missing_values='lag_impute'):
        # check if the file already exists
        file_exists = False
        sensor_preview_file = Path('../../data/sensor_preview_file.csv')
        if sensor_preview_file.is_file():
            file_exists = True
        with open('../settings/Sensoren.csv', mode='r', encoding='utf-8-sig') as sensoren_file:
            csv_reader = csv.reader(sensoren_file, delimiter=';')
            for row in csv_reader:
                sensor_id, data_vdd, rssi, snr = row
                key_list = [data_vdd, rssi, snr]
                print(key_list)
                self.preview(sensor_id, key_list, file_exists, fetch, missing_values)
                file_exists = False

    @staticmethod
    def ad_test(dataset):
        dftest = adfuller(dataset, autolag='AIC')
        print(f'1. ADF : {dftest[0]}')
        print(f'2. P-Value : {dftest[1]}')
        print(f'3. Num Of Lags : {dftest[2]}')
        print(f'4. Num Of Observations Used For ADF Regression and Critical Values Calculation : {dftest[3]}')
        print(f'5. Critical Values : ')
        for key, value in dftest[4].items():
            print(f'\t{key} : {value}')

            # adjusts the data for LSTM

    @staticmethod
    def restructure_data_for_lstm(X, y, window, horizon):
        X_, y_ = [], []
        idx_range = range(len(X) - (window + horizon))
        for idx in idx_range:
            X_.append(X[idx:idx + window])
            y_.append(y[idx + window + horizon])
        X_ = np.array(X_)
        y_ = np.array(y_)
        return X_, y_

    @staticmethod
    def restructure_data_for_lstm_constant(X, y, constant, window, horizon):
        X_, y_, const = [], [], []
        idx_range = range(len(X) - (window + horizon))
        for idx in idx_range:
            X_.append(X[idx:idx + window])
            y_.append(y[idx + window + horizon])
            const.append(constant[idx])
        X_ = np.array(X_)
        y_ = np.array(y_)
        const = np.array(const)
        return X_, y_, const

        # adjusts the data for LSTM

    @staticmethod
    def restructure_data_for_lstm_constants(X, y, constant, sensor, window, horizon):
        X_, y_, const, sensor_list = [], [], [], []
        idx_range = range(len(X) - (window + horizon))
        for idx in idx_range:
            X_.append(X[idx:idx + window])
            y_.append(y[idx + window + horizon])
            const.append(constant[idx])
            sensor_list.append(sensor[idx])
        X_ = np.array(X_)
        y_ = np.array(y_)
        const = np.array(const)
        sensor_list = np.array(sensor_list)
        return X_, y_, const, sensor_list

    @staticmethod
    def restructure_data_for_lstm_seq_to_seq(X, y, window, horizon, seq):
        X_, y_, = [], [],
        idx_range = range(len(X) - (window + horizon + seq))
        for idx in idx_range:
            X_.append(X[idx:idx + window])
            y_.append(y[idx + window + horizon: idx + window + horizon + seq])
        X_ = np.array(X_)
        y_ = np.array(y_)
        return X_, y_

    @staticmethod
    def multiple_prediction(window_slot, window, indicators, future_steps, model, input_data, scaler):
        x_train = input_data[window_slot]
        current_input = x_train
        y_train_future = np.zeros((future_steps,))
        for i in range(future_steps):
            current_input = np.reshape(current_input, (1, window, indicators))
            y_train_future[i] = model.predict(current_input)
            current_input = np.insert(current_input, window,
                                      (y_train_future[i], x_train[0][1], x_train[0][2], x_train[0][3], x_train[0][4]),
                                      axis=1)
            current_input = np.delete(current_input, 0, axis=1)
        return scaler.inverse_transform(np.reshape(y_train_future, (-1, 1)))

    @staticmethod
    def plot_multi_pred(y_raw, predicted, period, y_lim, window_slot, data_name, fontsize=10, offset=0):
        plt.plot(np.linspace(1, len(y_raw[offset:offset + period]), len(y_raw[offset:offset + period])),
                 y_raw[offset:offset + period],
                 label='Original Trainingskurve')
        plt.plot(np.arange(window_slot, window_slot + len(predicted)),
                 predicted, label='Prädiktion', color='green')
        plt.legend()
        plt.xlabel('Stunden', fontsize=fontsize)
        plt.ylabel('Spannung in Volt', fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.legend(fontsize=fontsize)
        plt.ylim(y_lim)
        plt.tight_layout()
        plt.savefig(f'../../data/png/evaluate_train_predict_future_{data_name}', pad_inches=2)
        plt.show()

    def draw_sensor_for_analysis(self, x_axis, y_axis, sensor_id, sensor_name, scatter=True, save_plot=False):
        if scatter:
            plt.scatter(x_axis, y_axis, alpha=.3)
            extension = 'scatter'
        else:
            plt.plot(x_axis, y_axis)
            extension = 'plot'
        plt.title(sensor_name)
        plt.xticks(rotation=45, fontsize=self.settings.fontsize)
        plt.yticks(fontsize=self.settings.fontsize)
        plt.ylim(self.settings.y_lim)
        plt.xlabel('Time')
        plt.ylabel('Spannung')
        plt.tight_layout()
        if save_plot:
            plt.savefig(f'../../data/png/{sensor_id}_{extension}_Thesis_extended', pad_inches=2)
        plt.show()
