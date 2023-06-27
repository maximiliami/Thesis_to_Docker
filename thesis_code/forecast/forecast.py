import datetime
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from joblib import load

from settings.Authentication import Authentication
from service.ThingsBoardClientLight import ThingsBoardClientLight
from service.DataService import DataService
from settings.Settings import Settings


# Forecasts the in settings.Settings given Sensors
class Forecast:

    # initializes various settings and objects required for the forecast
    def __init__(self):
        # initialize ThingsBoardClient
        self.settings = Settings()
        self.auth = Authentication()
        self.tc = ThingsBoardClientLight(username=self.auth.USERNAME,
                                         api_url=self.auth.URL,
                                         password=self.auth.PASSWORD)
        self.ds = DataService()
        self.scaler_x = load(self.settings.scaler_x)
        self.scaler_y = load(self.settings.scaler_y)
        self.model = load_model(self.settings.model)  # loads the trained model which is used for the prediction
        self.df_trained_data = pd.read_pickle(self.settings.trained_data)

    # retrieves sensor data using the ThingsBoard client and stores it in a list of data frames
    def fetch_sensor_data(self):
        df_list = []
        for sensor in self.settings.sensor_list:
            self.tc.fetch_timeseries_to_pickle(sensor, self.settings.key_list, current=True)
            df = self.ds.pickle_to_pandas_dataframe(f'{sensor}_current_RAW',
                                                    self.settings.key_list,
                                                    self.settings.missing_values)
            df = self.ds.create_full_curve(df, self.settings.key_list[0], test=True, current=True)

            for sensor_id in self.settings.sensor_list:
                if sensor_id not in df.columns:
                    df[sensor_id] = 0

            df_list.append(df)

        return df_list

    # method prepares the data sets for input to the LSTM model. It applies scaling to the raw data, restructures it
    # according to the model's input requirements, and stores the prepared data sets in prepared_data_set_list
    def prepare_data_sets_for_lstm(self, df_list):
        prepared_data_set_list = []

        for df in df_list:
            X_prepared_raw = df[self.settings.feature_list].values
            X_prepared_std = self.scaler_x.transform(X_prepared_raw)
            X_prepared = self.ds.restructure_data_for_lstm_production(X_prepared_std, self.settings.window)
            prepared_data_set_list.append([X_prepared])

        return prepared_data_set_list

    # takes the prepared data sets, iterates over them, and generates predictions using the trained LSTM model.
    # The predictions are stored in predicted_list
    def generate_predictions(self, prepared_data_set_list):
        predicted_list = []

        for data in prepared_data_set_list:
            current_input = data[0][-1]
            y_predicted = np.zeros((self.settings.future_steps,))

            for i in range(self.settings.future_steps):
                current_input = np.reshape(current_input, (1, self.settings.window, self.settings.indicators))
                y_predicted[i] = self.model.predict(current_input)
                current_input = np.insert(current_input, self.settings.window, (
                    y_predicted[i], data[0][-1][-1][1], data[0][-1][-1][2], data[0][-1][-1][3],
                    data[0][-1][-1][4]), axis=1)
                current_input = np.delete(current_input, 0, axis=1)
            to_descale = np.reshape(y_predicted, (-1, 1))
            predicted_list.append(self.scaler_y.inverse_transform(to_descale))

        return predicted_list

    # method converts the predicted data from the list format to pandas data frames list predicted_df_list
    @staticmethod
    def predicted_to_df(predicted_list):

        predicted_df_list = []

        for e in predicted_list:
            my_date_range = pd.date_range(datetime.datetime.now(), periods=len(e), freq='H')
            df = pd.DataFrame(e, columns=['predicted_data_BatV'])
            df['date'] = my_date_range
            df.set_index('date', inplace=True)
            predicted_df_list.append(df)
        return predicted_df_list

    # plots the training curves, current data sets, and predicted data.
    # It uses matplotlib to create a plot with multiple lines representing the various curves.
    def plot_predictions(self, predicted_list, df_list):
        fig, ax = plt.subplots(figsize=(self.settings.fig_size_x, self.settings.fig_size_y))

        for i in range(self.settings.end_curve_four // self.settings.end_curve_one):
            start_idx = i * self.settings.end_curve_one
            end_idx = (i + 1) * self.settings.end_curve_one
            sensor_id = self.df_trained_data.sensor_id[start_idx]
            ax.plot(np.linspace(1, self.settings.length_curve, self.settings.length_curve),
                    self.df_trained_data.iloc[start_idx:end_idx]['rm_data_BatV'],
                    label=f'{sensor_id} Trainingskurve')

        for i, df in enumerate(df_list):
            sensor_id = df.sensor_id[0]
            ax.plot(np.linspace(1, len(df.index), len(df.index)), df['rm_data_BatV'],
                    label=f'{sensor_id} letzte 48h')

        for i, predicted in enumerate(predicted_list):
            sensor = df_list[i]
            sensor_id = sensor.sensor_id[0]
            ax.plot(np.arange(len(df_list[0].index), len(predicted) + len(df_list[0].index)),
                    predicted,
                    label=f'Prädiktion {sensor_id}', linestyle='dotted')

        ax.set_xlabel('Stunden', fontsize=self.settings.fontsize)
        ax.set_ylabel('Spannung in Volt', fontsize=self.settings.fontsize)
        ax.tick_params(labelsize=self.settings.fontsize)
        ax.legend(fontsize=self.settings.fontsize, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)
        plt.tight_layout()

        if self.settings.save_plot:
            plt.savefig(f'../../data/png/production_Thesis_{datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")}',
                        pad_inches=2)
        plt.show()

    # plots the predicted data from each sensor.
    # If certain conditions (critical voltage or empty battery) are met, it prints a message indicating
    # the corresponding index and voltage level.
    def plot_prediction(self, predicted_df_list):

        for i, e in enumerate(predicted_df_list):
            if (e['predicted_data_BatV'] < self.settings.critical_voltage).any():
                index_critical_voltage = (e['predicted_data_BatV'] < int(self.settings.critical_voltage)).idxmax()
            else:
                index_critical_voltage = None
            if (e['predicted_data_BatV'] < self.settings.empty_battery).any():
                index_empty_battery = (e['predicted_data_BatV'] < int(self.settings.empty_battery)).idxmax()
            else:
                index_empty_battery = None

            if index_critical_voltage is not None:
                print(
                    f'{index_critical_voltage} {self.settings.critical_voltage}V reached for Sensor: '
                    f'{self.settings.sensor_list[i]}')
            if index_empty_battery is not None:
                print(f'{index_empty_battery} {self.settings.empty_battery}'
                      f'V reached for Sensor: {self.settings.sensor_list[i]}')
            e.plot()

    # ends the predicted data to the ThingsBoard server for each sensor.
    # It checks if the critical voltage or empty battery conditions are met and constructs a telemetry message
    # in JSON format to be pushed to the server.
    def send_data_to_things_board(self, predicted_df_list):

        for i, e in enumerate(predicted_df_list):
            if (e['predicted_data_BatV'] < self.settings.critical_voltage).any():
                index_critical_voltage = str(
                    (e['predicted_data_BatV'] < self.settings.critical_voltage).idxmax().date())
            else:
                index_critical_voltage = 'nicht erreicht'
            if (e['predicted_data_BatV'] < self.settings.empty_battery).any():
                index_empty_battery = str((e['predicted_data_BatV'] < self.settings.empty_battery).idxmax().date())
            else:
                index_empty_battery = 'nicht erreicht'

            scope = {
                'ts': datetime.datetime.now().timestamp() * 1000,
                'values': {
                    f'{self.settings.critical_voltage}V_reached ': index_critical_voltage,
                    f'{self.settings.empty_battery}V_reached ': index_empty_battery
                }
            }
            scope_json_dump = json.dumps(scope)
            scope_json_loads = json.loads(scope_json_dump)
            self.tc.push_telemetry_to_device(self.settings.sensor_list[i], scope_json_loads)

    # main entry point of the script. It executes the different steps of
    # the forecast process by calling the appropriate methods in the correct order.
    def run(self):
        df_list = self.fetch_sensor_data()
        prepared_data_set_list = self.prepare_data_sets_for_lstm(df_list)
        predicted_list = self.generate_predictions(prepared_data_set_list)
        print(predicted_list)
        predicted_df_list = self.predicted_to_df(predicted_list)
        print(predicted_df_list)
        self.plot_predictions(predicted_df_list, df_list)
        self.send_data_to_things_board(predicted_df_list)


# creates an instance of the Forecast class and calls its run method to start the forecast process.
if __name__ == '__main__':
    # TODO Authentication handler
    fore_cast = Forecast()
    fore_cast.run()
