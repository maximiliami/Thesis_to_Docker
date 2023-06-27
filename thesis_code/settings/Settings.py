class Settings:
    def __init__(self):
        # settings forecast
        self.window = 24
        self.horizon = 1
        self.indicators = 5
        self.future_steps = 24 * 7 * 4
        # SensorID's
        self.sensor_list = ['6aa082d0-a5f8-11ed-8fa7-8d95df41b8ce', '3f394130-a5f9-11ed-8fa7-8d95df41b8ce']
        self.key_list = ['data_BatV', 'data_rssi', 'data_snr']
        self.feature_list = ['data_BatV',
                             '6aa082d0-a5f8-11ed-8fa7-8d95df41b8ce',
                             '3f394130-a5f9-11ed-8fa7-8d95df41b8ce',
                             'start',
                             'end']
        self.missing_values = 'lag_impute'
        self.scaler_x = f'../../data/scaler/LGT92_LSTM_scaler_x_thesis.joblib'
        self.scaler_y = f'../../data/scaler/LGT92_LSTM_scaler_y_thesis.joblib'
        self.model = f'../../data/model/LGT92_LSTM_model_thesis.h5'
        self.trained_data = '../../data/sensor_data/dataframe_trained_data.pkl'

        # settings plot
        self.y_lim = 2.6, 4.1
        self.fontsize = 14
        self.length_curve = 1000
        self.end_curve_one = 1000
        self.end_curve_two = 2000
        self.end_curve_three = 3000
        self.end_curve_four = 4000
        self.fig_size_x = 16
        self.fig_size_y = 8
        self.sub_plot_axis = 111
        self.save_plot = True

        # settings critical Battery
        self.critical_voltage = 3.5
        self.empty_battery = 2.7
