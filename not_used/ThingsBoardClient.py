import logging  # for logging errors
import pandas as pd  # for data manipulation and analysis
import numpy as np  # for numerical computing
import pandas.errors
import pickle  # for saving/loading Python objects
import csv
import datetime
from pathlib import Path
from matplotlib import pyplot as plt  # for data visualization
from tb_rest_client.rest_client_pe import *  # Thingsboard API client
from tb_rest_client.rest import ApiException  # exception class for Thingsboard API
from statsmodels.tsa.stattools import adfuller  # statistical functions for time series analysis

'''Class to fetch and prepare Data from Thingsboard-API
    :parameter api_url, username, password
    '''


class ThingsBoardClient:

    def __init__(self, api_url, username, password):
        self.api_url = api_url
        self.username = username
        self.password = password

    # Function to fetch telemetrydata from Thingsboard API
    def fetch_timeseries_to_pickle(self, device_id, key_list, current=False,
                                   end_time=(int(datetime.datetime.now().timestamp() * 1000))):

        # set end_time
        end_time = end_time
        # set up logging
        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s - %(levelname)s - %(module)s - %(lineno)d - %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S')
        # creating the REST client object with context manager to get auto token refresh
        with RestClientPE(base_url=self.api_url) as rest_client:
            try:
                # authenticate with credentials
                rest_client.login(username=self.username, password=self.password)
                # get keys from Device
                timeseries_keys = rest_client.telemetry_controller.get_timeseries_keys_using_get1('DEVICE', device_id)
                # Get the device to fetch the createdtime of the device to set starttime
                device = rest_client.device_controller.get_device_by_id_using_get(device_id)
                # created_time of device start date for timeseries
                if not current:
                    start_ts = device.created_time
                else:
                    start_ts = end_time - 2 * 24 * 60 * 60 * 1000
                # get timeseries
                thread = rest_client.telemetry_controller.get_timeseries_using_get_with_http_info('DEVICE',
                                                                                                  device_id,
                                                                                                  keys=','.join(
                                                                                                      key_list),
                                                                                                  start_ts=start_ts,
                                                                                                  end_ts=end_time,
                                                                                                  limit=2147483647,
                                                                                                  async_req=True)
                result = thread.get()
                result[0]['sensor_type'] = device.type
                result[0]['sensor_name'] = device.name
            except ApiException as e:
                logging.exception(e)
            rest_client.logout()
            # save data in pickle
            if not current:
                data = open(f'../../data/sensor_data/{device_id}_RAW', 'wb')
            else:
                data = open(f'../../data/sensor_data/{device_id}_current_RAW', 'wb')
            pickle.dump(result[0], data)
            data.close()

    # Function to push telemetry to ThingsBoard API
    def push_telemetry_to_device(self, device_id, telemetry_data):

        # set up logging
        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s - %(levelname)s - %(module)s - %(lineno)d - %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S')
        # creating the REST client object with context manager to get auto token refresh
        with RestClientPE(base_url=self.api_url) as rest_client:
            try:
                # authenticate with credentials
                rest_client.login(username=self.username, password=self.password)
                # push telemetry to Device
                thread = rest_client.telemetry_controller.save_entity_telemetry_using_post_with_http_info('DEVICE',
                                                                                                          device_id,
                                                                                                          body=telemetry_data,
                                                                                                          scope='ANY',
                                                                                                          async_req=True)

            except ApiException as e:
                logging.exception(e)
            rest_client.logout()
            return thread

