import re
import gc
import math
import os
import time
import numpy as np
from multiprocessing import Pool
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib import dates
from configparser import ConfigParser
import datetime
from dateutil.relativedelta import relativedelta

# ######### date range section ########################

""" manually enter dates """
# end_date = '21-11-2021'  # date_pattern='dd-mm-yyyy'
# start_date = '19-02-2022'  # date_pattern='dd-mm-yyyy'

""" date range with end date set as today """
today = datetime.date.today()
end_date = today.strftime("%d-%m-%Y")

# for monthly intervals
'''change the number of months in relativedelta function 
Keep it as integer
Note: always use keyword "months"; do not use keyword "month"'''
start_date = (datetime.datetime.strptime(end_date, '%d-%m-%Y') - relativedelta(months=3)).strftime("%d-%m-%Y")

# for day interval
"""change the number of days in function timedelta"""
# nr_of_days = datetime.timedelta(days=25)
# start_date = (today - nr_of_days).strftime("%d-%m-%Y")


def read_mapping_csv(filepath):
    """
    :param filepath: path of the file which contains all the data regarding limits of respective taglogid
    :return: returns dataframe grouped by tag log id so that it can be easily mapped
    """
    map_df = pd.read_csv(filepath)
    grouped_map_df = map_df.groupby('TagLogId')
    return grouped_map_df


# ############################## functions for Creating Graph ###############################

def reading_one_folder(directory):
    """
    :param directory: the folder path which contains all the taglogid files
    :return: list all paths of the all files in the folder
    """
    try:
        list_csvs = [os.path.join(directory, x) for x in os.listdir(directory) if x.endswith('.csv')]
        return list_csvs
    except FileNotFoundError:
        print("showerror", "No Temp Files found. Maybe no data avaiable in time-period.")


def plot_graph(dataframe, photo_name):
    """
    :param dataframe: of the taglog id
    :param photo_name: name of the file which will be saved
    :return: nothing. just saves the graph
    """
    global start_datum, end_datum, company_name, final_path

    result_folder_path = final_path + "\\Graph_App_Results\\Graphs\\" + str(company_name) + "\\"

    if not os.path.exists(result_folder_path):  # checking if folder already exists, if not : it creates one.
        os.makedirs(result_folder_path, exist_ok=True)

    title = str(photo_name[0]) + '--' + str(photo_name[1]) + \
            '--' + str(photo_name[2]) + '--' + str(photo_name[3]) + '--' + \
            'AL' + '--' + str(photo_name[4]) + '--' + 'WL' + \
            '--' + str(photo_name[5]) + '--' + str(photo_name[6].split('.')[0])

    location_of_figure = result_folder_path + title + '.png'

    fig = plt.figure(figsize=(24, 8), clear=True, num=5)  # creating the blank plot in background, in order to avoid
    # fig object created again and again and take more RAM

    # colors of plot dots ##does not matter in line plot, only useful in scatterplot, so can be disregarded
    palette = {
        'ok': 'green',
    }

    # https://seaborn.pydata.org/generated/seaborn.lineplot.html
    plot = sns.lineplot(x=0, y=1, data=dataframe, style=2, palette=palette, ci=None, marker="o", legend=False)

    # plot.set(xlabel='Time', ylabel='Value')
    plot.set_xlabel("Time", fontsize=25)
    plot.set_ylabel("Value", fontsize=25)
    plot.set_title(title, fontsize=24)
    plot.tick_params(labelsize=20)
    plot.xaxis.set_major_formatter(dates.DateFormatter("%d-%b"))

    """ ####### this section below is to create the limits lines across the graphs in case they cross them ####### """
    handles, _ = plot.get_legend_handles_labels()  # calling all th defaults labels in the background
    max_limit = dataframe[1].max()  # selecting the max value for the partcilaur taglogid dataframe
    labels = []

    if max_limit > float(photo_name[4]):
        plt.axhline(float(photo_name[4]), color='crimson', ls='--', label='alarm_limit=' + str(photo_name[4]))
        plt.axhline(float(photo_name[5]), color='limegreen', ls='--', label='warn_limit=' + str(photo_name[5]))
        labels.append('alarm_limit=' + str(photo_name[4]))
        labels.append('warn_limit=' + str(photo_name[5]))
        plt.legend(title='Warning Signs', handles=handles[2:],
                   labels=labels)  # here adding our own labels for limits lines

    elif float(photo_name[4]) > max_limit > float(photo_name[5]):
        plt.axhline(float(photo_name[5]), color='limegreen', ls='--', label='warn_limit=' + str(photo_name[5]))
        labels.append('warn_limit=' + str(photo_name[5]))
        plt.legend(title='Warning Signs', handles=handles[2:],
                   labels=labels)  # here adding our own labels for limits lines

    else:
        labels.append('warn_limit=' + str(photo_name[5]))
        plt.legend(title='Warning Signs',
                   labels=labels)  # this means all the values in the graphs are below the warning limit and no need to show the line

    plt.xticks(rotation=0)
    format_start_date = datetime.datetime.strptime(start_datum, '%Y%m%d') - datetime.timedelta(days=1)
    format_end_date = datetime.datetime.strptime(end_datum, '%Y%m%d') + datetime.timedelta(days=2)
    plt.xlim([format_start_date, format_end_date])
    # plt.xlim([format_start_date, format_end_date])
    # plt.gca().set_xbound(format_start_date, format_end_date)
    # plt.autoscale()
    plt.tight_layout()

    fig.savefig(location_of_figure)

    # cleaning all the data from the graphs
    fig.clf()
    plt.cla()
    plt.clf()
    gc.collect()


def my_conv(x):
    """
    part of pd.read_csv converter method
    :param x: indivial values when using read_csv
    :return: makes error/unformatted values as NaN
    """

    try:
        return float(x)
    except ValueError:
        return math.nan


def pandas_date_converter(input_date):
    try:
        formatted_date = datetime.datetime.strptime(input_date, '%Y-%m-%d %H:%M:%S').strftime('%Y-%m-%d %H:%M:%S')
        return pd.to_datetime(input_date, errors='coerce')

    except:
        return np.datetime64('NaT')


def load_sensor_data_temp_files(filepath):
    """The function
        1. accepts the path as the argument,
        2. reads the data from the CSV file from particular IP address,
        3. returns the DataFrame.
    """
    date_column = [0]

    data_frame = pd.read_csv(filepath,
                             header=None,
                             sep=',',
                             parse_dates=date_column,
                             low_memory=False,
                             on_bad_lines='skip',
                             converters={1: my_conv,  # to evaluate the improper formatted values
                                         0: pandas_date_converter})  # checking the format of dates before plotting graphs

    current_csv_name = filepath.split(os.path.sep)[-1].split('--')
    print('current csv file being plotted: ', current_csv_name)

    plot_graph(data_frame, current_csv_name)

    # cleaning all the garbage data in the background
    del data_frame
    gc.collect()


def normal_pooling_graph(csv_list):
    """
        function is to activate multiple processes using multiprocessing library to create the graphs

        :param csv_list: list of paths of all  temp csvs files
        :return: None
    """
    start_datum = formatted_start_date
    end_datum = formatted_end_date
    company_name = client_name
    final_path = final_folder_path

    pool = Pool(5, initializer=init_pool_graph, initargs=(start_datum, end_datum, company_name, final_path,))

    pool.map(load_sensor_data_temp_files, csv_list)


# ############################## functions for TagLogId CSV temp files ###############################

def corresponding_limits(name, df_map, tag_df):
    """

    :param tag_df: dataframe currently associated with taglogid
    :param name: taglogid
    :param df_map: mapped dataframe from formatted.csv
    :return: returns the limits values of the particular taglogid
    """
    limit_list = []

    try:
        row = df_map.get_group(int(name))
    except KeyError:
        print('****************** TagLog Dataframe involved in the error ************* \n', tag_df)
        raise

    characteristic = row.iloc[0]['Characteristic']

    if characteristic == 'dkw':
        limit_list.append(row.iloc[0]['DKW alarm limit'])
        limit_list.append(row.iloc[0]['DKW warning limit'])

    elif characteristic == 'aRms':
        limit_list.append(row.iloc[0]['aRMS alarm limit'])
        limit_list.append(row.iloc[0]['aRMS warning limit'])

    elif characteristic == 'vRms':
        limit_list.append(row.iloc[0]['vRMS alarm limit'])
        limit_list.append(row.iloc[0]['vRMS warning limit'])

    else:
        limit_list.append(float('nan'))
        limit_list.append(float('nan'))

    limit_list.append(row.iloc[0]['DB variable'])
    limit_list.append(row.iloc[0]['IP address'])
    limit_list.append(characteristic)
    limit_list.append(row.iloc[0]['Channel NR'])

    return limit_list


def hue_conditions(data, limits):
    """
    sets each value as ok so that later line plot can happen in same colors and same line
    """
    return 'ok'


def is_valid_date(year, month, day):
    day_count_for_month = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0):
        day_count_for_month[2] = 29
    return 1 <= month <= 12 and 1 <= day <= day_count_for_month[month]


def reading_big_folder(directory, start_datum, end_datum):
    """
    1. This function goes through the parent folder which contains all the kennwerte folders of all separated by dates.
    2. It goes through all the child folders and cross-check their names validity based on the writing format (eg 20211119)
    3. It checks the condition that folders which are selected for data frame and visualisation are within the selected
    date range.

    :param start_datum: start date within which the data is to be processed
    :param directory: the path of the parent folder
    :param end_datum: end date within which the data is to be processed
    :return: list of all the paths of all CSVs satisfying the date condition
    """

    list_csv = []
    matching_format = re.compile("^\d{8}$")  # providing default format of date in name of the folder (i.e. 20210101)

    for path, subdirs, files in os.walk(directory):  # loop to check all subfolders and files in a parent folder

        date_of_current_running_folder = (path.split(os.path.sep)[-1])  # splits the path string into list, so that
        # we can pick the last item which has to be date of the folder containing csv files

        if matching_format.match(date_of_current_running_folder):  # checking if the folder name is 8 digit number

            stringmonth = int(date_of_current_running_folder[4:6])
            stringdate = int(date_of_current_running_folder[6:])
            stringyear = int(date_of_current_running_folder[:4])

            if is_valid_date(stringyear, stringmonth, stringdate):

                current_formatted_folder_date = datetime.datetime.strptime(date_of_current_running_folder, '%Y%m%d').strftime(
                    '%Y%m%d')  # changing the format of the folder name date to check with date range

                if start_datum <= current_formatted_folder_date <= end_datum:

                    for name in files:
                        list_csv.append(os.path.join(path, name))

    return list_csv


def init_pool_graph(s_date, e_date, c_name, path):
    """
    :param c_name: name of the client selected in GUI
    :return: returns the name itself so that it becomes the global variable

    """

    global start_datum, end_datum, company_name, final_path
    start_datum = s_date
    end_datum = e_date
    company_name = c_name
    final_path = path


def init_pool(name, path):
    """
    :param name: name of the client selected in GUI
    :return: returns the name itself so that it becomes the global variable
    """

    global company_name, final_path
    company_name = name
    final_path = path


def normal_pooling(csv_list):
    """
    function is to activate multiple processes using multiprocessing library to create the temp files

    :param csv_list: list of paths of all csvs files
    :return: None
    """
    company_name = client_name
    final_folder = final_folder_path

    pool = Pool(4, initializer=init_pool,
                initargs=(company_name, final_folder))  # mentioning how many processors should be used
    # initargs is used so that company name variable remains a global variable to use in later functions

    pool.map(load_sensor_datafile, csv_list)  # map function sends values of csv_list into load_sensor_datafile()
    # one by one through multiple processes


def load_sensor_datafile(filepath):
    """The function
        1. accepts the path as the argument,
        2. reads the data from the CSV file from particular IP address,
        3. returns the DataFrame.
        """

    date_of_current_running_folder = (filepath.split(os.path.sep)[-2])
    current_formatted_folder_date = datetime.datetime.strptime(date_of_current_running_folder, '%Y%m%d').strftime(
        '%Y-%m-%d')

    print('Raw Data Folder Name', filepath)
    try:
        data_frame = pd.read_csv(filepath)
        filter_data_frame = filter_dates(data_frame, current_formatted_folder_date)
        grouped_data = grouping_data(filter_data_frame)  # call func to sort by dates and group_by data based on taglog
        write_final_data(grouped_data)  # calling func to write data on new csv files
    except pd.errors.EmptyDataError:  # exception to deal with files with empty raw data
        print('########### Empty Data error, File ignored ############ \n filepath: ', filepath)
        pass


def filter_dates(pd_data, current_date):
    """
    :param pd_data: dataframe right-after reading the data from csv
    :param current_date: date mentioned on the folder
    :return: the dataframe after filtering the values/rows with false dates
    """
    # removes rows of previous days
    pd_data = pd_data[pd.to_datetime(pd_data.ValueTime).dt.date.astype(str) == current_date].copy()
    return pd_data


def grouping_data(df):
    """
    :param df: Pandas Class DataFrame
    :return: data segregated on basis of the Tag ID
    """
    column_name = df.columns[0].lower()

    if column_name == 'id':
        df.rename(columns={'Id': 'TagLogId'}, inplace=True)

    grouped = df.sort_values('ValueTime').groupby('TagLogId')  # sorting the data and grouping them
    return grouped


def num_Value_filter(df):
    """
    :param df: dataframe of single tag log id from single csv raw file
    :return: df which removes all rows containing the NumValue equal to 0.
    """
    df2 = df.copy()
    df2['NumValue'] = df['NumValue'].replace({'0': np.nan, 0: np.nan})
    return df2


def write_final_data(data_by_tag):
    """
    Saves the data into separate CSV files based on the TagLogID

    :param data_by_tag: the dataframe which is already grouped by TagId
    :return: nothing
    """
    global company_name, final_path

    result_folder_path = final_path + "\\Graph_App_Results\\temp\\" + str(company_name)

    if not os.path.exists(result_folder_path):
        os.makedirs(result_folder_path, exist_ok=True)

    for tag_id in data_by_tag.groups.keys():  # .groups.keys() pulls all the unique TagIDs in a list
        tag_dataframe = data_by_tag.get_group(tag_id)  # fetching the particular group according to their TagID
        limits = corresponding_limits(tag_id, mapping_file_grouped[company_name], tag_dataframe)

        if not limits[4] == 'speed':
            data_in_csv = num_Value_filter(tag_dataframe)

        else:
            data_in_csv = tag_dataframe

        if not limits[4] == 'dkw':

            data_in_csv = data_in_csv.drop(['TagLogId'], axis=1)
            data_in_csv["alerts"] = data_in_csv.apply(hue_conditions, axis=1, limits=limits)

            temp_file_name = os.path.join(result_folder_path, company_name + '--' +
                                          str(limits[3]) + '&' + str(limits[5]) + '--' + str(tag_id) + '--' +
                                          str(limits[4]) + '--' + str(limits[0]) + '--' + str(limits[1])
                                          + '--' + str(limits[2]) + '.csv')

            if not os.path.exists(temp_file_name):
                data_in_csv.to_csv(temp_file_name, header=True, mode='a', sep=',', index=False)
            else:
                data_in_csv.to_csv(temp_file_name, header=False, mode='a', sep=',', index=False)


def starting_process():
    """
    function which is triggered when 'create temp files' button is pushed.
    """
    start_time = time.time()
    # directory = os.path.expanduser("~/Desktop") + "\\Graph_App_Results\\temp\\"  # here the temp files will be stored

    directory = final_folder_path + "\\Graph_App_Results\\temp\\" + client_name

    # deletes the previous files in the folder
    if os.path.exists(directory):
        for f in os.listdir(directory):
            os.remove(os.path.join(directory, f))

    list_of_all_csvs = reading_big_folder(data_folder_path, formatted_start_date,
                                          formatted_end_date)  # filtering csvs on basis of date range

    normal_pooling(list_of_all_csvs)  # to activate multiprocessing

    print("*** Temp Files Created *** \n Time Taken--- %s seconds ---" % round(time.time() - start_time, 2))
    print("Progress : ", "All Temp files are successfully created")


def start_plotting_graph():
    """
        function which is triggered when 'start plotting' button is pushed.
    """
    start_time = time.time()
    result_folder_path = final_folder_path

    directory = result_folder_path + "\\Graph_App_Results\\Graphs\\" + client_name  # here the graphs are stored

    # deletes the previous files in the folder
    if os.path.exists(directory):
        for f in os.listdir(directory):
            os.remove(os.path.join(directory, f))

    folder_with_final_csvs = result_folder_path + "\\Graph_App_Results\\temp\\" + client_name
    list_of_all_csvs = reading_one_folder(folder_with_final_csvs)
    normal_pooling_graph(list_of_all_csvs)

    print("*** All Graphs Files Created *** \n Time Taken--- %s seconds ---" % round(time.time() - start_time, 2))
    print("Progress : ", "All Graphs are successfully plotted")


# #################################### Main ###############################################

map_files_directory_path = [
                            # 'XX_formatted.csv',
                            # 'XXX_formatted.csv',
                            # 'XXXX_formatted.csv',
                            'XXXXX_formatted.csv',
                            # 'XXXXXX_formatted.csv'
                            ]  # path of Mapped files containing details about each TagLogID incl. limits

mapping_file_grouped = {}
for file in map_files_directory_path:
    key_name = str(file.split('_')[0])
    mapping_file_grouped[key_name] = read_mapping_csv(file)

config = ConfigParser()
config.read('config.ini')
data_folder_path = config['paths']['kennwerte_path']
final_folder_path = config['paths']['final_destination']

client_name = 'XXXXX'
formatted_start_date = datetime.datetime.strptime(start_date, '%d-%m-%Y').strftime('%Y%m%d')
formatted_end_date = datetime.datetime.strptime(end_date, '%d-%m-%Y').strftime('%Y%m%d')
print('formatted_end_date : ', formatted_end_date)
print('formatted_start_date : ', formatted_start_date)

if __name__ == '__main__':
    temp_num = 0
    csv_name = ''
    for path, subdirs, files in os.walk(data_folder_path):
        temp_num += 1
        for filename in files:
            csv_name = filename
            break
        if temp_num == 2:
            break

    # getting the main ip address from the file in the kennwerte folder
    ip_add = '.'.join(csv_name.split(' ')[0].split('.')[0:3])

    # default list of IP address of the companies
    ip_addresses_dict = {'XXX': 'XX.XX.XX', 'XXXX': 'XX.XX.XX',
                         'XX': 'XX.XX.XX', 'XXXXX': 'XX.XX.XX', 'XXXXXX': 'XX.XX.XX'}

    if ip_addresses_dict[client_name] != ip_add:
        raise ValueError('Kennwerte Folder selected does not belong to selected Client' + client_name)

    if formatted_end_date < formatted_start_date:
        raise ValueError("End Date cannot be before Start Date")

    starting_process()
    start_plotting_graph()
