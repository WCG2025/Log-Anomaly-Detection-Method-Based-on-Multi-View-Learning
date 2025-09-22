## 将原始日志文件处理为csv文件
import os.path
from datetime import datetime
import re
import numpy as np
import pandas as pd
data_dir = r'/home/jpy/graduation_design_final/BGL'
log_name = "BGL.log"

output_dir = data_dir

def log_to_dataframe(log_file, regex, headers, start_line, end_line):
    """ Function to transform log file to dataframe
    """
    log_messages = []
    linecount = 0
    cnt = 0

    if end_line is None:
        with open(log_file, 'r', encoding='latin-1') as fin:  # , encoding='latin-1'
            while True:
                line = fin.readline()
                if not line:
                    break
                # for line in fin.readlines():
                cnt += 1
                try:
                    match = regex.search(line.strip())
                    message = [match.group(header) for header in headers]
                    log_messages.append(message)
                    linecount += 1
                except Exception as e:
                    # print("\n", line)
                    # print(e)
                    pass

    else:
        line_pos = -1
        with open(log_file, 'r', encoding='latin-1') as fin:
            while True:
                line = fin.readline()
                line_pos += 1
                if line_pos < start_line:
                    continue
                if not line or line_pos >= end_line:
                    break
                cnt += 1
                try:
                    match = regex.search(line.strip())
                    message = [match.group(header) for header in headers]
                    log_messages.append(message)
                    linecount += 1
                except Exception as e:
                    # print("\n", line)
                    # print(e)
                    pass

    print("Total size is {}; Total size after encoding is {}".format(linecount, cnt))
    logdf = pd.DataFrame(log_messages, columns=headers)
    return logdf

def generate_logformat_regex(logformat):
    """ Function to generate regular expression to split log messages
    """
    headers = []
    splitters = re.split(r'(<[^<>]+>)', logformat)
    regex = ''
    for k in range(len(splitters)):
        if k % 2 == 0:
            splitter = re.sub(' +', '\\\s+', splitters[k])
            regex += splitter
        else:
            header = splitters[k].strip('<').strip('>')
            regex += '(?P<%s>.*?)' % header
            headers.append(header)
    regex = re.compile('^' + regex + '$')
    return headers, regex   

def structure_log(input_dir, output_dir, log_name, log_format,  start_line = 0, end_line = None):
    print('Structuring file: ' + os.path.join(input_dir, log_name))
    start_time = datetime.now()
    headers, regex = generate_logformat_regex(log_format)
    df_log = log_to_dataframe(os.path.join(input_dir, log_name), regex, headers, start_line, end_line)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    df_log.to_csv(os.path.join(output_dir, log_name + '_structured.csv'), index=False, escapechar='\\')

    print('Structuring done. [Time taken: {!s}]'.format(datetime.now() - start_time))

def fixedSize_window(raw_data, window_size, step_size):
    aggregated = [
        [raw_data['Content'].iloc[i:i + window_size].values,
        max(raw_data['Label'].iloc[i:i + window_size]),
         raw_data['Label'].iloc[i:i + window_size].values.tolist()
         ]
        for i in range(0, len(raw_data), step_size)
    ]
    return pd.DataFrame(aggregated, columns=list(raw_data.columns)+['item_Label'])

if 'thunderbird' in log_name.lower():
    log_format = '<Label> <Timestamp> <Date> <User> <Month> <Day> <Time> <Location> <Content>'   #thunderbird  , spirit, liberty
elif 'bgl' in log_name.lower():
    log_format = '<Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Level> <Content>'  #bgl

start_line=0
end_line=None

## Thunderbird
# start_line=16000000 #待定
# end_line=17000000 #待定


structure_log(data_dir,output_dir,log_name,log_format,start_line=start_line,end_line=end_line)
'''
train_ratio=0.8
df=pd.read_csv(os.path.join(output_dir,f'{log_name}_structured.csv'))
print(len(df))

df["Label"] = df["Label"].apply(lambda x: int(x != "-"))
train_len = int(train_ratio*len(df))

df_train = df[:train_len]
df_test = df[train_len:]
df_test = df_test.reset_index(drop=True)

print('Start grouping.')

# grouping with fixedSize window
session_train_df = fixedSize_window(
    df_train[['Content', 'Label']],
    window_size=window_size, step_size=step_size
)

# grouping with fixedSize window
session_test_df=fixedSize_window(
    df_test[['Content', 'Label']],
    window_size=window_size, step_size=step_size
)

  col = ['Content', 'Label','item_Label']
    spliter=' ;-; '

    session_train_df = session_train_df[col]
    session_train_df['session_length'] = session_train_df["Content"].apply(len)
    session_train_df["Content"] = session_train_df["Content"].apply(lambda x: spliter.join(x))

    mean_session_train_len = session_train_df['session_length'].mean()
    max_session_train_len = session_train_df['session_length'].max()
    num_anomalous_train= session_train_df['Label'].sum()
    num_normal_train = len(session_train_df['Label']) - session_train_df['Label'].sum()

    session_test_df = session_test_df[col]
    session_test_df['session_length'] = session_test_df["Content"].apply(len)
    session_test_df["Content"] = session_test_df["Content"].apply(lambda x: spliter.join(x))

    mean_session_test_len = session_test_df['session_length'].mean()
    max_session_test_len = session_test_df['session_length'].max()
    num_anomalous_test= session_test_df['Label'].sum()
    num_normal_test = len(session_test_df['Label']) - session_test_df['Label'].sum()


    session_train_df.to_csv(os.path.join(output_dir, 'train.csv'),index=False)
    session_test_df.to_csv(os.path.join(output_dir, 'test.csv'),index=False)

    print('Train dataset info:')
    print(f"max session length: {max_session_train_len}; mean session length: {mean_session_train_len}\n")
    print(f"number of anomalous sessions: {num_anomalous_train}; number of normal sessions: {num_normal_train}; number of total sessions: {len(session_train_df['Label'])}\n")

    print('Test dataset info:')
    print(f"max session length: {max_session_test_len}; mean session length: {mean_session_test_len}\n")
    print(f"number of anomalous sessions: {num_anomalous_test}; number of normal sessions: {num_normal_test}; number of total sessions: {len(session_test_df['Label'])}\n")

    with open(os.path.join(output_dir, 'train_info.txt'), 'w') as file:
        # 写入内容到文件
        file.write(f"max session length: {max_session_train_len}; mean session length: {mean_session_train_len}\n")
        file.write(f"number of anomalous sessions: {num_anomalous_train}; number of normal sessions: {num_normal_train}; number of total sessions: {len(session_train_df['Label'])}\n")

    with open(os.path.join(output_dir, 'test_info.txt'), 'w') as file:
        # 写入内容到文件
        file.write(f"max session length: {max_session_test_len}; mean session length: {mean_session_test_len}\n")
        file.write(f"number of anomalous sessions: {num_anomalous_test}; number of normal sessions: {num_normal_test}; number of total sessions: {len(session_test_df['Label'])}\n")
'''