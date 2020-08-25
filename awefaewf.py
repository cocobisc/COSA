import configparser

config=configparser.ConfigParser()

config['anchor_parameters'] = {
    'size': [8, 16, 32, 64, 128],
    'strides': [2, 4, 8, 16, 32]
}

with open('./test.ini', 'w') as f:
    config.write(f)