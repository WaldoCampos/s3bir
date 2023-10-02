import argparse
import configparser

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', help='path to the config file', required=True)
    args = parser.parse_args()
    config = configparser.ConfigParser()
    config.read(args.config)

    