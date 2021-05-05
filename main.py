import sys, os


from util.config import Config

def main(argv):
    config = Config()
    #config.LR = config.LR/10
    path = 'config.json'

    if len(argv) == 1:
        print("Using default config, specify the config file path to use custom config...")
    elif len(argv) == 2:
        if argv[1] == '-h' or argv[1] == '--help':
            print("Usage: {} [path to config file]".format(argv[0]))
            print("Refer to config.json and util/config.py for further information about the usage")
            exit(0)
        else:
            path = argv[1]
    else:
        print("Illegal number of arguments, use {0} -h or {0} --help for help".format(argv[0]))
    
    config.load(path)

    return print(config)

    # for compatibility when running in a notebook
    os.chdir('..')
    sys.path.append('src')

    if config.MODEL == 1:
        train('edge', config)
    elif config.MODEL == 2:
        train('sr', config)
    elif config.MODEL == 3:
        train('both', config)

if __name__ == '__main__':
    main(sys.argv)
