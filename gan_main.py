#! python3

import argparse
import importlib
import logging
import os
import shutil
import urllib3
import zipfile

import train_data_bigan

# Logging
console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter('[%(asctime)s %(levelname)-3s @%(name)s] %(message)s', datefmt='%H:%M:%S'))
logging.basicConfig(level=logging.DEBUG, handlers=[console])
logging.getLogger("tensorflow").setLevel(logging.WARNING)
logger = logging.getLogger("AnomalyDetection")


def run(args):
    print("""
    starting ...

""")

    has_effect = False
    mod_name="bigan.run6"
    mod = importlib.import_module(mod_name)
    a={'DoS GoldenEye': 1, 'FTP-Patator': 1, 'SSH-Patator': 1, 'Heartbleed': 1, 'BENIGN': 1, 'DoS Hulk': 1, 'DDoS': 1, 'PortScan': 1, 'DoS slowloris': 1, 'Web Attack \x96 Brute Force': 1, 'Web Attack \x96 XSS': 1, 'Web Attack \x96 Sql Injection': 1, 'Bot': 1, 'DoS Slowhttptest': 1, 'Infiltration': 1}
    for i in range(1):
        mod.run(args.nb_epochs, args.w, args.m, args.d, args.label, 0,1639,args.rd)
    i=5
    # for i in range(i):
    #     mod.run(args.nb_epochs, args.w, args.m, args.d, args.label,0.1,args.rd)
    #
    # for i in range(i):
    #     mod.run(args.nb_epochs, args.w, args.m, args.d, args.label,0.2,args.rd)
    #
    # for i in range(i):
    #     mod.run(args.nb_epochs, args.w, args.m, args.d, args.label,0.3,args.rd)
    #
    # for i in range(i):
    #     mod.run(args.nb_epochs, args.w, args.m, args.d, args.label,0.4,args.rd)
    #
    # for i in range(i):
    #     mod.run(args.nb_epochs, args.w, args.m, args.d, args.label,0.5,args.rd)
    #
    # for i in range(i):
    #     mod.run(args.nb_epochs, args.w, args.m, args.d, args.label,0.6,args.rd)
    #
    # for i in range(i):
    #     mod.run(args.nb_epochs, args.w, args.m, args.d, args.label,0.7,args.rd)
    #
    # for i in range(i):
    #     mod.run(args.nb_epochs, args.w, args.m, args.d, args.label,0.8,args.rd)
    #
    # for i in range(i):
    #     mod.run(args.nb_epochs, args.w, args.m, args.d, args.label,0.9,args.rd)
    #
    # for i in range(i):
    #     mod.run(args.nb_epochs, args.w, args.m, args.d, args.label,0.98,args.rd)
    # mod.run(args.nb_epochs, args.w, args.m, args.d, args.label, 0, args.rd)


def path(d):
    try:
        assert os.path.isdir(d)
        return d
    except Exception as e:
        raise argparse.ArgumentTypeError("Example {} cannot be located.".format(d))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run examples from the DL 2.0 Anomaly Detector.')
    parser.add_argument('--nb_epochs', nargs="?", type=int, default=300,help='number of epochs you want to train the dataset on')
    parser.add_argument('--label', nargs="?", default=1, type=int, help='anomalous label for the experiment')
    parser.add_argument('--w', nargs="?", default=0.1, type=float,
                        help='weight for the sum of the mapping loss function')
    parser.add_argument('--m', nargs="?", default='cross-e', choices=['cross-e', 'fm'],
                        help='mode/method for discriminator loss')
    parser.add_argument('--d', nargs="?", default=1, type=int, help='degree for the L norm')
    parser.add_argument('--rd', nargs="?", default=42, type=int, help='random_seed')

    run(parser.parse_args())
