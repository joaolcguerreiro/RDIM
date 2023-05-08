

arser = argparse.ArgumentParser(description='Testing Script')
parser.add_argument('--exp', default=None, help="Path to experiment directory", type=str)
parser.add_argument('--epoch', help="Epoch to be tested", type=str)