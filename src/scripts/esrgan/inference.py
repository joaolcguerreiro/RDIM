NOT 100% SURE I WILL IMPLEMENT INFERENCE SCRIPTS

arser = argparse.ArgumentParser(description='Testing Script')
parser.add_argument('--checkpoint', default=None, help='Path to model checkpoint', type=str)
parser.add_argument('--lr-dir', help='Path to low-resolution data directory', type=str)
parser.add_argument('--save-dir', help='Path to directory where results are saved', type=str)


maybe criar um dataset.py para isto (igual ao sr_dataset mas sem as lr e hr transforms, retorna logo a ground truth).
O script faz loop sobre os volumes (que devem ser à partida lr) e super-resolve-os. Dá save na dir passada nos args.
