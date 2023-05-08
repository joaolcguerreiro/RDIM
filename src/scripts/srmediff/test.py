if __name__ == '__main__':
    # Define argument parser
    parser = argparse.ArgumentParser(description='Training Script')
    parser.add_argument('--exp', default=None, help="Path to experiment directory", type=str)
    args = parser.parse_args()

    if args.exp == parser.get_default('exp'):
        # Create experiment folder

        ...

    # Load configuration file
    with open(os.path.join(args.exp, "config.yaml"), 'r') as f:
        config = yaml.safe_load(f)

    if not config["TEST"]["CHECKPOINT_PATH"]:
        ERROR!!!!!!!!