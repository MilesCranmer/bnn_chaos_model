import argparse

def parse(glob=False):

    # Instantiate the parser
    parser = argparse.ArgumentParser(description='Optional app description')
    # Required positional argument
    parser.add_argument('--version', type=int, help='')
    parser.add_argument('--total_steps', type=int, default=100000, help='default=100000')
    parser.add_argument('--swa_steps', type=int, default=30000, help='default=30000')
    parser.add_argument('--hidden', type=int, default=40, help='default=40')
    parser.add_argument('--latent', type=int, default=20, help='default=20')
    parser.add_argument('--seed', type=int, default=0, help='default=0')
    parser.add_argument('--beta', type=float, default=0.001, help='default=0.001')
    parser.add_argument('--angles', action='store_true', default=False, help='default=False')
    parser.add_argument('--megno', action='store_true', default=False, help='default=False')
    parser.add_argument('--no_mmr', action='store_true', default=False, help='default=False')
    parser.add_argument('--no_nan', action='store_true', default=False, help='default=False')
    parser.add_argument('--no_eplusminus', action='store_true', default=False, help='default=False')
    parser.add_argument('--power_transform', action='store_true', default=False, help='default=False')
    parser.add_argument('--plot', action='store_true', default=False, help='default=False')
    parser.add_argument('--plot_random', action='store_true', default=False, help='default=False')
    parser.add_argument('--train_all', action='store_true', default=False, help='default=False')
    parser.add_argument('--lower_std', action='store_true', default=False, help='default=False')
    args = parser.parse_args()
    extra = ''
    if args.no_nan:
        extra += '_nonan=1' 
    if args.no_eplusminus:
        extra += '_noeplusminus=1' 
    if args.train_all:
        extra += '_train_all=1' 

    checkpoint_filename = (
            'steps=%d_megno=%d_angles=%d_power=%d_hidden=%d_latent=%d_nommr=%d' %
            (args.total_steps, args.megno, args.angles, args.power_transform, args.hidden, args.latent, args.no_mmr)
        + extra + '_v' + str(args.version)
    )
    if not glob:
        checkpoint_filename += '_%d' %(args.seed,)

    return args, checkpoint_filename
