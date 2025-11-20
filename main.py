import argparse
from model.MOLLM import MOLLM
try:
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
except:
    print('rdkit not installed, skip setting its warnings')
import warnings
warnings.simplefilter("ignore", FutureWarning)
def main(arg_list=None):
    parser = argparse.ArgumentParser(description='Run MOLLM with a configuration file')
    
    parser.add_argument('config', type=str, default='<TO_BE_FILLED>',  help='Path to the configuration file (YAML format)')
    parser.add_argument('--resume', action='store_true', help='resume training from the last checkpoint')
    parser.add_argument('--eval', action='store_true', help='evaluate this results according to the yaml file')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--objectives', type=str, nargs='+', default=None)
    parser.add_argument('--directions', type=str, nargs='+', default=None)
    parser.add_argument('--num_offspring', type=int, default=2)
    parser.add_argument('--save_suffix',type=str,default='')
    if arg_list:
        return parser.parse_args(arg_list)
    args = parser.parse_args()

    mollm = MOLLM(args,args.config,resume=args.resume,eval=args.eval,seed=args.seed,objectives=args.objectives,directions=args.directions)
    
    if args.eval:
        print(f'start evaluation of {args.config}')
        mollm.load_evaluate()
    else:
        mollm.run()

if __name__ == "__main__":
    main()
