import argparse
import subprocess

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cfg', type=str)
    parser.add_argument('-w', '--work_dir', type=str)
    parser.add_argument('--ckpt', type=str)
    args = parser.parse_args()
    
    result = subprocess.run(['python', 'tools/test.py', '-c', args.cfg, '-w', args.work_dir, '--ckpt', args.ckpt, '--selected_seq', '0'])
    print(result.stdout)