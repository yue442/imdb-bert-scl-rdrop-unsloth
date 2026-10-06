"""Launch one saved configuration in the currently activated environment."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('--output-dir', help='Optional new output directory')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config = json.loads(Path(args.config).read_text(encoding='utf-8'))
    if args.output_dir:
        config['output_dir'] = args.output_dir
    command = [sys.executable, str(root / 'src/train.py')]
    for key, value in config.items():
        option = '--' + key.replace('_', '-')
        if isinstance(value, bool):
            if value:
                command.append(option)
        elif value is not None:
            command.extend([option, str(value)])
    env = os.environ.copy()
    if config['backend'] == 'unsloth':
        env['UNSLOTH_COMPILE_DISABLE'] = '1'
        env['UNSLOTH_DISABLE_FAST_GENERATION'] = '1'
    print(' '.join(command), flush=True)
    subprocess.run(command, cwd=root, env=env, check=True)

if __name__ == '__main__':
    main()
