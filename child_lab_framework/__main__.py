import sys
from colorama import Fore

# from .demo import main
# from .task.depth.depth import main
# from .experiments.parallel_onnx import main
from .demo import new_architecture


if __name__ == '__main__':
    try:
        new_architecture()

    except KeyboardInterrupt:
        print('\n' + Fore.RED + 'Interrupted by user', file=sys.stderr)
