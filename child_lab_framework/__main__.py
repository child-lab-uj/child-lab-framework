import sys
from colorama import Fore

from .demo import main


if __name__ == '__main__':
    try:
        main()

    except KeyboardInterrupt:
        print(Fore.RED + 'Interrupted by user', file=sys.stderr)
