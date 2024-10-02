import asyncio
import sys

from colorama import Fore

from .demo import main

if __name__ == '__main__':
    try:
        asyncio.run(main())

    except KeyboardInterrupt:
        print('\n' + Fore.RED + 'Interrupted by user', file=sys.stderr)
