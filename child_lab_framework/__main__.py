import sys

from colorama import Fore

# from .demo import main
from .demo_sequential import main

if __name__ == '__main__':
    try:
        main()
        # asyncio.run(main())

    except KeyboardInterrupt:
        print('\n' + Fore.RED + 'Interrupted by user' + Fore.RESET, file=sys.stderr)
