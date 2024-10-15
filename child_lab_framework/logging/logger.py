import colorama


class Logger:
    @staticmethod
    def info(*args, title: str = 'Info') -> None:
        title = colorama.Fore.GREEN + f'[{title}]:' + colorama.Fore.RESET
        print(title, *args, flush=True)

    @staticmethod
    def error(*args, title: str = 'Error') -> None:
        title = colorama.Fore.RED + f'[{title}]:' + colorama.Fore.RESET
        print(title, *args, flush=True)
