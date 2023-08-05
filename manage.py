import argparse
import code
import logging
import readline
import rlcompleter

logger = logging.getLogger(__name__)


def initialise_command_line_args():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--example", help="argument to be specified for examples")
    arg_parser.add_argument(
        "--delay",
        nargs="?",
        default=False,
        const=True,
        help="Specify to run the task in background",
    )
    arg_parser.add_argument(
        "--console",
        nargs="?",
        default=False,
        const=True,
        help="Use this to open interactive console",
    )
    return arg_parser


def init_task(func, delay):
    if delay:
        method = func.delay
    else:
        method = func
    method()


def open_console():
    vars = globals()
    vars.update(locals())
    readline.set_completer(rlcompleter.Completer(vars).complete)
    readline.parse_and_bind("tab: complete")
    shell = code.InteractiveConsole(vars)
    shell.interact()


if __name__ == "__main__":
    parser = initialise_command_line_args()
    args = vars(parser.parse_args())
    if args.get("example", None):
        from task import *

        init_task(globals()[args.get("example")], args.get("delay"))
    elif args.get("console", None):
        open_console()
    else:
        raise "Specify task or one_timer to execute"
