from argparse import Namespace

from vianu.lasa.__main__ import main

_ARGS = {
    "search": "CABAZITAXEL",
    "source": ["swissmedic"],
    "log_level": "DEBUG",
}

if __name__ == "__main__":
    N_MATCH = 5
    args_ = Namespace(**_ARGS)
    matches = main(args_=args_)
    matches.sort(key=lambda m: m.combined, reverse=True)
    print(f"Best {N_MATCH} matches:")
    for m in matches[:N_MATCH]:
        print(m)
