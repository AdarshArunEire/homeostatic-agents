#temp script
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

def trivial(x):
    return x * 2

def main():
    with ProcessPoolExecutor(max_workers=4) as ex:
        futs = [ex.submit(trivial, i) for i in range(8)]
        for f in as_completed(futs):
            print("got", f.result(), flush=True)

if __name__ == "__main__":
    mp.freeze_support()
    main()