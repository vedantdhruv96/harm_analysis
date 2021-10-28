import multiprocessing as mp
import psutil

def calc_threads(pad=0.4):
	Nthreads = int(psutil.cpu_count(logical=False)*pad)
	return Nthreads

def run_parallel(function,dlist,Nthreads):
	pool = mp.Pool(Nthreads)
	pool.map_async(function,dlist).get(720000)
	pool.close()
	pool.join()
