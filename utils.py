import os, subprocess

def compute_command(modeldir,pwd,dfirst,dlast):
	diagfile = os.path.join(pwd,'diagnostics.py')
	command = 'python3 {} {} {:d} {:d}'.format(diagfile,modeldir,dfirst,dlast)
	return command

def launcher_call(**kwargs):
	os.environ['LAUNCHER_JOB_FILE'] = kwargs['jobfile']
	os.environ['LAUNCHER_WORKDIR'] = kwargs['workdir']
	os.environ['LAUNCHER_PPN'] = str(1)
	os.environ['LAUNCHER_BIND'] = str(1)
	os.environ['LAUNCHER_BIND_HT'] = str(0)
	os.environ['OMP_NUM_THREADS'] = str(int(int(os.environ['SLURM_CPUS_ON_NODE'])/int(os.environ['LAUNCHER_PPN'])))
	subprocess.call('$LAUNCHER_DIR/paramrun',shell=True)

def postprocess_command(modeldir,pwd,dstart,dlast):
	ppfile = os.path.join(pwd,'postprocessing.py')
	command = 'python3 {} {} {:d} {:d}'.format(ppfile,modeldir,dstart,dlast)
	return command
