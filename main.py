import numpy as np
import h5py
import os,sys,glob,pickle,fnmatch
import read_data as read
import plotall,utils

if __name__=="__main__":
  params = {}
  pwd = os.getcwd()
  if len(sys.argv)>1 and sys.argv[1]=='-p':
    fparams_name = sys.argv[2]
  else:
    sys.exit('No param file provided')
  with open(fparams_name,'r') as fparams:
    lines = fparams.readlines()
    for line in lines:
      if line[0]=='#' or line.isspace():
        pass
      else:
        line_stripped = line.split()
        params[line_stripped[0]] = line_stripped[-1]
        if fnmatch.fnmatch(line_stripped[0],'dstart*') or fnmatch.fnmatch(line_stripped[0],'flux_smoothed'):
          params[line_stripped[0]] = int(line_stripped[-1])
        if line_stripped[-1]=='LAST':
          params[line_stripped[0]] = int(sorted(glob.glob(os.path.join(params['dumpsdir'],'torus.out0*')))[-1][-7:-3])
        if fnmatch.fnmatch(line_stripped[0],'rout') or fnmatch.fnmatch(line_stripped[0],'gam*'):
          params[line_stripped[0]] = float(line_stripped[-1])
  grid = {}
  read.load_grid(params,grid)
  print("--------------------PRE-PROCESSING--------------------")
  print("\nGrid dict generated")
  if int(params['combined_plots']):
    plotall.plot_all_models(params)
  else:
    print("\nWorking on {}{:1.2f}".format(grid['type'],grid['a']))
    num_of_dumps = len(list(range(params['dstart'],params['dlast']+1)))
    alldumps = list(range(params['dstart'],params['dlast']+1))
    dumps_per_rank = int(num_of_dumps/int(os.environ['SLURM_JOB_NUM_NODES']))
    #dumps_per_rank = int(num_of_dumps/20)
    dstarts = np.arange(params['dstart'],params['dlast'],dumps_per_rank)
    dumpslists = []
    for i in range(len(dstarts)):
      if dstarts[i]==dstarts[-1]:
        dumpslists.append(list(np.arange(dstarts[i],params['dlast']+1)))
      else:
        dumpslists.append(list(np.arange(dstarts[i],dstarts[i+1])))
    if grid['a']>0:
      params['modeldir'] = os.path.join(params['grmhddir'],grid['type'],'a+'+'{:.2}'.format(grid['a']))
      grid['spinstring'] = 'a+{:.2}'.format(grid['a'])
    elif grid['a']==0:
      params['modeldir'] = os.path.join(params['grmhddir'],grid['type'],'a0')
      grid['spinstring'] = 'a0'
    else:
      params['modeldir'] = os.path.join(params['grmhddir'],grid['type'],'a'+'{:.2}'.format(grid['a']))
      grid['spinstring'] = 'a{:.2}'.format(grid['a'])
    try:
      os.makedirs(params['modeldir'])
      print('\nModel directory created')
    except OSError:
      print('\nModel directory exists. Continuing...')
      pass
    pickle.dump(params,open(os.path.join(params['modeldir'],'params.p'),'wb'))
    pickle.dump(grid,open(os.path.join(params['modeldir'],'grid.p'),'wb'))
    
    print("\n\n--------------------PROCESSING--------------------\n")
    jobfile = os.path.join(params['modeldir'],'jobfile.txt')
    fjob = open(jobfile,'w')
    for d in range(len(dumpslists)):
      fjob.write(utils.compute_command(params['modeldir'],pwd,dumpslists[d][0],dumpslists[d][-1])+'\n')
    fjob.close()
    os.system('dos2unix {}'.format(jobfile))
    print('\nJobfiles written for computing diagnostics. Parallelizing analysis across nodes using LAUNCHER and across cores using MULTIPROCESSING')
    utils.launcher_call(jobfile=jobfile,workdir=params['modeldir'])
    print('\nDiagnostics for {}{} over dumps {}-{} computed'.format(grid['type'],grid['spinstring'],params['dstart'],params['dlast']))
    
    print("\n\n--------------------POST-PROCESSING--------------------\n")
    dstarts = []
    if int(params['intervals'])!=0:
      for key in params.keys():
        if fnmatch.fnmatch(key,'dstart[0-9]'):
          dstarts.append(params[key])
    dumpslists = []
    for i in range(len(dstarts)):
      if dstarts[i]==dstarts[-1]:
        dumpslists.append(list(np.arange(dstarts[i],params['dlast']+1)))
      else:
        dumpslists.append(list(np.arange(dstarts[i],dstarts[i+1])))
    jobfile_pp = os.path.join(params['modeldir'],'jobfile_pp.txt')
    fjob_pp = open(jobfile_pp,'w')
    for d in range(len(dumpslists)):
      fjob_pp.write(utils.postprocess_command(params['modeldir'],pwd,dumpslists[d][0],dumpslists[d][-1])+'\n')
    fjob_pp.close()
    os.system('dos2unix {}'.format(jobfile_pp))
    print('\nJobfiles written for postprocessing. Parallelizing analysis across nodes using LAUNCHER and across cores using MULTIPROCESSING')
    utils.launcher_call(jobfile=jobfile_pp,workdir=params['modeldir'])
    print('\nPost-processing for {}{} complete'.format(grid['type'],grid['spinstring']))
    """ 
    print("\n\n--------------------PLOTTING--------------------\n")
    os.system('python3 plot.py {}'.format(params['modeldir']))
    print('Plotting done. Figures have been saved at {}'.format(params['modeldir']))
    print("\n\n--------------------COMPLETE--------------------")
    """
