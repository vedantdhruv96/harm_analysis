import numpy as np
import h5py
import os,sys
import matplotlib,pickle,fnmatch
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

modeldir = sys.argv[1]
params = pickle.load(open(os.path.join(modeldir,'params.p'),'rb'))
grid = pickle.load(open(os.path.join(modeldir,'grid.p'),'rb'))

def xz_slice(var, patch_pole=False):
  xz_var = np.zeros((2*grid['n1'],grid['n2']))
  for i in range(grid['n1']):
    xz_var[i,:] = var[grid['n1']-1-i,:,grid['n3']//2]
    xz_var[i+grid['n1'],:] = var[i,:,0]
  if patch_pole:
    xz_var[:,0] = xz_var[:,-1] = 0
  return xz_var

if __name__=="__main__":
  # plot adjustment parameters
  rmax = 50 # max radial value for disk-average and scale-height plots
  cadence = 5 # dump cadence, same as 'cadence' in post_processing
  dspace = 1000 # number of dumps in intervals

  # obtaining starting dump number of each interval
  dstarts = []
  for key in params.keys():
    if fnmatch.fnmatch(key,'dstart[0-9]'):
      dstarts.append(params[key])

  # reading output of postprocessing script
  data_dict = {}
  variables = ['rho','pg','B','bsq','ptot','uphi','sigma','betainv']
  for dstart in dstarts:
    if dstart==dstarts[-1]:
      hfp = h5py.File(os.path.join(modeldir,'more_reductions_{}{}_{:04d}to6000.h5'.format(grid['type'],grid['spinstring'],dstart)),'r')
    else:
      hfp = h5py.File(os.path.join(modeldir,'more_reductions_{}{}_{:04d}to{:04d}.h5'.format(grid['type'],grid['spinstring'],dstart,dstart+dspace-1)),'r')
    data_dict['mdot'+str(dstart)] = hfp['mdot'][()]
    data_dict['phibh'+str(dstart)] = hfp['phibh'][()]
    data_dict['ldot'+str(dstart)] = hfp['ldot'][()]
    data_dict['edot'+str(dstart)] = hfp['edot'][()]
    for var in variables:
      data_dict[var+'diskavg'+str(dstart)] = hfp[var+'diskavg'][()]
    data_dict['hbyr_num'+str(dstart)] = hfp['hbyr_num'][()]
    data_dict['hbyr_denom'+str(dstart)] = hfp['hbyr_denom'][()]
    data_dict['omega_num'+str(dstart)] = hfp['omega_num'][()]
    data_dict['omega_denom'+str(dstart)] = hfp['omega_denom'][()]
    data_dict['rhophiavg'+str(dstart)] = hfp['rhophiavg'][()]	
    data_dict['sigmaphiavg'+str(dstart)] = hfp['sigmaphiavg'][()]	
    data_dict['betainvphiavg'+str(dstart)] = hfp['betainvphiavg'][()]	
    data_dict['Thetaphiavg'+str(dstart)] = hfp['Thetaphiavg'][()]
    data_dict['Rrho'+str(dstart)] = hfp['Rrho'][()]	
    data_dict['Rbetainv'+str(dstart)] = hfp['Rbetainv'][()]	
    data_dict['Rbsq'+str(dstart)] = hfp['Rbsq'][()]
    data_dict['jmcon'+str(dstart)] = hfp['jmcon'][()]
    data_dict['jecon'+str(dstart)] = hfp['jecon'][()]
    data_dict['jlcon'+str(dstart)] = hfp['jlcon'][()]
    hfp.close()

	# plotting disk-averaged radial profiles
  rmaxind = np.argmax(grid['r'][:,0,0]-rmax>0)-1

  fig = plt.gcf()
  fig.set_size_inches(16,9)
  nrows = 2; ncols = 4
  gs = gridspec.GridSpec(nrows=nrows,ncols=ncols,figure=fig)

  ax00 = fig.add_subplot(gs[0,0]); ax00.set_xlabel('$r(GM/c^2)$',size=14); ax00.set_ylabel('$\\langle\\rho\\rangle$',size=14); ax00.tick_params(axis='both',which='major',labelsize=12)
  ax01 = fig.add_subplot(gs[0,1]); ax01.set_xlabel('$r(GM/c^2)$',size=14); ax01.set_ylabel('$\\langle p_{g}\\rangle$',size=14); ax01.tick_params(axis='both',which='major',labelsize=12)
  ax02 = fig.add_subplot(gs[0,2]); ax02.set_xlabel('$r(GM/c^2)$',size=14); ax02.set_ylabel('$\\langle B\\rangle$',size=14); ax02.tick_params(axis='both',which='major',labelsize=12)
  ax03 = fig.add_subplot(gs[0,3]); ax03.set_xlabel('$r(GM/c^2)$',size=14); ax03.set_ylabel('$\\langle b^{2}\\rangle$',size=14); ax03.tick_params(axis='both',which='major',labelsize=12)
  ax10 = fig.add_subplot(gs[1,0]); ax10.set_xlabel('$r(GM/c^2)$',size=14); ax10.set_ylabel('$\\langle p_{tot}\\rangle$',size=14); ax10.tick_params(axis='both',which='major',labelsize=12)
  ax11 = fig.add_subplot(gs[1,1]); ax11.set_xlabel('$r(GM/c^2)$',size=14); ax11.set_ylabel('$\\langle u^{\\phi}\\rangle$',size=14); ax11.tick_params(axis='both',which='major',labelsize=12)
  ax12 = fig.add_subplot(gs[1,2]); ax12.set_xlabel('$r(GM/c^2)$',size=14); ax12.set_ylabel('$\\langle\\sigma\\rangle$',size=14); ax12.tick_params(axis='both',which='major',labelsize=12)
  ax13 = fig.add_subplot(gs[1,3]); ax13.set_xlabel('$r(GM/c^2)$',size=14); ax13.set_ylabel('$\\langle\\beta^{-1}\\rangle$',size=14); ax13.tick_params(axis='both',which='major',labelsize=12)
  for dstart in dstarts:
    ax00.loglog(grid['r'][:rmaxind,0,0],data_dict['rhodiskavg'+str(dstart)][:rmaxind],label=str(dstart))
    ax01.loglog(grid['r'][:rmaxind,0,0],data_dict['pgdiskavg'+str(dstart)][:rmaxind],label=str(dstart))
    ax02.loglog(grid['r'][:rmaxind,0,0],data_dict['Bdiskavg'+str(dstart)][:rmaxind],label=str(dstart))
    ax03.loglog(grid['r'][:rmaxind,0,0],data_dict['bsqdiskavg'+str(dstart)][:rmaxind],label=str(dstart))
    ax10.loglog(grid['r'][:rmaxind,0,0],data_dict['ptotdiskavg'+str(dstart)][:rmaxind],label=str(dstart))
    ax11.loglog(grid['r'][:rmaxind,0,0],data_dict['uphidiskavg'+str(dstart)][:rmaxind],label=str(dstart))
    ax12.loglog(grid['r'][:rmaxind,0,0],data_dict['sigmadiskavg'+str(dstart)][:rmaxind],label=str(dstart))
    ax13.loglog(grid['r'][:rmaxind,0,0],data_dict['betainvdiskavg'+str(dstart)][:rmaxind],label=str(dstart))

  ax00.legend(loc='upper right')
  plt.tight_layout()
  plt.savefig(os.path.join(modeldir,'{}{}_diskavg_radial_profiles.png'.format(grid['type'],grid['spinstring'])))
  plt.clf()

  # flux plots
  # overlapping flux plots for different intervals
  t = np.arange(0,dspace)*5

  fig = plt.gcf()
  fig.set_size_inches(16,9)
  nrows = 2; ncols = 2
  gs = gridspec.GridSpec(nrows=nrows,ncols=ncols,figure=fig)

  ax0 = fig.add_subplot(gs[0,0])
  for dstart in dstarts:
    ax0.plot(t,np.abs(data_dict['mdot'+str(dstart)])[:len(t)],label=str(dstart))
  ax0.set_xlabel('$t(GM/c^3)$')
  ax0.set_ylabel('$\\vert\\dot{M}\\vert$')

  ax1 = fig.add_subplot(gs[0,1])
  for dstart in dstarts:
    ax1.plot(t,np.abs(data_dict['phibh'+str(dstart)])[:len(t)]/np.sqrt(np.abs(data_dict['mdot'+str(dstart)]))[:len(t)],label=str(dstart))
  ax1.set_xlabel('$t(GM/c^3)$')
  ax1.set_ylabel('$\\frac{\\Phi_{BH}}{\\sqrt{\\vert\\dot{M}\\vert}}$')

  ax2 = fig.add_subplot(gs[1,0])
  for dstart in dstarts:
    ax2.plot(t,np.abs(data_dict['ldot'+str(dstart)])[:len(t)]/np.abs(data_dict['mdot'+str(dstart)])[:len(t)],label=str(dstart))
  ax2.set_xlabel('$t(GM/c^3)$')
  ax2.set_ylabel('$\\frac{\\vert\\dot{L}\\vert}{\\vert\\dot{M}\\vert}$')

  ax3 = fig.add_subplot(gs[1,1])
  for dstart in dstarts:
    ax3.plot(t,np.abs(data_dict['edot'+str(dstart)]+data_dict['mdot'+str(dstart)])[:len(t)]/np.abs(data_dict['mdot'+str(dstart)])[:len(t)],label=str(dstart))
  ax3.set_xlabel('$t(GM/c^3)$')
  ax3.set_ylabel('$\\frac{\\vert\\dot{E}-\\dot{M}\\vert}{\\vert\\dot{M}\\vert}$')

  ax0.legend(loc='upper right')
  plt.tight_layout()
  plt.savefig(os.path.join(modeldir,'{}{}_fluxes_overlap.png'.format(grid['type'],grid['spinstring'])))
  plt.clf()
  """
  # smoothed flux plots
  if params['flux_smoothed']:
    
    convol_array = np.ones(dumps_smoothed)
    for dstart in dstarts:
      data_dict['mdot'] = np.concatenate(data_dict['mdot'+str(dstart)]) 
      data_dict['phibh'] = np.concatenate(data_dict['phibh'+str(dstart)]) 
      data_dict['ldot'] = np.concatenate(data_dict['ldot'+str(dstart)]) 
      data_dict['edot'] = np.concatenate(data_dict['edot'+str(dstart)]) 
    t = np.arange(0,data_dict['mdot'].size)*5
    print(data_dict['mdot'].size)

    fig = plt.gcf()
    fig.set_size_inches(7,20)
    nrows = 4; ncols = 1
    gs = gridspec.GridSpec(nrows=nrows,ncols=ncols,figure=fig)

    ax0 = fig.add_subplot(gs[0,0])
    ax0.plot(t,np.convolve(np.abs(data_dict['mdot']),convol_array,'same')/np.abs(data_dict['mdot']))
    ax0.set_xlabel('$t(GM/c^3)$')
    ax0.set_ylabel('$\\vert\\dot{M}\\vert$')

    ax1 = fig.add_subplot(gs[1,0])
    ax1.plot(t,np.convolve(np.abs(data_dict['phibh'+str(dstart)])/np.sqrt(np.abs(data_dict['mdot'])),convol_array,'same')/(np.abs(data_dict['phibh'+str(dstart)])/np.sqrt(np.abs(data_dict['mdot']))))
    ax1.set_xlabel('$t(GM/c^3)$')
    ax1.set_ylabel('$\\frac{\\Phi_{BH}}{\\sqrt{\\vert\\dot{M}\\vert}}$')

    ax2 = fig.add_subplot(gs[2,0])
    ax2.plot(t,np.convolve(np.abs(data_dict['ldot'])/np.abs(data_dict['mdot']),convol_array,'same')/(np.abs(data_dict['ldot'])/np.abs(data_dict['mdot'])))
    ax2.set_xlabel('$t(GM/c^3)$')
    ax2.set_ylabel('$\\frac{\\vert\\dot{L}\\vert}{\\vert\\dot{M}\\vert}$')

    ax3 = fig.add_subplot(gs[3,0])
    ax3.plot(t,np.convolve(np.abs(data_dict['edot']+data_dict['mdot'])/np.abs(data_dict['mdot']),convol_array,'same')/(np.abs(data_dict['edot']+data_dict['mdot'])/np.abs(data_dict['mdot'])))
    ax3.set_xlabel('$t(GM/c^3)$')
    ax3.set_ylabel('$\\frac{\\vert\\dot{E}-\\dot{M}\\vert}{\\vert\\dot{M}\\vert}$')

    ax0.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(modeldir,'{}{}_fluxes_smoothed.png'.format(grid['type'],grid['spinstring'])))
    plt.clf()
  """ 
  # scale height plots    
  fig = plt.gcf()
  fig.set_size_inches(7,7)
  ax = plt.gca()
  for dstart in dstarts:
    ax.plot(grid['r'][:rmaxind,0,0],(data_dict['hbyr_num'+str(dstart)]/data_dict['hbyr_denom'+str(dstart)])[:rmaxind],label=str(dstart))
  ax.set_xlabel('$r(GM/c^2)$',size=14)
  ax.set_ylabel('$H/R(r)$',size=14)
  ax.tick_params(axis='both',which='major',labelsize=12)
  ax.grid(True)
  ax.legend(loc='best')
  plt.savefig(os.path.join(modeldir,'{}{}_scale_height.png'.format(grid['type'],grid['spinstring'])))
  plt.clf()

  # rotational profile plot
  kepfit = 1/(grid['r']**1.5+grid['a'])

  fig = plt.gcf()
  fig.set_size_inches(7,7)
  ax = plt.gca()
  for dstart in dstarts:
    ax.loglog(grid['r'][:rmaxind,0,0],(data_dict['omega_num'+str(dstart)]/data_dict['omega_denom'+str(dstart)])[:rmaxind],label=str(dstart))
  ax.loglog(grid['r'][:rmaxind,0,0],kepfit[:rmaxind,0,0],label='Keplerian fit')
  ax.set_xlabel('$r(GM/c^2)$',size=14)
  ax.set_ylabel('$\\Omega(r)$',size=14)
  ax.tick_params(axis='both',which='major',labelsize=12)
  ax.grid(True)
  ax.legend(loc='best')
  plt.savefig(os.path.join(modeldir,'{}{}_rotational_profile.png'.format(grid['type'],grid['spinstring'])))
  plt.clf()

  # correlation function plots
  x3_pi_ind = np.argmin(abs(grid['X3'][0,0,:]-np.pi))
  corr_vars = ['Rrho','Rbetainv','Rbsq']

  for corr_var in corr_vars:
    fig = plt.gcf()
    fig.set_size_inches(16,9)
    nrows = 2; ncols = 2
    gs = gridspec.GridSpec(nrows=nrows,ncols=ncols,figure=fig)

    ax0 = fig.add_subplot(gs[0,0])
    for dstart in dstarts:
      ax0.plot(grid['X3'][0,0,:x3_pi_ind],data_dict[corr_var+str(dstart)][0,:x3_pi_ind],label=str(dstart))
    ax0.set_xlim(0,np.pi)
    ax0.set_ylim(-0.5,1)
    ax0.set_xlabel('$\\phi(rad)$',size=14)
    ax0.set_ylabel('$\\bar{R}(r=10M)$',size=14)
    if corr_var=='Rrho':  ax0.set_title('$\\rho$')
    elif corr_var=='Rbetainv':  ax0.set_title('$\\beta^{-1}$')
    elif corr_var=='Rbsq':  ax0.set_title('$b^2$')
    ax0.grid(True)
    
    ax1 = fig.add_subplot(gs[0,1])
    for dstart in dstarts:
      ax1.plot(grid['X3'][0,0,:x3_pi_ind],data_dict[corr_var+str(dstart)][1,:x3_pi_ind],label=str(dstart))
    ax1.set_xlim(0,np.pi)
    ax1.set_ylim(-0.5,1)
    ax1.set_xlabel('$\\phi(rad)$',size=14)
    ax1.set_ylabel('$\\bar{R}(r=20M)$',size=14)
    if corr_var=='Rrho':  ax1.set_title('$\\rho$')
    elif corr_var=='Rbetainv':  ax1.set_title('$\\beta^{-1}$')
    elif corr_var=='Rbsq':  ax1.set_title('$b^2$')
    ax1.grid(True)

    ax2 = fig.add_subplot(gs[1,0])
    for dstart in dstarts:
      ax2.plot(grid['X3'][0,0,:x3_pi_ind],data_dict[corr_var+str(dstart)][2,:x3_pi_ind],label=str(dstart))
    ax2.set_xlim(0,np.pi)
    ax2.set_ylim(-0.5,1)
    ax2.set_xlabel('$\\phi(rad)$',size=14)
    ax2.set_ylabel('$\\bar{R}(r=30M)$',size=14)
    if corr_var=='Rrho':  ax2.set_title('$\\rho$')
    elif corr_var=='Rbetainv':  ax2.set_title('$\\beta^{-1}$')
    elif corr_var=='Rbsq':  ax2.set_title('$b^2$')
    ax2.grid(True)
   
    ax3 = fig.add_subplot(gs[1,1])
    for dstart in dstarts:
      ax3.plot(grid['X3'][0,0,:x3_pi_ind],data_dict[corr_var+str(dstart)][3,:x3_pi_ind],label=str(dstart))
    ax3.set_xlim(0,np.pi)
    ax3.set_ylim(-0.5,1)
    ax3.set_xlabel('$\\phi(rad)$',size=14)
    ax3.set_ylabel('$\\bar{R}(r=50M)$',size=14)
    if corr_var=='Rrho':  ax3.set_title('$\\rho$')
    elif corr_var=='Rbetainv':  ax3.set_title('$\\beta^{-1}$')
    elif corr_var=='Rbsq':  ax3.set_title('$b^2$')
    ax3.grid(True)

    ax0.legend(loc='best')
    plt.savefig(os.path.join(modeldir,'{}{}_corr_'.format(grid['type'],grid['spinstring'])+corr_var+'.png'))
    plt.clf()

  # poloidal slice plots
  x = xz_slice(grid['x'],patch_pole=True)[grid['n1']:,:]
  z = xz_slice(grid['z'])[grid['n1']:,:]
  phiavg_vars = ['rho','sigma','betainv','Theta']
  for var in phiavg_vars:
    fig = plt.gcf()
    fig.set_size_inches(16,9)
    nrows = 1; ncols = len(dstarts)
    gs = gridspec.GridSpec(nrows=nrows,ncols=ncols,figure=fig)
    if var=='rho':  cmap='plasma'; vmin=-5; vmax=0; title='$\\rho$'
    elif var=='sigma':  cmap='GnBu'; vmin=-3; vmax=3; title='$\\sigma$'
    elif var=='betainv':  cmap='BuPu'; vmin=-2; vmax=3; title='$\\beta^{-1}$'
    elif var=='Theta':  cmap='afmhot'; vmin=-3; vmax=-1; title='$\\Theta$'
    for d in range(len(dstarts)):
      var_slice = np.log10(data_dict[var+'phiavg'+str(dstarts[d])])
      ax = fig.add_subplot(gs[0,d])
      polplot = ax.pcolormesh(x,z,var_slice,cmap=cmap,vmin=vmin,vmax=vmax,shading='gouraud')
      ax.set_xlim(0,40)
      ax.set_ylim(-40,40)
      ax.set_xlabel('$x (GM/c^2)$')
      ax.set_ylabel('$z (GM/c^2)$')
      circle = plt.Circle((0,0),grid['rEH'],color='k')
      ax.add_artist(circle)
      ax.set_aspect('equal')
      divider = make_axes_locatable(ax)
      cax = divider.append_axes('right',size='5%',pad=0.05)
      plt.colorbar(polplot,cax=cax)
      ax.set_title(title)
    
    plt.savefig(os.path.join(modeldir,'{}{}_phiavg_'.format(grid['type'],grid['spinstring'])+var+'.png'))
    plt.clf()
