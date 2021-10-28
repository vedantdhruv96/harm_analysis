import numpy as np
import h5py
import os,sys,glob
import matplotlib,pickle
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec
from decimal import Decimal

modeldir = sys.argv[1]
params = pickle.load(open(os.path.join(modeldir,'params.p'),'rb'))
grid = pickle.load(open(os.path.join(modeldir,'grid.p'),'rb'))

def plot_fluxes_entire_range(fig,t,mdot,phibh,ldot,edot):
	gs = gridspec.GridSpec(nrows=4,ncols=1,figure=fig)
	
	ax0 = fig.add_subplot(gs[0,0])
	ax0.plot(t,np.abs(mdot))
	ax0.set_xlabel('$t(GM/c^3)$')
	ax0.set_ylabel('$\\vert\\dot{M}\\vert$')

	ax1 = fig.add_subplot(gs[1,0])
	ax1.plot(t,phibh/np.sqrt(np.abs(mdot)))
	ax1.set_xlabel('$t(GM/c^3)$')
	ax1.set_ylabel('$\\frac{\\Phi_{BH}}{\\sqrt{\\vert\\dot{M}\\vert}}$')

	ax2 = fig.add_subplot(gs[2,0])
	ax2.plot(t,np.abs(ldot)/np.abs(mdot))
	ax2.set_xlabel('$t(GM/c^3)$')
	ax2.set_ylabel('$\\frac{\\vert\\dot{L}\\vert}{\\vert\\dot{M}\\vert}$')

	ax3 = fig.add_subplot(gs[3,0])
	ax3.plot(t,np.abs(edot+mdot)/np.abs(mdot))
	ax3.set_xlabel('$t(GM/c^3)$')
	ax3.set_ylabel('$\\frac{\\vert\\dot{E}-\\dot{M}\\vert}{\\vert\\dot{M}\\vert}$')

	plt.tight_layout()

if __name__=="__main__":
  flux_plots_all_time = 0
  cadence = 5
  dfirst = int(sys.argv[2])
  dlast = int(sys.argv[3])
  Ndumps = dlast-dfirst+1
  dlist = np.arange(dfirst,dlast+1)
  pp_output = os.path.join(modeldir,'{}{}_post_processed_output_{:04d}_{:04d}.txt'.format(grid['type'],grid['spinstring'],dfirst,dlast))
  fpp = open(pp_output,'w')
  fpp.close()
  if (params['intervals']!=0 and dfirst==params['dstart0']):
    flux_plots_all_time = 1

  # printing (and plotting) average fluxes
  if flux_plots_all_time==1:
    dstartall = int(sorted(glob.glob(os.path.join(modeldir,'reductions_*.h5')))[0][-7:-3])
    dlastall = int(sorted(glob.glob(os.path.join(modeldir,'reductions_*.h5')))[-1][-7:-3])
    t = np.arange(dstartall*cadence,(dlastall+1)*cadence,cadence)
    Ntotal = dlastall-dstartall+1
    mdot = np.zeros(Ntotal,dtype=float)
    phibh = np.zeros_like(mdot)
    ldot = np.zeros_like(mdot)
    edot = np.zeros_like(mdot)
    for i in range(Ntotal):
      hfp = h5py.File(os.path.join(modeldir,'reductions_{}{}_{:04d}.h5'.format(grid['type'],grid['spinstring'],i)),'r')
      mdot[i] = hfp['mdot'][()]
      phibh[i] = hfp['phibh'][()]
      ldot[i] = hfp['ldot'][()]
      edot[i] = hfp['edot'][()]
      hfp.close()
    fpp = open(pp_output,'a')
    fpp.write('AVERAGE FLUXES {:04d}-{:04d}\n'.format(dfirst,dlast))
    fpp.write('\nAccretion rate: {:5.2f}'.format(np.abs(mdot[dfirst:dlast+1]).mean()))
    fpp.write('\nMagnetic flux: {:5.2f}'.format(phibh[dfirst:dlast+1].mean()/np.sqrt(np.abs(mdot[dfirst:dlast+1]).mean())))
    fpp.write('\nAngular momentum flux: {:5.2f}'.format(np.abs(ldot[dfirst:dlast+1]).mean()/np.abs(mdot[dfirst:dlast+1]).mean()))
    fpp.write('\nEnergy flux: {:5.2f}'.format(np.abs(edot+mdot)[dfirst:dlast+1].mean()/np.abs(mdot[dfirst:dlast+1]).mean()))
    fpp.close()
    
    fig = plt.gcf()
    plot_fluxes_entire_range(fig,t,mdot,phibh,ldot,edot)
    plt.savefig(os.path.join(modeldir,'{}{}_fluxes_all_time.png'.format(grid['type'],grid['spinstring'])))
    plt.clf()

  else:
    mdot = np.zeros(Ndumps,dtype=float)
    phibh = np.zeros_like(mdot)
    ldot = np.zeros_like(mdot)
    edot = np.zeros_like(mdot)
    for i in range(dfirst,dlast+1):
      hfp = h5py.File(os.path.join(modeldir,'reductions_{}{}_{:04d}.h5'.format(grid['type'],grid['spinstring'],i)),'r')
      mdot[i-dfirst] = hfp['mdot'][()]
      phibh[i-dfirst] = hfp['phibh'][()]
      ldot[i-dfirst] = hfp['ldot'][()]
      edot[i-dfirst] = hfp['edot'][()]
      hfp.close()
    fpp = open(pp_output,'a')
    fpp.write('AVERAGE FLUXES {:04d}-{:04d}\n'.format(dfirst,dlast))
    fpp.write('\nAccretion rate: {:5.2f}'.format(np.abs(mdot).mean()))
    fpp.write('\nMagnetic flux: {:5.2f}'.format(phibh.mean()/np.sqrt(np.abs(mdot).mean())))
    fpp.write('\nAngular momentum flux: {:5.2f}'.format(np.abs(ldot).mean()/np.abs(mdot).mean()))
    fpp.write('\nEnergy flux: {:5.2f}'.format(np.abs(edot+mdot).mean()/np.abs(mdot).mean()))
    fpp.close()
      
  # disk profile dict	
  variables = ['rho','pg','B','bsq','ptot','uphi','sigma','betainv']
  disk_dict = {}
  for var in variables:
    disk_dict[var] = np.zeros(grid['n1'],dtype=float)

  # scale height arrays
  hbyr_num = np.zeros(grid['n1'],dtype=float)
  hbyr_denom = np.zeros_like(hbyr_num)

  # jet power arrays
  fdiff = np.zeros(grid['n2'],dtype=float)
  fout = np.zeros_like(fdiff)
  fm = np.zeros_like(fdiff)
  mdot_pj = 0.0

  # rotational profile array
  omega_num = np.zeros(grid['n1'],dtype=float) 
  omega_denom = np.zeros_like(omega_num)

  # poloidal slices array
  rhophiavg = np.zeros((grid['n1'],grid['n2']),dtype=float)
  sigmaphiavg = np.zeros_like(rhophiavg)
  betainvphiavg = np.zeros_like(rhophiavg)
  Thetaphiavg = np.zeros_like(rhophiavg)

  # correlation function arrays
  radii = [10,20,30,50]
  Rrho = np.zeros((len(radii),grid['n3']),dtype=float)
  Rbetainv = np.zeros_like(Rrho)
  Rbsq = np.zeros_like(Rrho)

  # currents arrays
  jmcon = np.zeros((grid['n1'],grid['n2'],grid['ndim']),dtype=float)
  jecon = np.zeros_like(jmcon)
  jlcon = np.zeros_like(jmcon)
    
  # reading data
  for i in range (dfirst,dlast+1):
    hfp = h5py.File(os.path.join(modeldir,'reductions_{}{}_{:04d}.h5'.format(grid['type'],grid['spinstring'],i)),'r')
    for var in variables:
      disk_dict[var]+=np.array(hfp[var+'diskavg'])
    hbyr_num += hfp['hbyr'][()][0]
    hbyr_denom += hfp['hbyr'][()][1]
    fdiff += hfp['fdiff'][()]
    fout += hfp['fout'][()]
    fm += hfp['fm'][()]
    mdot_pj += hfp['mdot'][()]
    omega_num += hfp['omega'][()][0]
    omega_denom += hfp['omega'][()][1]
    rhophiavg += hfp['rhophiavg'][()]	
    sigmaphiavg += hfp['sigmaphiavg'][()]	
    betainvphiavg += hfp['betainvphiavg'][()]
    Thetaphiavg += hfp['Thetaphiavg'][()]
    Rrho += hfp['Rrho'][()]	
    Rbetainv += hfp['Rbetainv'][()]	
    Rbsq += hfp['Rbsq'][()]
    jmcon += hfp['jmcon'][()]
    jecon += hfp['jecon'][()]
    jlcon += hfp['jlcon'][()]
    hfp.close()

  # time-averaging disk-averaged radial profiles
  for var in variables:
    disk_dict[var]/=Ndumps

  # time-averaging scale-height radial profile
  hbyr_num/=Ndumps		
  hbyr_denom/=Ndumps

  # jet power calculation
  routind = np.argmin(abs(grid['r'][:,0,0]-params['rout']))
  fdiff/=Ndumps
  fout/=Ndumps
  fm/=Ndumps
  mdot_pj/=Ndumps
  bgsq = (fout/fm)**2-1
  bg = np.sqrt(np.where(bgsq<0,0,bgsq))
  bgcut = 1
  pjet = np.where((bg>bgcut)&((grid['th'][routind,:,0]<1.)|(grid['th'][routind,:,0]>np.pi-1))&(fdiff>0.),fdiff,0.)
  pjet = np.sum(pjet)*grid['dx2']
  pout = np.where(((grid['th'][routind,:,0]<1.)|(grid['th'][routind,:,0]>np.pi-1))&(fdiff>0.),fdiff,0)
  pout = np.sum(pout)*grid['dx2']
  fpp = open(pp_output,'a')
  fpp.write('\nAccretion rate: {:.4E}'.format(Decimal(mdot_pj)))
  fpp.write('\nJet power: {:.4E}'.format(Decimal(pjet)))
  fpp.write('\nJet power efficienty: {:.4E}'.format(Decimal(pjet/mdot_pj)))
  fpp.write('\nOutput power: {:.4E}'.format(Decimal(pout)))
  fpp.write('\nOutput power efficienty: {:.4E}'.format(Decimal(pout/mdot_pj)))
  fpp.close()

  # time-averaging rotational profile
  omega_num/=Ndumps		
  omega_denom/=Ndumps

  # time-averaging poloidal slices
  rhophiavg/=Ndumps
  sigmaphiavg/=Ndumps
  betainvphiavg/=Ndumps
  Thetaphiavg/=Ndumps

  # time-averaging correlation profiles
  Rrho/=Ndumps
  Rbetainv/=Ndumps
  Rbsq/=Ndumps

  # time-averaging conserved currents
  jmcon/=Ndumps
  jecon/=Ndumps
  jlcon/=Ndumps

  # storing post-processed output for amalgamated plots
  hfp = h5py.File(os.path.join(modeldir,'more_reductions_{}{}_{:04d}to{:04d}.h5'.format(grid['type'],grid['spinstring'],dfirst,dlast)),'w')
  if flux_plots_all_time:
    hfp['mdot'] = mdot[dfirst:dlast+1]
    hfp['phibh'] = phibh[dfirst:dlast+1]
    hfp['ldot'] = ldot[dfirst:dlast+1]
    hfp['edot'] = edot[dfirst:dlast+1]
  else:
    hfp['mdot'] = mdot
    hfp['phibh'] = phibh
    hfp['ldot'] = ldot
    hfp['edot'] = edot
  for var in variables:
    hfp[var+'diskavg'] = disk_dict[var]
  hfp['hbyr_num'] = hbyr_num
  hfp['hbyr_denom'] = hbyr_denom
  hfp['omega_num'] = omega_num
  hfp['omega_denom'] = omega_denom
  hfp['rhophiavg'] = rhophiavg
  hfp['sigmaphiavg'] = sigmaphiavg
  hfp['betainvphiavg'] = betainvphiavg
  hfp['Thetaphiavg'] = Thetaphiavg
  hfp['Rrho'] = Rrho
  hfp['Rbetainv'] = Rbetainv
  hfp['Rbsq'] = Rbsq
  hfp['jmcon'] = jmcon
  hfp['jecon'] = jecon
  hfp['jlcon'] = jlcon
  hfp.close()
