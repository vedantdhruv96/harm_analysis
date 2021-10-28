import numpy as np
import h5py, os, sys
import scipy.fftpack as fft
import pickle
import integrate
import quantities as quant
import read_data as read
import parallelize as par

# read params and grid
modeldir = sys.argv[1]
dfirst = int(sys.argv[2])
dlast = int(sys.argv[3])
params = pickle.load(open(os.path.join(modeldir,'params.p'),'rb'))
grid = pickle.load(open(os.path.join(modeldir,'grid.p'),'rb'))

def compute(dumpnum):
	
	# loading dump
	dump = {}
	read.load_dump(os.path.join(params['dumpsdir'],'torus.out0.{:05d}.h5'.format(dumpnum)),dump,grid)
	
	# calculating relevant indices
	rEHind = np.argmin(abs(grid['r'][:,0,0]-grid['rEH']))
	routind = np.argmin(abs(grid['r'][:,0,0]-params['rout']))
	thmin = np.pi/3.
	thmax = 2*np.pi/3.
	thminind = np.argmin(abs(grid['th'][-1,:,0]-thmin))
	thmaxind = np.argmin(abs(grid['th'][-1,:,0]-thmax))
	
	# EH-threading flux
	# eqns (13)-(16) in SANECC
	mdot = -integrate.shell_sum(dump['rho']*dump['ucon'][Ellipsis,1],grid,rEHind)
	phibh = 0.5*integrate.shell_sum(abs(dump['B'][Ellipsis,0]),grid,rEHind)
	ldot = integrate.shell_sum(quant.Tmixed(dump,i=1,j=3),grid,rEHind)
	edot = integrate.shell_sum(-quant.Tmixed(dump,i=1,j=0),grid,rEHind)
	
	# disk-averging
	# eqn (17) in SANECC
	rhodiskavg = integrate.disk_avg(dump['rho'],grid,thminind,thmaxind)
	pgdiskavg = integrate.disk_avg(dump['pg'],grid,thminind,thmaxind)
	Bdiskavg = integrate.disk_avg(np.sqrt(dump['bsq']),grid,thminind,thmaxind)
	bsqdiskavg = integrate.disk_avg(dump['bsq'],grid,thminind,thmaxind)
	ptotdiskavg = integrate.disk_avg(dump['pg']+dump['bsq'],grid,thminind,thmaxind)
	uphidiskavg = integrate.disk_avg(dump['ucon'][Ellipsis,3],grid,thminind,thmaxind)
	sigmadiskavg = integrate.disk_avg(dump['sigma'],grid,thminind,thmaxind)
	betainvdiskavg = integrate.disk_avg((0.5*dump['bsq']/dump['pg']),grid,thminind,thmaxind)

	# scale-height
	# eqn (21) in SANECC
	hbyr = integrate.rho_weighted_frac(abs(np.pi/2-grid['th']),dump['rho'],grid)
	
	# jet power calculation
	# eqns (10) and (11) in PaperV (2019)
	fdiff = (np.sum((-quant.Tmixed(dump,i=1,j=0)-dump['rho']*dump['ucon'][Ellipsis,1])*grid['gdet'],axis=2)*grid['dx3'])[routind,:]
	fout = np.mean(-quant.Tmixed(dump,i=1,j=0),axis=2)[routind,:]
	fm = np.mean(dump['rho']*dump['ucon'][Ellipsis,1],axis=2)[routind,:]
	
	# rotational profile
	omega = integrate.rho_weighted_frac(dump['ucon'][Ellipsis,3]/dump['ucon'][Ellipsis,0],dump['rho'],grid)
	
	# azimuthal averaging
	# eqn (19) in SANECC	
	rhophiavg = np.mean(dump['rho'],axis=2)
	sigmaphiavg = np.mean(dump['sigma'],axis=2)
	betainvphiavg = np.mean(0.5*dump['bsq']/dump['pg'],axis=2)
	Thetaphiavg = np.mean(dump['pg']/dump['rho'],axis=2)
	
	# azimuthal correlation functions at 4 radii
	radii = [10,20,30,50]
	radii_indices = []
	for r in radii:
		radii_indices.append(np.argmin(abs(grid['r'][:,0,0]-r)))
	x2min = grid['n2']//2-1; x2max = grid['n2']//2+1
	Rrho = np.zeros((len(radii),grid['n3']),dtype=float)
	Rbetainv = np.zeros((len(radii),grid['n3']),dtype=float)
	Rbsq = np.zeros((len(radii),grid['n3']),dtype=float)
	for i in range(len(radii_indices)):
		rho_phi = np.mean(dump['rho'][radii_indices[i],x2min:x2max,:],axis=0)
		rho_phi_normal = (rho_phi-np.mean(rho_phi))/np.std(rho_phi)
		rho_corr = fft.ifft(np.abs(fft.fft(rho_phi_normal))**2)
		betainv_phi = np.mean((0.5*dump['bsq']/dump['pg'])[radii_indices[i],x2min:x2max,:],axis=0)
		betainv_phi_normal = (betainv_phi-np.mean(betainv_phi))/np.std(betainv_phi)
		betainv_corr = fft.ifft(np.abs(fft.fft(betainv_phi_normal))**2)
		bsq_phi = np.mean(dump['bsq'][radii_indices[i],x2min:x2max,:],axis=0)
		bsq_phi_normal = (bsq_phi-np.mean(bsq_phi))/np.std(bsq_phi)
		bsq_corr = fft.ifft(np.abs(fft.fft(bsq_phi_normal))**2)
		Rrho[i] = np.real(rho_corr)/rho_corr.size
		Rbetainv[i] = np.real(betainv_corr)/betainv_corr.size
		Rbsq[i] = np.real(bsq_corr)/bsq_corr.size

	normR_rho = Rrho[:,0]
	normR_betainv = Rbetainv[:,0]
	normR_bsq = Rbsq[:,0]
	for k in range(grid['n3']):
		Rrho[:,k]/=normR_rho
		Rbetainv[:,k]/=normR_betainv
		Rbsq[:,k]/=normR_bsq

	# conserved currents
	jmcon = np.mean(np.einsum('ijk,ijkm->ijkm',dump['rho'],dump['ucon']),axis=2)
	jecon = np.zeros_like(jmcon)
	jlcon = np.zeros_like(jmcon)
	for mu in range(0,grid['ndim']):
		jecon[Ellipsis,mu] = np.mean(quant.Tmixed(dump,i=mu,j=0),axis=2)
		jlcon[Ellipsis,mu] = np.mean(quant.Tmixed(dump,i=mu,j=3),axis=2)
			
	
	# writing reductions for post-processing and plotting
	hfp = h5py.File(os.path.join(params['modeldir'],'reductions_{}{}_{:04d}.h5'.format(grid['type'],grid['spinstring'],dumpnum)),'w')
	hfp['t'] = dump['t']
	hfp['mdot'] = mdot
	hfp['phibh'] = phibh
	hfp['ldot'] = ldot
	hfp['edot'] = edot
	hfp['rhodiskavg'] = rhodiskavg
	hfp['pgdiskavg'] = pgdiskavg
	hfp['Bdiskavg'] = Bdiskavg
	hfp['bsqdiskavg'] = bsqdiskavg
	hfp['ptotdiskavg'] = ptotdiskavg
	hfp['uphidiskavg'] = uphidiskavg
	hfp['sigmadiskavg'] = sigmadiskavg
	hfp['betainvdiskavg'] = betainvdiskavg
	hfp['hbyr'] = hbyr
	hfp['fdiff'] = fdiff
	hfp['fout'] = fout
	hfp['fm'] = fm
	hfp['omega'] = omega
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

if __name__=="__main__":
	dlist = list(range(dfirst,dlast+1))
	pad = 0.6
	Nthreads = par.calc_threads(pad)
	par.run_parallel(compute,dlist,Nthreads)
