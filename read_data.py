import numpy as np
import h5py
import os
import quantities as quant

def load_grid(params,grid):
	dfile = h5py.File(os.path.join(params['dumpsdir'],'torus.out0.{:05d}.h5'.format(int(sorted(list(filter(lambda dump: 'torus' in dump,os.listdir(params['dumpsdir']))))[0][-7:-3]))),'r')
	grid['a'] = dfile['header/geom/fmks/a'][()]
	grid['rEH'] = dfile['header/geom/fmks/r_eh'][()]
	grid['dx1'] = dfile['/header/geom/dx1'][()]
	grid['dx2'] = dfile['/header/geom/dx2'][()]
	grid['dx3'] = dfile['/header/geom/dx3'][()]
	grid['ndim'] = dfile['/header/n_dim'][()]
	grid['n1'] = dfile['header/n1'][()]
	grid['n2'] = dfile['header/n2'][()]
	grid['n3'] = dfile['header/n3'][()]
	grid['type'] = dfile['/header/type'][()].decode('UTF-8')
	dfile.close()
	gfile = h5py.File(os.path.join(params['dumpsdir'],'grid.h5'),'r')
	grid['X1'] = gfile['X1'][()]
	grid['X2'] = gfile['X2'][()]
	grid['X3'] = gfile['X3'][()]
	grid['r'] = gfile['r'][()]
	grid['th'] = gfile['th'][()]
	grid['phi'] = gfile['phi'][()]
	grid['x'] = gfile['x'][()]
	grid['y'] = gfile['y'][()]
	grid['z'] = gfile['z'][()]
	grid['lapse'] = gfile['lapse'][()]
	grid['gcov'] = gfile['gcov'][()]
	grid['gcon'] = gfile['gcon'][()]
	grid['gdet'] = gfile['gdet'][()]
	gfile.close()
	

def load_dump(dumpfile,dump,grid):
	dfile = h5py.File(dumpfile,'r')
	dump['t'] = dfile['t'][()]
	dump['gam'] = dfile['/header/gam'][()]
	dump['nprim'] = dfile['/header/n_prim'][()]
	dump['rho'] = dfile['prims'][()][Ellipsis,0]
	dump['uu'] = dfile['prims'][()][Ellipsis,1]
	dump['u'] = dfile['prims'][()][Ellipsis,2:5]
	dump['B'] = dfile['prims'][()][Ellipsis,5:8]
	dump['ucon'],dump['ucov'],dump['bcon'],dump['bcov'] = quant.compute_ub(dump,grid)
	dump['pg'] = (dump['gam']-1)*dump['uu']
	dump['bsq'] = np.einsum('ijkm,ijkm->ijk',dump['bcon'],dump['bcov'])
	dump['beta'] = 2*dump['pg']/dump['bsq']
	dump['sigma'] = dump['bsq']/dump['rho']
	dfile.close()
