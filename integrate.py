import numpy as np

def disk_avg(var,grid,thmin,thmax):
	return (np.sum(np.sum(var[:,thmin:thmax,:]*grid['gdet'][:,thmin:thmax,:],axis=2)*grid['dx3'],axis=1)*grid['dx2'])/(np.sum(np.sum(grid['gdet'][:,thmin:thmax,:],axis=2)*grid['dx3'],axis=1)*grid['dx2'])


def shell_sum(var,grid,rind):
	return np.sum(np.sum(var[rind,:,:]*grid['gdet'][rind,:,:],axis=1)*grid['dx3'],axis=0)*grid['dx2']


def rho_weighted_frac(var,rho,grid):
	num = np.sum(np.sum(var*rho*grid['gdet'],axis=2)*grid['dx3'],axis=1)*grid['dx2']
	denom = np.sum(np.sum(rho*grid['gdet'],axis=2)*grid['dx3'],axis=1)*grid['dx2']
	return [num,denom]


def volume_sum(var,grid,subdomain=False,rind=None):
	if subdomain:
		return np.sum(np.sum(np.sum(var[:rind,:,:]*grid['gdet'][:rind,:,:],axis=2)*grid['dx3'],axis=1)*grid['dx2'],axis=0)*grid['dx1']
	else:
		return np.sum(np.sum(np.sum(var*grid['gdet'],axis=2)*grid['dx3'],axis=1)*grid['dx2'],axis=0)*grid['dx1']


def sector_sum(var,grid,thmin,thmax,phimin,phimax,subdomain=False,rind=None):
	if subdomain:
		return np.sum(np.sum(np.sum(var[:rind,thmin:thmax,phimin:phimax]*grid['gdet'][:rind,thmin:thmax,phimin:phimax],axis=2)*grid['dx3'],axis=1)*grid['dx2'],axis=0)*grid['dx1']
	else:
		return np.sum(np.sum(np.sum(var[:,thmin:thmax,phimin:phimax]*grid['gdet'][:,thmin:thmax,phimin:phimax],axis=2)*grid['dx3'],axis=1)*grid['dx2'],axis=0)*grid['dx1']


def phi_avg_r(var,rind):
	return np.mean(var,axis=2)[rind,:]
