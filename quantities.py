import numpy as np

def compute_ub(dump,grid):
	gti = grid['gcon'][Ellipsis,0,1:4]
	gij = grid['gcov'][Ellipsis,1:4,1:4]
	beta_i = np.einsum('ijks,ijk->ijks',gti,grid['lapse']**2)
	qsq = np.einsum('ijky,ijky->ijk',np.einsum('ijkxy,ijkx->ijky',gij,dump['u']),dump['u'])
	gamma = np.sqrt(1+qsq)
	ui = dump['u']-np.einsum('ijks,ijk->ijks',beta_i,gamma/grid['lapse'])
	ut = gamma/grid['lapse']
	ucon = np.append(ut[Ellipsis,None],ui,axis=3)
	ucov = np.einsum('ijkmn,ijkn->ijkm',grid['gcov'],ucon)
	bt = np.einsum('ijkm,ijkm->ijk',np.einsum('ijksm,ijks->ijkm',grid['gcov'][Ellipsis,1:4,:],dump['B']),ucon)
	bi = (dump['B']+np.einsum('ijks,ijk->ijks',ui,bt))/ut[Ellipsis,None]
	bcon = np.append(bt[Ellipsis,None],bi,axis=3)
	bcov = np.einsum('ijkmn,ijkn->ijkm',grid['gcov'],bcon)
	return ucon,ucov,bcon,bcov


def Tcon(dump,grid,i=0,j=0):
	return (np.einsum('ijk,ijkmn->ijkmn',dump['rho']+dump['uu']+dump['pg']+dump['bsq'],np.einsum('ijkm,ijkn->ijkmn',dump['ucon'],dump['ucon'])) + np.einsum('ijk,ijkmn->ijkmn',dump['pg']+dump['bsq']/2,grid['gcon']) - np.einsum('ijkm,ijkn->ijkmn',dump['bcon'],dump['bcon']))[Ellipsis,i,j]


def Tcov(dump,grid,i=0,j=0):
	return (np.einsum('ijk,ijkmn->ijkmn',dump['rho']+dump['uu']+dump['pg']+dump['bsq'],np.einsum('ijkm,ijkn->ijkmn',dump['ucov'],dump['ucov'])) + np.einsum('ijk,ijkmn->ijkmn',dump['pg']+dump['bsq']/2,grid['gcov']) - np.einsum('ijkm,ijkn->ijkmn',dump['bcov'],dump['bcov']))[Ellipsis,i,j]


def Tmixed(dump,i=0,j=0):
	if i==j:
		return ((np.einsum('ijk,ijkmn->ijkmn',dump['rho']+dump['uu']+dump['pg']+dump['bsq'],np.einsum('ijkm,ijkn->ijkmn',dump['ucon'],dump['ucov'])) - np.einsum('ijkm,ijkn->ijkmn',dump['bcon'],dump['bcov']))[Ellipsis,i,j] + (dump['pg']+dump['bsq']/2))
	else:
		return (np.einsum('ijk,ijkmn->ijkmn',dump['rho']+dump['uu']+dump['pg']+dump['bsq'],np.einsum('ijkm,ijkn->ijkmn',dump['ucon'],dump['ucov'])) - np.einsum('ijkm,ijkn->ijkmn',dump['bcon'],dump['bcov']))[Ellipsis,i,j]


def TmixedEM(dump,i=0,j=0):
	if i==j:
		return ((np.einsum('ijk,ijkmn->ijkmn',dump['bsq'],np.einsum('ijkm,ijkn->ijkmn',dump['ucon'],dump['ucov'])) - np.einsum('ijkm,ijkn->ijkmn',dump['bcon'],dump['bcov']))[Ellipsis,i,j] + dump['bsq']/2)
	else:
		return (np.einsum('ijk,ijkmn->ijkmn',dump['bsq'],np.einsum('ijkm,ijkn->ijkmn',dump['ucon'],dump['ucov'])) - np.einsum('ijkm,ijkn->ijkmn',dump['bcon'],dump['bcov']))[Ellipsis,i,j]


def TmixedFL(dump,i=0,j=0):
	if i==j:
		return ((np.einsum('ijk,ijkmn->ijkmn',dump['rho']+dump['uu']+dump['pg'],np.einsum('ijkm,ijkn->ijkmn',dump['ucon'],dump['ucov'])))[Ellipsis,i,j] + dump['pg'])
	else:
		return (np.einsum('ijk,ijkmn->ijkmn',dump['rho']+dump['uu']+dump['pg'],np.einsum('ijkm,ijkn->ijkmn',dump['ucon'],dump['ucov'])))[Ellipsis,i,j]
