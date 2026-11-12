"""Exact / iterative / sampling checks on a small known Gaussian Markov model.

This is a numerical diagnostic, NOT MovieLens recommendation evidence.
It holds the likelihood Gaussian so exact posterior moments are available.
"""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[key]='1'
import argparse
import time
from pathlib import Path
import numpy as np
from numba import njit
from numpy.polynomial.hermite import hermgauss
from scipy.special import ndtr
from bayes import local_messages
from data import write_json
from experiments import csvwrite


@njit(cache=True)
def gauss_seidel(j,h,sweeps):
    q=np.zeros(len(h))
    for _ in range(sweeps):
        for i in range(len(h)):
            q[i]=(h[i]-np.dot(j[i],q)+j[i,i]*q[i])/j[i,i]
    return q


@njit(cache=True)
def gibbs(j,h,steps,burn,seed):
    np.random.seed(seed)
    n=len(h);q=np.zeros(n);s=np.zeros(n);ss=np.zeros((n,n))
    for t in range(steps+burn):
        for i in range(n):
            mu=(h[i]-np.dot(j[i],q)+j[i,i]*q[i])/j[i,i]
            q[i]=mu+np.random.normal()/np.sqrt(j[i,i])
        if t>=burn:
            s+=q
            ss+=np.outer(q,q)
    m=s/steps
    return m,ss/steps-np.outer(m,m)


def reference(n):
    nodes,weights=hermgauss(n)
    s=.5+np.sqrt(.8)*nodes[:,None];t=.8+np.sqrt(.8)*nodes[None,:]
    q=weights[:,None]*weights[None,:]/np.pi*ndtr((3+s*t-4.5)/np.sqrt(1+2e-8))
    q/=q.sum();sm=(q*s).sum();tm=(q*t).sum()
    return np.array([sm,(q*(s-sm)**2).sum(),tm,(q*(t-tm)**2).sum(),(q*(s-sm)*(t-tm)).sum()])


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    n=12;w=np.zeros((n,n))
    for i in range(n-1):w[i,i+1]=w[i+1,i]=1.
    w[2,8]=w[8,2]=.5
    lap=np.diag(w.sum(axis=1))-w
    j=.8*np.eye(n)+2*lap+np.diag(np.linspace(.2,1.5,n))
    h=np.sin(np.arange(n)+.4)
    # Warm JIT before timings.
    gauss_seidel(j,h,1);gibbs(j,h,2,1,0)
    t=time.perf_counter();cov=np.linalg.inv(j);mu=np.linalg.solve(j,h);exactsec=time.perf_counter()-t
    rows=[dict(method='ExactGaussian',budget=1,seed=0,mean_rmse=0.,variance_rmse=0.,seconds=exactsec)]
    for steps in [1,5,20,100]:
        t=time.perf_counter();m=gauss_seidel(j,h,steps);sec=time.perf_counter()-t
        rows.append(dict(method='GaussSeidelMAP',budget=steps,seed=0,
            mean_rmse=float(np.sqrt(np.mean((m-mu)**2))),seconds=sec))
    for seed in [42,43,44,45,46]:
        for steps in [1000,5000,20000]:
            t=time.perf_counter();m,c=gibbs(j,h,steps,1000,seed);sec=time.perf_counter()-t
            rows.append(dict(method='Gibbs',budget=steps,seed=seed,
                mean_rmse=float(np.sqrt(np.mean((m-mu)**2))),
                variance_rmse=float(np.sqrt(np.mean((np.diag(c)-np.diag(cov))**2))),seconds=sec))
    assert np.max(np.abs(gauss_seidel(j,h,100)-mu))<1e-10
    csvwrite(args.out/'gaussian_inference.csv',rows)
    np.savez_compressed(args.out/'gaussian_reference.npz',W=w,J=j,h=h,mean=mu,covariance=cov)
    # Graph factorization / precision identity, checked independently from ALS.
    q=np.cos(np.arange(n))
    edge=sum(w[i,k]*(q[i]-q[k])**2 for i in range(n) for k in range(i+1,n))
    assert abs(edge-q@lap@q)<1e-10
    assert np.linalg.eigvalsh(.8*np.eye(n)+2*lap).min()>0
    cp=1/np.array([.4,.4,1e-8,1e-8]);ch=cp*np.array([.5,.8,0.,0.])
    th=np.array([-np.inf,1.5,2.5,3.5,4.5,np.inf])
    mp,mh,used,conv,ok=local_messages(cp,ch,5,3.,1.,th,500,1e-12,.5)
    m=(ch+mh)/(cp+mp);v=1/(cp+mp)
    low,high=reference(40),reference(80);approx=np.array([m[0],v[0],m[1],v[1],0.])
    assert conv and ok and np.max(abs(high-low))<1e-6
    names=['user_mean','user_variance','item_mean','item_variance','user_item_covariance']
    csvwrite(args.out/'bilinear_validation.csv',[dict(quantity=name,quadrature40=float(a),quadrature80=float(b),
        hybrid=float(c),absolute_error=float(abs(b-c)),quadrature_change=float(abs(b-a))) for name,a,b,c in zip(names,low,high,approx)])
    write_json(args.out/'validation.json',dict(gauss_seidel_matches_exact=True,proper_precision=True,
        laplacian_pair_factor_identity=True,bilinear_converged=bool(conv),bilinear_iterations=int(used),
        max_quadrature_change=float(np.max(abs(high-low))),scope='Small Gaussian and one-observation ordinal diagnostics only; not full-data exact inference.'))
    print('Gaussian exact/iterative/Gibbs and bilinear quadrature checks passed.')


if __name__=='__main__':main()
