"""Run: python tests/check.py. Checks inference and graph properties."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from scipy import sparse
from scipy.stats import truncnorm
import bayes
import gmrf
from reference.original_model import Matchbox
from metrics import evaluate

class Checks(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(12)
        self.u=np.repeat(np.arange(6),5)
        self.i=np.tile(np.arange(5),6)
        self.r=rng.integers(1,6,len(self.u))
        self.o=np.arange(len(self.u))

    def test_original_equivalence(self):
        for mode in ['adf','ep']:
            a=bayes.init(6,5,2,42,self.r,mode)
            b=Matchbox(6,5,k=2,seed=42)
            b.initialize_training(self.r,mode)
            for _ in range(1 if mode=='adf' else 3):
                bayes.step(a,self.u,self.i,self.r,self.o)
                b.fit_pass(self.u,self.i,self.r,self.o)
                for x in ['up','uh','ip','ih']:
                    np.testing.assert_array_equal(a[x],getattr(b,x))
                np.testing.assert_array_equal(bayes.prob(a,self.u,self.i),b.predict_proba(self.u,self.i))

    def test_adf_ep1(self):
        aa=[]
        for mode in ['adf','ep']:
            a=bayes.init(6,5,2,42,self.r,mode)
            bayes.step(a,self.u,self.i,self.r,self.o)
            aa.append(a)
        for x in ['up','uh','ip','ih']:
            np.testing.assert_array_equal(aa[0][x],aa[1][x])
        with self.assertRaises(ValueError):
            bayes.step(aa[0],self.u,self.i,self.r,self.o)
        self.assertEqual(aa[0]['sp'].size,0)

    def test_site_bookkeeping(self):
        a=bayes.init(6,5,2,42,self.r,'ep')
        prior={x:a[x].copy() for x in ['up','uh','ip','ih']}
        for _ in range(3):
            bayes.step(a,self.u,self.i,self.r,self.o)
        for field,site in [('p','sp'),('h','sh')]:
            us,is_=prior['u'+field].copy(),prior['i'+field].copy()
            for z,(u,i) in enumerate(zip(self.u,self.i)):
                us[u,:2]+=a[site][z,:2]; us[u,-1]+=a[site][z,-2]
                is_[i,:2]+=a[site][z,2:4]; is_[i,-1]+=a[site][z,-1]
            np.testing.assert_allclose(us,a['u'+field],atol=1e-12)
            np.testing.assert_allclose(is_,a['i'+field],atol=1e-12)

    def test_truncation(self):
        for mu,var,lo,hi in [(0.,1.,-np.inf,1.5),(3.,2.,3.5,4.5),(5.,.4,4.5,np.inf)]:
            m,v,_,ok=bayes.truncated_moments(mu,var,lo,hi)
            rm,rv=truncnorm.stats((lo-mu)/np.sqrt(var),(hi-mu)/np.sqrt(var),loc=mu,scale=np.sqrt(var),moments='mv')
            self.assertTrue(ok)
            np.testing.assert_allclose([m,v],[rm,rv],atol=1e-12)

    def test_metrics(self):
        r=np.array([1,3,5]); p=np.eye(5)[r-1]
        d=evaluate(r,p)
        for x in ['rmse','mae','nll','brier','ece']:
            self.assertEqual(d[x],0)

    def test_graph(self):
        g=np.array([[1,0],[1,0],[0,1],[0,1],[1,1.]])
        w=gmrf.graph(self.u,self.i,self.r,6,5,g,top=2,mix=.1)
        np.testing.assert_allclose(w.toarray(),w.T.toarray())
        self.assertTrue(np.all(w.data>=0)); self.assertTrue(np.all(w.diagonal()==0))
        l=np.diag(np.asarray(w.sum(axis=1)).ravel())-w.toarray()
        self.assertGreater(np.linalg.eigvalsh(10*np.eye(5)+100*l).min(),0)

    def test_als_decreases_objective(self):
        w=sparse.csr_matrix(np.ones((5,5))-np.eye(5))
        for gamma in [0.,10.]:
            m,trace,_=gmrf.fit(self.u,self.i,self.r,self.u,self.i,self.r,6,5,2,42,w,gamma,epochs=5)
            vals=np.array([t['objective'] for t in trace])
            self.assertTrue(np.all(np.diff(vals)<=1e-8))
            np.testing.assert_allclose(gmrf.prob(m,self.u,self.i).sum(axis=1),1)

    def test_edges_affect_predictions(self):
        w=sparse.csr_matrix(np.ones((5,5))-np.eye(5))
        a,_,_=gmrf.fit(self.u,self.i,self.r,self.u,self.i,self.r,6,5,2,42,w,0,epochs=4,reg=.1)
        b,_,_=gmrf.fit(self.u,self.i,self.r,self.u,self.i,self.r,6,5,2,42,w,10,epochs=4,reg=.1)
        self.assertGreater(np.max(np.abs(gmrf.point(a,self.u,self.i)-gmrf.point(b,self.u,self.i))),1e-4)

    def test_gamma_zero_ignores_edges(self):
        w=sparse.csr_matrix(np.ones((5,5))-np.eye(5)); z=sparse.csr_matrix((5,5))
        a,_,_=gmrf.fit(self.u,self.i,self.r,self.u,self.i,self.r,6,5,2,42,w,0,epochs=3)
        b,_,_=gmrf.fit(self.u,self.i,self.r,self.u,self.i,self.r,6,5,2,42,z,0,epochs=3)
        np.testing.assert_array_equal(a['q'],b['q'])

if __name__=='__main__':
    unittest.main(verbosity=2)
