import numpy as np
import random

class random_process:
    def __init__(self,func,props):
        self.func=func
        self.props=props
        self.state=np.array([random.gauss(props['mu_init'][x],props['sig_init'][x]) for x in xrange(props['numDim'])])
        self.num_steps=20
        self.mu=0
        self.sig=1
    def rand(self):
        return np.array([random.gauss(self.mu,self.sig) for x in xrange(self.props['numDim'])])
    def sample(self):
        for _i in xrange(self.num_steps):
            test_point=self.state+self.rand()
            accept=random.random()
            if self.func(test_point)/self.func(self.state)>accept:
                self.state=test_point
        return self.state.copy()

bimode_props={'numDim':1,'mu_init':[0],'sig_init':[1]}
def bimode(x):
    mu1=-1
    mu2=1
    sig=0.5
    return (np.exp(-0.5*(x[0]-mu1)/sig*(x[0]-mu1))+np.exp(-0.5*(x[0]-mu2)/sig*(x[0]-mu2)))

bimode_2_props={'numDim':2,'mu_init':[0,0],'sig_init':[1,1]}
def bimode_2(x):
    mu1=-1
    mu2=1
    sig=0.5
    return (np.exp(-0.5*(x[0]-mu1)/sig*(x[0]-mu1))+np.exp(-0.5*(x[0]-mu2)/sig*(x[0]-mu2)))*(np.exp(-0.5*x[1]/sig*x[1]))

def main():
    bimode_rand=random_process(bimode,bimode_props)
    print np.mean([bimode_rand.sample() for x in xrange(200)],0)
    bimode_rand=random_process(bimode_2,bimode_2_props)
    print np.mean([bimode_rand.sample() for x in xrange(200)],0)

if __name__=="__main__":
    main()
