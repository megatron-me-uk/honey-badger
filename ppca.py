import numpy as np
from random import gauss,random
from scipy.stats import nanmean

def randn():
    return gauss(0,1)

def pca(data):
    cov=np.cov(data)
    evs,eigs=np.linalg.eig(cov)
    return evs,eigs

MAX_ITERS=10000

def ppca(data,no_pcs=None):
    data=data.copy().T
    mean=nanmean(data,0)
    data=data-[mean for i in xrange(data.shape[0])]
    unknown_idx=np.where(np.isfinite(data)==False)
    num_missing=len(unknown_idx[0])
    for i in xrange(num_missing):
        data[tuple(x[i] for x in unknown_idx)]=randn()
    num_points,dims=data.shape
    if no_pcs==None:
        no_pcs=dims-1
    expected=np.array([[randn() for x in xrange(no_pcs)] for x in xrange(num_points)])
    pcs=np.array([[randn() for x in xrange(no_pcs)] for x in xrange(dims)])
    noise_variance=1e3
    noise_variance_old=np.inf
    cov_model=(pcs.T).dot(pcs)
    converged=1e-10
    iters=0
    while (noise_variance_old-noise_variance)>converged and iters<MAX_ITERS:
        #Expectation
        inv_Sig=np.linalg.inv(np.eye(no_pcs)+cov_model/noise_variance)
        noise_variance_old=noise_variance
        if num_missing:
            projection=expected.dot(pcs.T)
            for i in xrange(num_missing):
                data[tuple(x[i] for x in unknown_idx)]=projection[tuple(x[i] for x in unknown_idx)]
        expected=data.dot(pcs).dot(inv_Sig/noise_variance)
        #Maximisation
        sum_expected_squared=expected.T.dot(expected)
        pcs=data.T.dot(expected).dot(np.linalg.inv(sum_expected_squared+num_points*inv_Sig))
        cov_model=(pcs.T).dot(pcs)
        noise_variance=(np.sum((expected.dot(pcs.T)-data)**2)+num_points*np.sum(cov_model.dot(inv_Sig))+num_missing*noise_variance_old)/(num_points*dims)
        iters+=1
    if iters==MAX_ITERS:
        print 'maximum iterations reached'
    if num_missing:
        return pcs,data+[mean for i in xrange(num_points)]
        #[tuple(x[0] for x in unknown_idx),:]
    else:
        return pcs

def main():
    import mcmc
    def mvn(x):
        return np.exp(-0.5*(x).dot(np.linalg.inv(np.array([[1,-0.8],[-0.8,3]]))).dot(x))
    mvn_process=mcmc.random_process(mvn,{'numDim':2,'mu_init':[0,0],'sig_init':[3,4]})
    data=np.array([mvn_process.sample() for i in xrange(1000)]).T
    data[0,:]+=1
    print data
    print np.mean(data,1)
    print np.cov(data)
    print '---PCA---'
    print pca(data)
    print '---PPCA, first component---'
    print ppca(data)
    print '---PPCA, both components---'
    print ppca(data,2)
    data2=np.concatenate((data,np.array([[1,np.nan,2],[np.nan,3,np.nan]])),1)
    print data2
    print '---PPCA, missing data---'
    print ppca(data2)
    print '---PPCA, missing data, both components---'
    print ppca(data2,2)
if __name__=="__main__":
    main()
