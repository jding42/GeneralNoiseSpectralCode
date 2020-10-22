''' the spectrum of associated non-backtracking matrix'''

import model
def leading_eigenvalues(lamda):
    size=600;
    lamda=1.5/np.sqrt(size);
    prior='Rademacher';
    #noise='Gaussian';
    noise='ErdosRenyi';
    order=2;
    x,Y=model.spikedTensorModel(lamda,prior,noise,size,order)
    print()