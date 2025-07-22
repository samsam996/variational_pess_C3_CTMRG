import torch



def density_matrix_pess(A1symm, A2symm, C, Ea, Eb):

    """
        ARGS: 
            - A1symm & A2symm : local tensors
            - C : Corner matrix of the CTMRG environment
            - Ea & Eb : row tensors of the CTMRG environment 

        return the three sites density matrix for the iPESS wavefunction on the triangular lattice
    
    """
    
    Da = A1symm.size()
    D = Da[1]
    d = Da[0]

    Tda = torch.einsum('mefg,nabc->eafbgcmn',(A1symm,torch.conj(A1symm))).reshape(D**2, D**2, D**2, d, d)    
    Ta = torch.einsum('mefg,mabc->eafbgc',(A1symm,torch.conj(A1symm))).reshape(D**2, D**2, D**2)

    Tb = torch.einsum('efg,abc->eafbgc',(A2symm,torch.conj(A2symm))).reshape(D**2, D**2, D**2)

    """
       C -f--- Ea -- a
       |e      |g
       |       Tb -- p
       |      / l  
       Eb-j-Ta 
       |    |
       i    k
    """

    CEETdT = torch.einsum('ije,ef,fga,jlksu,lgp->ikapsu',(Eb,C,Ea,Tda,Tb)) # da db da' db' = s t u v
 
    Rho = torch.einsum('ijaprw,aplmsu,lmijtv->rstwuv',(CEETdT,CEETdT,CEETdT))  # s t u v =  da db da' db' 
    Rho = Rho.reshape(d**3,d**3)  # dbda db'da'
    Rho = 0.5*(Rho + torch.conj(Rho.t()))
    Rho = Rho/Rho.trace()

    return Rho



def get_obs_honeycomb(A1symm, A2symm, H, Sx, Sy, Sz, C, Ea, Eb):
    
    Rho1 = RhoAB(A1symm, A2symm, C, Ea, Eb)
    Rho2 = RhoBA(A1symm, A2symm, C, Ea, Eb)

    Ener1 = torch.mm(Rho1,H).trace()
    Ener2 = torch.mm(Rho2,H).trace()

    Mx = torch.mm(Rho1,Sx).trace()
    My = torch.mm(Rho1,Sy).trace()
    Mz = torch.mm(Rho1,Sz).trace()  

    eig1, lambda1 = torch.linalg.eig(Rho1)
    eig2, lambda2 = torch.linalg.eig(Rho2)
        
    return Ener1, Ener2, Mx, My, Mz



def get_energy_pess(A1symm, A2symm, H, Sx, Sy, Sz, C, Ea, Eb):

    Rho = density_matrix_pess(A1symm, A2symm, C, Ea, Eb)
    Ener = torch.mm(Rho, H).trace()

    Mx = torch.mm(Rho, Sx).trace()
    My = torch.mm(Rho, Sy).trace()
    Mz = torch.mm(Rho, Sz).trace()

    return Ener, Mx, My, Mz




def RhoBA(A1symm, A2symm, C, Ea, Eb):

    Da = A1symm.size()
    D = Da[1]
    d = Da[0]

    Tda = torch.einsum('mefg,nabc->eafbgcmn',(A1symm,torch.conj(A1symm))).reshape(D**2, D**2, D**2, d, d)
    Tdb = torch.einsum('mefg,nabc->eafbgcmn',(A2symm,torch.conj(A2symm))).reshape(D**2, D**2, D**2, d, d)
    
    Ta = torch.einsum('mefg,mabc->eafbgc',(A1symm,torch.conj(A1symm))).reshape(D**2, D**2, D**2)
    Tb = torch.einsum('mefg,mabc->eafbgc',(A2symm,torch.conj(A2symm))).reshape(D**2, D**2, D**2)

    EETT = torch.einsum('ijk,kmn,jlp,mpq->ilnq',(Ea,Eb,Tb,Ta))   # CEE(ijga)*Tab(jkpg) = CEETT(i,a,k,p)
    CEETT2 = torch.einsum('ijlk,lm,mkpq->ijpq',(EETT,C,EETT))
    C3 = torch.einsum('ij,jkml,mp->ikpl',(C,CEETT2,C))
    TT = torch.einsum('ijk,klm,pqjst,qrluv->ipmrsutv',(Ea, Eb, Tdb, Tda))  # db da db' da'
    

    Rho = torch.einsum('kjpl,plkjsutv->usvt',(C3,TT))
    Rho = Rho.reshape((d**2,d**2))   # dbda db'da'

    Rho = 0.5*(Rho + torch.conj(Rho.t()))
    Rho = Rho/Rho.trace()

    return Rho



def RhoAB(A1symm, A2symm,  C, Ea, Eb):

    Da = A1symm.size()
    D = Da[1]
    d = Da[0]

    Tda = torch.einsum('mefg,nabc->eafbgcmn',(A1symm,torch.conj(A1symm))).reshape(D**2, D**2, D**2, d, d)
    Tdb = torch.einsum('mefg,nabc->eafbgcmn',(A2symm,torch.conj(A2symm))).reshape(D**2, D**2, D**2, d, d)
    
    Ta = torch.einsum('mefg,mabc->eafbgc',(A1symm,torch.conj(A1symm))).reshape(D**2, D**2, D**2)
    Tb = torch.einsum('mefg,mabc->eafbgc',(A2symm,torch.conj(A2symm))).reshape(D**2, D**2, D**2)

    """
       C -f--- Ea -- a
       |e      |g
       |       Tb -- p
       |      / l  
       Eb-j-Ta 
       |    |
       i    k
    
    Ta : jlk and Tb : lgp    
    """

    CEETT = torch.einsum('ije,ef,fga,jlk,lgp->ikap',(Eb,C,Ea,Ta,Tb))
    CEETdTd = torch.einsum('ije,ef,fga,jlksu,lgptv->ikapstuv',(Eb,C,Ea,Tda,Tdb)) # da db da' db' = s t u v
    Rho = torch.einsum('ijkl,klpq,pqijstuv->stuv',(CEETT,CEETT,CEETdTd))  # s t u v =  da db da' db' 
    Rho = Rho.reshape(d**2,d**2)  # dbda db'da'
    
    Rho = 0.5*(Rho + torch.conj(Rho.t()))
    Rho = Rho/Rho.trace()

    return Rho
