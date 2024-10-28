import torch


def get_obs_honeycomb(A1symm, A2symm, H, Sx, Sy, Sz, C, Ea, Eb):
    # C(d,r), E(u,r,d)
    
    Da = A1symm.size()
    D = Da[1]
    d = Da[0]

    Tda = torch.einsum('mefg,nabc->eafbgcmn',(A1symm,torch.conj(A1symm))).reshape(D**2, D**2, D**2, d, d)
    Tdb = torch.einsum('mefg,nabc->eafbgcmn',(A2symm,torch.conj(A2symm))).reshape(D**2, D**2, D**2, d, d)
    
    Ta = torch.einsum('mefg,mabc->eafbgc',(A1symm,torch.conj(A1symm))).reshape(D**2, D**2, D**2)
    Tb = torch.einsum('mefg,mabc->eafbgc',(A2symm,torch.conj(A2symm))).reshape(D**2, D**2, D**2)




    #    C -f--- Ea -- a
    #    |e      |g
    #    |       Tb -- p
    #    |      / l  
    #    Eb-j-Ta 
    #    |    |
    #    i    k
        
    ### COMPUTING ENER 1

    CEETT = torch.einsum('ije,ef,fga,jkl,lpg->ikap',(Eb,C,Ea,Ta,Tb))
    CEETdTd = torch.einsum('ije,ef,fga,jklsu,lpgtv->ikapstuv',(Eb,C,Ea,Tda,Tdb))
    Rho = torch.einsum('ijkl,klpq,pqijstuv->stuv',(CEETT,CEETT,CEETdTd)) 
    Rho = Rho.reshape(d**2,d**2)
    
    Rho1 = 0.5*(Rho + torch.conj(Rho.t()))
    Rho1 = Rho1/Rho1.trace()

    Tnorm = Rho1.trace() 
    
    Ener1 = torch.mm(Rho1,H).trace()/Tnorm
    Mx = torch.mm(Rho1,Sx).trace()/Tnorm
    My = torch.mm(Rho1,Sy).trace()/Tnorm
    Mz = torch.mm(Rho1,Sz).trace()/Tnorm

    ### COMPUTING ENER 2

    EETT = torch.einsum('ijk,kmn,jlp,mpq->ilnq',(Ea,Eb,Tb,Ta))   # CEE(ijga)*Tab(jkpg) = CEETT(i,a,k,p)
    CEETT2 = torch.einsum('ijlk,lm,mkpq->ijpq',(EETT,C,EETT))
    C3 = torch.einsum('ij,jkml,mp->ikpl',(C,CEETT2,C))
    TT = torch.einsum('ijk,klm,pqjst,qrluv->ipmrsutv',(Ea, Eb, Tdb, Tda))
    Rho2 = torch.einsum('kjpl,plkjsutv->sutv',(C3,TT))
    Rho2 = Rho2.reshape((d**2,d**2))
    Rho2 = Rho2/Rho2.trace()
    Rho2 = 0.5*(Rho2 + Rho2.t())
    Ener2 = torch.mm(Rho2,H).trace()/Rho2.trace()

    ### COMPUTING ENER 3

    h0 = H.reshape(d,d,d,d)
    hh = torch.einsum('lijk,okmn,lopu,pqrs,uswv->iqjrmwnv',(A1symm, A2symm, h0, torch.conj(A1symm), torch.conj(A2symm))).reshape(D**2,D**2,D**2,D**2)

    CEETT = torch.einsum('ije,ef,fga,jkl,lpg->ikap',(Eb,C,Ea,Ta,Tb))
    CEEH = torch.einsum('ije,ef,fga,jkpg->ikap',(Eb,C,Ea,hh))
    Z = torch.einsum('ijkl,klpq,pqij',(CEETT,CEETT,CEETT))
    ee = torch.einsum('ijkl,klpq,pqij',(CEETT,CEEH,CEETT))
    
    Ener3 = ee/Z

    Energy = (Ener1)
   
    # print('Ener1 ', Ener1)
    # print('Ener2 ', Ener2)
    # print('Ener3 ', Ener3)

    # Mx = torch.tensor([0])
    # My = torch.tensor([0])
    # Mz = torch.tensor([0])

    # print(type(Mx))

    return Energy, Mx, My, Mz

