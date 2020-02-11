###########################################################################################################
### resolution de y'=Delta y + Az +B1_{omega}u  euler implicite en temps et elements finis P1 en espace ###
###########################################################################################################




############################################
### differentes librairies necessaires : ###
############################################
from __future__ import division #permet de faire des divisions qui renvoient des floats
import sys
import numpy as np #pour les operations matrices
from matplotlib import pyplot as plt #affichege des courbes
from IPython import embed #permet de stopper le programme a un certain endroit
from scipy.sparse.linalg import LinearOperator, cg #gradient conjugue pour les matrices
#from cmath import * #operation sur les complexes
from dolfin import * #librairie elements finis
import scipy
import scipy.linalg   # SciPy Linear Algebra Library
import scipy.sparse as sp
#from params import *
#set_log_level(WARNING)#permet de ne pas afficher tous les warnings


##############################
### differentes parametres ###
##############################
### borne en espace ###
# xmin dans params
# xmax dans params

### bornes de omega (domaine de controle) ###
#window_min dans params
#window_max dans params

### discretisation en espace ###
#IMPORTANT : variable dont depend la penalite et les pas de temps et espace 
#R dans params : Nombre d intervalle en espace : 
#dx dans params : pas d'espace
#steps_x dans params : nombre de pas non nul (sans les bords)
#space_discretization dans params : maillage

###  discretisation en temps ###
#Temps dans params : temps final
#M dans params : Nombre d intervalle de temps  
#dt dans params : pas de temps
#steps_t dans params : nombre de pas


### matrices du systeme ###
#n dans params : nombre d equation
#m dans params : nombre de controle
#A dans params : matrice de couplage
#B dans params : matrice de controle
#P dans params : matrice de projection




### indicatrice devant le controle ### INUTIL !!!
#def ind(x):
#    res=0.0
#    if x>=window_min and x<=window_max:
#        res=1.0
#    return res

############################################################
### construction des matrices de projection  P  discrete ###
############################################################
def P_disc_fct(steps_x,proj,n):
    blocks_P=[]
    for i in range(proj):
        row=[]
        for j in range(n):
            if i==j:
                row.append(sp.spdiags(np.ones(steps_x), [0], steps_x, steps_x))
            if i<>j:
                row.append(sp.spdiags(np.zeros(steps_x), [0], steps_x, steps_x))
        blocks_P.append(row)
    return sp.bmat(blocks_P, format="csc")



###########################
### condition aux bords ###
###########################
def boundary(x, on_boundary):
	return on_boundary


##########################################
### calcul des matrices de propagation ###
##########################################
def mat_matT_fct(R,xmin,xmax,n,A,dt,diffusion):
    mesh = IntervalMesh(R, xmin, xmax)
    V = FunctionSpace(mesh, "P", 1)# espace de dimension finie P1
    #DG = FunctionSpace(mesh, "DG", 1)# espace de dimension finie galerkine discontinue
    dofmap = V.dofmap()
    bc = DirichletBC(V, 0.0, boundary)#condition aux bord
    u = TrialFunction(V)#fonctions test
    v = TestFunction(V)#variable de formulation variationnelle
    K=assemble(inner(grad(u),grad(v))*dx)#matrice de corelation
    K=sp.csc_matrix(K.array()[1:-1,1:-1])
    M1=assemble(u*v*dx)
    M1=sp.csc_matrix(M1.array()[1:-1,1:-1])
    blocks = []
    for i in range(n):
        row = []
        for j in range(n):
            a=Function(V)
            a.vector()[:]=A[i,j]
            MA=assemble(a*u*v*dx)
            MA=sp.csc_matrix(MA.array()[1:-1,1:-1])
            if i == j:
                row.append(M1+dt*diffusion*K-dt*MA)
            else:
                row.append(-dt*MA)
        blocks.append(row)
    mat = sp.bmat(blocks, format="csc")
    matT =  mat.T.tocsc()
    return mat, matT


########################
### matrice de masse ###
########################
def M_n_fct(R,xmin,xmax,n):
    mesh = IntervalMesh(R, xmin, xmax)#maillage en espace
    V = FunctionSpace(mesh, "P", 1)# espace de dimension finie P1
    #DG = FunctionSpace(mesh, "DG", 1)# espace de dimension finie galerkine discontinue
    dofmap = V.dofmap()
    bc = DirichletBC(V, 0.0, boundary)#condition aux bord
    u = TrialFunction(V)#variable de la formulation variationnelle
    v = TestFunction(V)#fonction test
    K=assemble(inner(grad(u),grad(v))*dx)#matrice de corelation
    K=sp.csc_matrix(K.array()[1:-1,1:-1])
    M1=assemble(u*v*dx)
    M1=sp.csc_matrix(M1.array()[1:-1,1:-1])
    blocks = []
    for i in range(n):
        row = []
        for j in range(n):
            if i == j:
                row.append(M1)
            else:
                row.append(sp.csc_matrix(np.zeros((R-1,R-1))))
        blocks.append(row)
    return sp.bmat(blocks, format="csc")


####################################
### matrice de controle discrete ###
####################################
def B_disc_fct(window_min,window_max,space_disc,steps_x,n,m,B):
    ind_vec = sp.spdiags(np.array([1.0 if x >= window_min and x <= window_max else 0 for x in space_disc]), [0], steps_x, steps_x)
    blocks = []
    for i in range(n):
        row = []
        for j in range(m):
            row.append(ind_vec*B[i,j])
        blocks.append(row)
    return sp.bmat(blocks, format="csc")




##############################################
### resolution de l'equation de la chaleur ###
##############################################
def heat(y0,u,M_n,heat_solver,steps_t,B_disc,dt):
    y=np.copy(y0)
    for i in range(steps_t-1):
        #utilisation de la decomposition LU: P*mat=L*U
        y1 = np.array([heat_solver.solve(M_n.dot(y[:,-1]) + dt*M_n.dot(B_disc.dot(u[:,i])))]).T
        y=np.append(y, y1, axis=1)
    return y

####################################################
### resolution de l'equation de la chaleur duale ###
####################################################
def reverse_heat(phiT,M_n,reverse_heat_solver,steps_t):
    phi=np.copy(phiT)
    for i in range(steps_t-1):
        phi1 = np.array([reverse_heat_solver.solve(M_n.dot(phi[:,-1]))]).T
        phi=np.append(phi, phi1, axis=1)
    return phi[:,::-1]

###############################################
### test de la solution libre sans controle ###
###############################################
def test():
    u=np.zeros((m*(R-1),steps_t))
    for i in range(steps_t):
         u[:,i]=y0[m*(R-1),:]
    y=heat(y0,u)
    plt.plot(y)
    plt.show()
    return

#test()
