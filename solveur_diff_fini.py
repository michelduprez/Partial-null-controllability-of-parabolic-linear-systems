############################################################################################################
### resolution de y'=Delta y + Az +B1_{omega}u  euler implicite en temps et differences finies en espace ###
############################################################################################################




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
#from dolfin import * #librairie elements finis
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




#############################################
### construction des matrices P  discrete ###
#############################################
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

######################################################
### construction de la matrice de masse : identite ###
######################################################
def M_n_fct(R,xmin,xmax,n):
    blocks_M=[]
    for i in range(n):
        row=[]
        for j in range(n):
            if i==j:
                row.append(sp.spdiags(np.ones(R-1), [0], R-1, R-1))
            if i<>j:
                row.append(sp.spdiags(np.zeros(R-1), [0], R-1, R-1))
        blocks_M.append(row)
    return sp.bmat(blocks_M, format="csc")

def mat_matT_fct(R,xmin,xmax,n,A,dt,diffusion):
    dx_float = (xmax-xmin)/R
    c=diffusion*dt/(dx_float*dx_float)
    laplace_mat = sp.spdiags(np.array([-c*np.ones(R-1),(1+2*c)*np.ones(R-1),-c*np.ones(R-1)]),np.array([-1,0,1]), R-1, R-1)
    I = sp.eye(R-1)
    blocks = []
    for i in range(n):
        row = []
        for j in range(n):
            a=sp.spdiags(A[i,j][1:-1], [0], R-1, R-1)
            if i == j:
                row.append(laplace_mat - dt*a)
            else:
                row.append(-dt*a)
        blocks.append(row)
    mat = sp.bmat(blocks, format="csc")
    matT = mat.T.tocsc()
    return mat, matT

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
    B_disc = sp.bmat(blocks, format="csc")
    return B_disc





##############################################
### resolution de l'equation de la chaleur ###
##############################################
def heat(y0,u,M_n,heat_solver,steps_t,B_disc,dt):
    y=np.copy(y0)
    for i in range(steps_t-1):
        #utilisation de la decomposition LU: P*mat=L*U
        y1 = np.array([heat_solver.solve(y[:,-1] + dt*B_disc.dot(u[:,i]))]).T
        y=np.append(y, y1, axis=1)
    return y


####################################################
### resolution de l'equation de la chaleur duale ###
####################################################
def reverse_heat(phiT,M_n,reverse_heat_solver,steps_t):
    phi=np.copy(phiT)
    for i in range(steps_t-1):
        phi1 = np.array([reverse_heat_solver.solve(phi[:,-1])]).T
        phi=np.append(phi, phi1, axis=1)
    return phi[:,::-1]

###############################################
### test de la solution libre sans controle ###
###############################################
def test():
    contr=np.zeros((m*(R-1),steps_t))
    for i in range(steps_t):
         contr[:,i:i+1]=y0[m*(R-1):]
    y=heat(y0,contr)
    plt.plot(y)
    plt.show()
    embed()
    return

#test()
