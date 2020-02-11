##########################################################################
############################## parametres ################################
##########################################################################


############################################
### differentes librairies necessaires : ###
############################################ 
from __future__ import division #permet de faire des divisions qui renvoient des floats
import sys
import numpy as np #pour les operations matrices
from matplotlib import pyplot as plt #affichege des courbes
#from IPython import embed #permet de stopper le programme a un certain endroit
from scipy.sparse.linalg import LinearOperator, cg #gradient conjugue pour les matrices
#from cmath import * #operation sur les complexes
import scipy
import scipy.linalg   # SciPy Linear Algebra Library
#set_log_level(WARNING)#permet de ne pas afficher tous les warnings
import scipy.sparse as sp



##################
### parametres ###
##################


# Nombre de points en espace et en temps
R_list=np.array(input('Nb de point en espace (entre crochets): '))
M_list=np.array(input('Nb de point en temps  (entre crochets): '))
# Nombre de points en espace et en temps test
#R_list=np.array([100])
#M_list=np.array([200])




#borne en espace #les plot sont donne sur [0,2pi]
xmin = 0
xmax = 1

#bornes de omega (domaine de controle)
window_min = 0.5
window_max = 1.0



#discretisation en temps
Temps = 2 #temps final


#penalite controle optimal
#expo=int(input('exposant sur h dans phi :'))
expo=4

#parametre gradient conjugue 
tolerance = 1e-15 #tolerance
#itemax = 2000# int(raw_input('Nombre t iterations maximum pour le gradient conjugue : '))  #limite d iterations

#diffusion   ###vaut 1 lorsque l on se place sur [0,2pi] en espace
diffusion=(4.0*np.pi**2)**(-1)

#nombre de controle
m=1 

#nombre d equation
n=2#test
#n=int(raw_input('Nombre d equation : '))

#cas controlable ou non controlable
cas_test=int(raw_input('1) alpha=1 (Controlable)\n2) alpha=sum (1/p^2)cos(15*p*x) (Non controlable)\n3)x\n4) alpha=(x-pi/2)*1_[0,pi]\n5)alpha=1_[pi/8,2*pi/8]\n6)alpha=cos(5*x)\n7)alpha=sum (1/p^2)sin(p.x)\n8)alpha=sum (1/p^2)cos(p.x)\nChoix : '))

#nombre de composantes controlees 
proj=1  

#####################################################################################
### construction de la donnee initiale et des matrices de couplage et de controle ###
#####################################################################################
def y0_A_B(n, R, space_disc):
    if n==2:
        #donnee initiale :
        y0 = np.array([[np.sin(2*np.pi*x)] for x in space_disc][1:-1], dtype=np.float64)
        y0 = np.vstack((np.zeros_like(y0), y0))
        if cas_test==1:
             A=np.array([[np.array([0.0 for x in space_disc], dtype=np.float64),np.array([1.0 for x in space_disc], dtype=np.float64)], [np.array([0.0 for x in space_disc], dtype=np.float64), np.array([0.0 for x in space_disc], dtype=np.float64)]])
        #matrices du systeme :
        if cas_test==2:
             A=np.array([[np.array([0.0 for x in space_disc], dtype=np.float64),np.array([sum(np.cos(30*p*np.pi*x)/p**2 for p in range(1,200)) for x in space_disc], dtype=np.float64)], [np.array([0.0 for x in space_disc], dtype=np.float64), np.array([0.0 for x in space_disc], dtype=np.float64)]])
	if cas_test==3:
             A=np.array([[np.array([0.0 for x in space_disc], dtype=np.float64),np.array([x-np.pi/4 if x<=1/2 else 0 for x in space_disc], dtype=np.float64)], [np.array([0.0 for x in space_disc], dtype=np.float64), np.array([0.0 for x in space_disc], dtype=np.float64)]])
        if cas_test==4:
             A=np.array([[np.array([0.0 for x in space_disc], dtype=np.float64),np.array([x if x <= 0.5 else 0 for x in space_disc], dtype=np.float64)], [np.array([0.0 for x in space_disc], dtype=np.float64), np.array([0.0 for x in space_disc], dtype=np.float64)]])
        if cas_test==5:
             A=np.array([[np.array([0.0 for x in space_disc], dtype=np.float64),np.array([1.0 if x >= 1/8 and x <=2/8 else 0 for x in space_disc], dtype=np.float64)], [np.array([0.0 for x in space_disc], dtype=np.float64), np.array([0.0 for x in space_disc], dtype=np.float64)]])
        if cas_test==6:
             A=np.array([[np.array([0.0 for x in space_disc], dtype=np.float64),np.array([np.cos(10*np.pi*(x)) for x in space_disc], dtype=np.float64)], [np.array([0.0 for x in space_disc], dtype=np.float64), np.array([0.0 for x in space_disc], dtype=np.float64)]])
	if cas_test==7:
             A=np.array([[np.array([0.0 for x in space_disc], dtype=np.float64),np.array([sum(np.sin(2*(p)*np.pi*x)/(p**2) for p in range(1,50)) for x in space_disc], dtype=np.float64)], [np.array([0.0 for x in space_disc], dtype=np.float64), np.array([0.0 for x in space_disc], dtype=np.float64)]])
        if cas_test==8:
             A=np.array([[np.array([0.0 for x in space_disc], dtype=np.float64),np.array([sum(np.cos(2*p*np.pi*x)/p**2 for p in range(1,50)) for x in space_disc], dtype=np.float64)], [np.array([0.0 for x in space_disc], dtype=np.float64), np.array([0.0 for x in space_disc], dtype=np.float64)]])

        #matrice de controle :
        B=np.array([[1],[0]]) #que des 1 et des 0 !!!!
    if n==3:
        penalty = 10e30#float(R)**expo
        print("penalty = h^{0} = {1}^{0} = {2}".format(expo, float(R), penalty))
        #nombre de composantes controlees :
        proj=3
        #donnee initiale :
        y0 = np.array([[1*np.sin(np.pi*x)] for x in space_disc][1:-1], dtype=np.float64)
        y0 = np.vstack((np.zeros_like(y0), np.zeros_like(y0),y0))
        #matrices du systeme :
        A=np.array([[np.array([0.0 for x in space_disc], dtype=np.float64),np.array([0.0 for x in space_disc], dtype=np.float64),np.array([0.0 for x in space_disc], dtype=np.float64)],[np.array([1.0 for x in space_disc], dtype=np.float64),np.array([0.0 for x in space_disc], dtype=np.float64),np.array([0.0 for x in space_disc], dtype=np.float64)],[np.array([0.0 for x in space_disc], dtype=np.float64),np.array([1.0 for x in space_disc],dtype=np.float64),np.array([0.0 for x in space_disc], dtype=np.float64)]])
        #matrice de controle :
        B=np.array([[1],[0],[0]]) #que des 1 et des 0 !!!!
    return y0, A, B

