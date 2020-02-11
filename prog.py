#########################
### probleme etudie : ###
#########################

#schema euler implicite en temps et differences/elements finies en espace pour la controlabilite partielle du systeme parabolique:

#y=Delta y + Az +B1_{omega}u
#y=0 sur le bord",
#y(0)=y0

#existe-t-il un controle u tel que Py(T)=0 ? (P matrice de projection)

#fonctionnelle :
#J(u) = 0.5*int B^*u^2 + penalite*0.5*int Py(T)^2
#ou la penalite depend du pas en espace

############################################
### differentes librairies necessaires : ###
############################################
from __future__ import division, print_function #permet de faire des divisions qui renvoient des floats
import sys
import numpy as np #pour les operations matrices
from matplotlib import pyplot as plt #affichege des courbes
from IPython import embed #permet de stopper le programme a un certain endroit
from scipy.sparse.linalg import LinearOperator, cg #gradient conjugue pour les matrices
#from cmath import * #operation sur les complexes
from dolfin import * #librairie elements finis
#import scipy
#import scipy.linalg   # SciPy Linear Algebra Library
r
import scipy.sparse as sp
from matplotlib import pyplot as plt

print()
print("Schema euler implicite en temps et differences/elements finies en espace pour la controlabilite partielle du systeme parabolique:")
print("y=Delta y + Az +B1_{omega}u    (avec A=(0 alpha ; 0 0) et B=e1)")
print("y=0 sur le bord")
print("y(0)=y0")

print()
print("Fonctionnelle :")
print("J(u) = 0.5*int B^*u^2 + penalite*0.5*int Py(T)^2")
print("ou la penalite depend du pas en espace")
print()

from params import * #fichier params.py dans le dossie
##########################
### choix du solveur : ###
##########################
solveur=int(raw_input('1) differences finies, 2) Elements finis  : '))
if solveur==1:
    from solveur_diff_fini import * #fichier solveur_diff_fini.py dans le dossier
else:
    from solveur_elts_fini import *
from plot import *


##############################
### liste des parametres : ###
##############################
#nombre d'intervalle en espace
#R = int(raw_input('Nombre d intervalle en espace : '))

#discretisation en temps
#M=int(raw_input('Nombre d intervalle de temps : '))

### borne en espace ###
# xmin dans params
# xmax dans params

### bornes de omega (domaine de controle) ###
#window_min dans params
#window_max dans params

### penalite controle optimal ###
#expo dans params
#penalty dans params

###  temps ###
#Temps dans params : temps final

### parametre gradient conjugue ###
#tolerance dans params : tolerance
#itemax dans params : Nombre t iterations maximum pour le gradient conjugue

### matrices du systeme ###
#n dans params : nombre d equation
#m dans params : nombre de controle
#A dans params : matrice de couplage
#B dans params : matrice de controle
#proj dans params : nombre de composante de y a annuler

### indicatrice devant le controle ###
#ind dans solveur

### precalcul des matrices pour le schema ###
#mat dans solveur : matrice a gauche dans le schema difference finie

#matB dans solveur : matrice de controle discrete

### factorisation LU de la matrice difference finie ###
#P, L, U dans solveur

### condition initiale ###
#y0 dans params : condition initale


##################################################################
### gramien et operateur minimiser par  le gradient conjugue : ###
##################################################################
#HUM operateur
def Lambda(phi_T,P_disc,n,R,B_disc,M_n,reverse_heat_solver,steps_t,heat_solver,dt):
    phi = reverse_heat(P_disc.T.dot(phi_T),M_n,reverse_heat_solver,steps_t)
    w = heat(np.zeros((n*(R-1),1)),B_disc.T.dot(phi),M_n,heat_solver,steps_t,B_disc,dt)
    return np.transpose([P_disc.dot(w[:,-1])])

#application lineaire du gradient conjugue
def Agc(phi_T,penalty,R,B_disc,P_disc,M_n,reverse_heat_solver,steps_t,heat_solver,dt):
    phi_TT = phi_T.reshape((-1,1))
    penalty_1 = 1.0/penalty
    La = Lambda(phi_TT,P_disc,n,R,B_disc,M_n,reverse_heat_solver,steps_t,heat_solver,dt)
    res = (penalty_1*phi_TT+La).flatten()
    return res


############################################################
### compteur du nombre d'iterations du gradient conjugue ###d
############################################################
itns = 0
def callback(x):
    global itns
    itns += 1


##########################################
### resolution du probleme de controle ###
##########################################
def resolution():
    matrice_J=np.zeros((R_list.shape[0],M_list.shape[0])) #matrice ou sont rempli les valeurs des minimum des fonctionnelles
    matrice_dual=np.zeros((R_list.shape[0],M_list.shape[0])) #matrice ou sont rempli les valeurs des minimum des fonctionnelles duales
    matrice_yT=np.zeros((R_list.shape[0],M_list.shape[0])) #matrice ou sont rempli les valeurs des instants finaux
    matrice_u=np.zeros((R_list.shape[0],M_list.shape[0]))#matrice ou sont rempli les valeurs des normes des controles
    matrice_grad=np.zeros((R_list.shape[0],M_list.shape[0]))#matrice ou sont rempli le nombre d iterations du gradien conjugue
    for i_R in range(R_list.shape[0]):
        for i_M in range(M_list.shape[0]):
            R=R_list[i_R]#nombre de point en espace
            M=M_list[i_M]#nombre de point en temps
            penalty = float(R)**expo #penalite de la fonctionnelle
            dx_float = (xmax-xmin)/R #pas d'espace
            dt = Temps/M #pas de temps
            steps_t = M+1 #nombre de pas
            steps_x = R-1 #nombre de pas non nul (sans les bords)
            space_disc = np.linspace(xmin, xmax,R+1)#intervalle en espace discretise
            y0, A, B = y0_A_B(n, R, space_disc)#calcul de la donnee initiale et des matrices de couplage et de controle (voir params.py
            M_n = M_n_fct(R,xmin,xmax,n)#matrice de masse
            B_disc = B_disc_fct(window_min,window_max,space_disc,steps_x,n,m,B)#matrice de controle discretisee
            P_disc = P_disc_fct(steps_x,proj,n)#matrice de projection discretise
            mat, matT = mat_matT_fct(R,xmin,xmax,n,A,dt,diffusion)
            heat_solver = sp.linalg.splu(mat)# factorisation LU de la matrice de propagation  
            reverse_heat_solver = sp.linalg.splu(matT)# factorisation LU de la matrice de propagation duale 
            z = heat(y0,np.zeros((m*(R-1),steps_t)),M_n,heat_solver,steps_t,B_disc,dt)#calcul de la deuxieme composante
            y_T = P_disc.dot(z[:,-1]) #instant final de la solution libre
            def Agc2(phi_T):
                return Agc(phi_T,penalty,R,B_disc,P_disc,M_n,reverse_heat_solver,steps_t,heat_solver,dt)
            Abis = sp.linalg.LinearOperator((proj*(R-1),proj*(R-1)), Agc2, dtype=np.float64)#transformation de la matrice de propagation en operateur
            phi_T, info = scipy.sparse.linalg.cg(Abis,-y_T,x0=np.zeros(proj*(R-1)),tol=tolerance,callback=callback)# minimisation de la fonctionnelle avec le gradient conjugue
            ctrl = reverse_heat(P_disc.T.dot(phi_T.reshape((-1,1))),M_n,reverse_heat_solver,steps_t)# calcul du controle
            y_sol = heat(y0,B_disc.T.dot(ctrl),M_n,heat_solver,steps_t,B_disc,dt)#resolution de y avec le controle
            ctrl=B_disc.T.dot(ctrl)#calcul du controle avec l indicatrice
            norm_ctrl = np.linalg.norm(ctrl[:,:-1])**2*dx_float*dt#calcul de la norme du controle
            dual =0.5*norm_ctrl+ 0.5*np.linalg.norm(phi_T)**2*dx_float/penalty + np.dot(phi_T, y_T)*dx_float#minimum de la fonctionnelle duale
            normyT=float(np.array(sum(y_sol[:proj*(R-1),-1]**2*dx_float)))#calcul de la norme de la solution a l instant T
            Ju=0.5*norm_ctrl+0.5*normyT*penalty#calcul de la fonctionnelle
            matrice_J[i_R,i_M]=Ju
            matrice_dual[i_R,i_M]=dual
            matrice_yT[i_R,i_M]=normyT
            matrice_u[i_R,i_M]=norm_ctrl
            matrice_grad[i_R,i_M]=itns-np.sum(matrice_grad)


    ############################################################################################################################
    ### affichage de l energie primal energie duale, etat final, ud controle et du nombre d iterations du grandient conjugue ###
    ############################################################################################################################
    ordre=1/R_list
    Ju=matrice_J[:,0].T# valeur du minimum de la fonctionnelle
    yT=np.sqrt(matrice_yT[:,0].T)# norme de la solution a l instant final
    u=np.sqrt(matrice_u[:,0].T)#norme du controle
    grad=matrice_grad[:,0].T#nombre d iterations du grandient conjugue
    print("penalty = h^{0} = {1}^{0} = {2}".format(expo, float(R), penalty))
    print("Energie")
    print(matrice_J)
    print("Energie dual")
    print(matrice_dual)
    print("Etat final")
    print(np.sqrt(matrice_yT))
    print("Controle")
    print(np.sqrt(matrice_u))
    print("Nombre iterations gradient conjugue")
    print(matrice_grad)

    ##############################################
    ### choix du type d affichage (Cf plot.py) ###
    ##############################################
    aff=input('Type affichage :\n1)ordre\n2)Affichage courbe\n3)affichage courbe separe\nChoix : ')
    if aff==1:
        plot_ordre(ordre,Ju,yT,u)#affichage des differents ordres

    if aff==2:
        plot_graph(Temps,M,R,y_sol,ctrl,n,window_max,window_min)#affichage des graphs dans la meme fenetre

    if aff==3:
        plot_aff_sep(Temps,M,R,y_sol,ctrl,n,window_max,window_min)#affichage des graphs dans des fenetres separees
    return

resolution()

