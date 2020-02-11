######################################################
### fichiers permettant les differents affichage : ###
######################################################





############################################
### differentes librairies necessaires : ###
############################################
from __future__ import division
from matplotlib import pyplot as plt
from math import *
import numpy as np
from mpl_toolkits.mplot3d import Axes3D#affichage des courbe 3D
from matplotlib import cm


import matplotlib
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.unicode'] = True


###########################################################################################################
#### afffichage des courbes d ordre pour l'instant final, le controle et la valeur de la fonctionnelle ####
###########################################################################################################
def plot_ordre(ordre,Ju,yT,u):
	ordre_array=np.vstack([np.log(np.array(ordre)),np.ones(len(ordre))]).T
	yT_array=np.array(np.log(yT))
	u_array=np.array(np.log(u))
	Ju_array=np.array(np.log(Ju))

        #calcul des pentes
	coeffyT, b = np.linalg.lstsq(ordre_array,yT_array)[0]
	coeffu, b = np.linalg.lstsq(ordre_array,u_array)[0]
	coeffJu, b = np.linalg.lstsq(ordre_array,Ju_array)[0]
	
        #affichage des pentes
        print('coefficient directeur de yT, u, Ju :')
	print(coeffyT,coeffu,coeffJu)
	
	fig=plt.figure(1)
	plt.loglog(ordre,yT,'-v',basex=10,basey=10,label='$||y_{{\epsilon}}^{{h,\delta t}}(T)||_{{E_h}} $   $(slope = {0:.2})$'.format(coeffyT))
        plt.loglog(ordre,u,'-^',basex=10,basey=10,label='$||u_{{\epsilon}}^{{h,\delta t}}||_{{L^2_{{\delta t}}(0,T;U_h)}} $   $(slope = {0:.2})$'.format(coeffu))
	plt.loglog(ordre,Ju,'-ro',basex=10,basey=10,label='$\inf_{{u^{{h,\delta t}}}}\in  L^2_{{\delta t}}(0,T;U_h)}} F_{{\epsilon}}^{{h,\delta t}}(u^{{h,\delta t}}) $   $(slope = {0:.2})$'.format(coeffJu))
	
	plt.xlabel('h')
	art=[]
	lgd = plt.legend(loc='upper center',bbox_to_anchor=(0.5,1.3))
	art.append(lgd)

        #sauvegarde du fichier
	fig.savefig('ordre.eps',additional_artists=art,bbox_inches='tight', format='eps', dpi=1000)
        plt.show()
        return

#################################################################################
#### afffichage des graph de la solution et du controle dans la meme fenetre ####
#################################################################################
def plot_graph(Temps,M,R,y_sol,ctrl,n,window_max,window_min):
    fig=plt.figure(1)
    dim=ctrl.shape


    X, Y = np.mgrid[:dim[0],:dim[1]]
    Y=Y*Temps/M
    X=2*np.pi*X/R


    ax = fig.add_subplot(2,2,1, projection='3d')
    surf = ax.plot_surface(Y,X,y_sol[:R-1,:], rstride = 1, cstride = 1, cmap = plt.get_cmap('jet'),linewidth = 0, antialiased = False)
    plt.xlabel('t')
    plt.ylabel('x')
    plt.title('y')


    ax = fig.add_subplot(2,2,2, projection='3d')
    surf = ax.plot_surface(Y,X,y_sol[R-1:2*R-2,:], rstride = 1, cstride = 1, cmap = plt.get_cmap('jet'),linewidth = 0, antialiased = False)
    plt.xlabel('t')
    plt.ylabel('x')
    plt.title('z')

    if n==3:
        ax = fig.add_subplot(2,2,3, projection='3d')
        surf = ax.plot_surface(Y,X,y_sol[2*R-2:3*R-3,:], rstride = 1, cstride = 1, cmap = plt.get_cmap('jet'),linewidth = 0, antialiased = False)
        plt.xlabel('t')
        plt.ylabel('x')
        plt.title('w')

    ax = fig.add_subplot(2,2,4, projection='3d')
    surf = ax.plot_surface(Y,X,ctrl, rstride = 1, cstride = 1, cmap = plt.get_cmap('jet'),linewidth = 0, antialiased = False)
    plt.xlabel('t')
    plt.ylabel('x')
    plt.title('u')

    art=[]
    lgd = plt.legend(loc='upper center',bbox_to_anchor=(0.5,1.3))
    art.append(lgd)
    fig.savefig('graph.eps',additional_artists=art,bbox_inches='tight', format='eps', dpi=1000)
    plt.show()
    return


#########################################################################################
#### afffichage des graph de la solution et du controle dans des fenetre differentes ####
#########################################################################################
def plot_aff_sep(Temps,M,R,y_sol,ctrl,n,window_max,window_min):
    dim=ctrl.shape


    X, Y = np.mgrid[:dim[0],:dim[1]]
    Y=Y*Temps/M
    X=2*np.pi*X/R

    fig1=plt.figure(1)
    ax = fig1.add_subplot(1,1,1, projection='3d')
    surf = ax.plot_surface(Y,X,y_sol[:R-1,:], rstride = 1, cstride = 1, cmap = plt.get_cmap('jet'),linewidth = 0, antialiased = False)
    ax.set_xlabel(r'$t$')
    ax.set_ylabel(r'$x$')
    ax.set_zlabel(r'$y_1$')
    art=[]
    lgd = plt.legend(loc='upper center',bbox_to_anchor=(0.5,1.3))
    art.append(lgd)
    fig1.savefig('graph_y.eps',additional_artists=art,bbox_inches='tight', format='eps', dpi=1000)
    plt.show()

    fig2=plt.figure(2)
    ax = fig2.add_subplot(1,1,1, projection='3d')
    surf = ax.plot_surface(Y,X,y_sol[R-1:2*R-2,:], rstride = 1, cstride = 1, cmap = plt.get_cmap('jet'),linewidth = 0, antialiased = False)
    ax.set_xlabel(r'$t$')
    ax.set_ylabel(r'$x$')
    ax.set_zlabel(r'$y_2$')
    art=[]
    lgd = plt.legend(loc='upper center',bbox_to_anchor=(0.5,1.3))
    art.append(lgd)
    fig2.savefig('graph_z.eps',additional_artists=art,bbox_inches='tight', format='eps', dpi=1000)
    plt.show()

    if n==3:
        ax = fig.add_subplot(2,2,3, projection='3d')
        surf = ax.plot_surface(Y,X,y_sol[2*R-2:3*R-3,:], rstride = 1, cstride = 1, cmap = plt.get_cmap('jet'),linewidth = 0, antialiased = False)
        ax.set_xlabel('x')
        ax.set_ylabel('t')
        ax.set_zlabel('w')

    fig3=plt.figure(3)
    ax = fig3.add_subplot(1,1,1, projection='3d')
    surf = ax.plot_surface(Y,X,ctrl, rstride = 1, cstride = 1, cmap = plt.get_cmap('jet'),linewidth = 0, antialiased = False)
    ax.set_xlabel('t')
    ax.set_ylabel('x')
    ax.set_zlabel('u')
    art=[]
    lgd = plt.legend(loc='upper center',bbox_to_anchor=(0.5,1.3))
    art.append(lgd)
    fig3.savefig('graph_u.eps',additional_artists=art,bbox_inches='tight', format='eps', dpi=1000)
    plt.show()
    return

