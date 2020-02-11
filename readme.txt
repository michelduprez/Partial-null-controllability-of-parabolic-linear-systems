#########################
### probleme etudie : ###
#########################

schema euler implicite en temps et differences/elements finies en espace pour la controlabilite partielle du systeme parabolique:

y=Delta y + Az +B1_{omega}u
y=0 sur le bord
y(0)=y0

Question : existe-t-il un controle u tel que Py(T)=0  ou ||Py(T)||<epsilon ? (P matrice de projection)

fonctionnelle :
J(u) = 0.5*int B^*u^2 + penalite*0.5*int Py(T)^2
ou la penalite depend du pas en espace


##############################
### Gestion des fichiers : ###
##############################

Dans le terminal ou se trouve les fichiers : python prog.py (fichier mère)

Gestion des differents parametres : params.py

Reslution du probleme direct  : solver_diff_fini.py et solver_elts_fini.py 

Affichage des courbes : plot.py
