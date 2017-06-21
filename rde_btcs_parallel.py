"""
Approximation numérique en parallèle de l’équation de réaction-diffusion avec condition
de Dirichlet avec la méthode BTCS (Backward-Time Central-Space).
L'équation est résolu avec et sans le terme de réaction.
La matrice de passage d’un pas de temps à un autre est dispersée avec la fonction scipy.sparse.diags .
Les résultats sont visualisés sous forme de film mp4

"""

# -*- coding: utf-8 -*-

from __future__ import division
from mpi4py import MPI
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import os
import sys
import time

def Sol_exa(t,X,xmax) :
  """
  Calcul la valeur exacte de la solution à un temps t si elle
  est connue et rentrée par l'utilisateur.

  Arguments:
    t (float) : Temps.
    X (array) : Points aux quels la solution va être calculée
    xmax (float) : Borne supérieur de l'intervalle de définition de l'espace

  Retour:
    array : Valeurs de la solution au temps t donné

  """
  
  #Solution exacte pour CI sin(pi*x/x_max) 
  return np.sin(np.pi*X/xmax)*np.exp(-np.pi**2*t/xmax**2)

def Grap(space,X1,X2,t,x0,xmax,ymin,ymax,i,rank) :
  """
  Créer le graphe de la solution trouvée à l'instant t et le sauvegarde.

  Arguments:
    space (array(1,n)) : Intervalle [x0,xmax]
    X1,X2 (array(1,n)) : Solutions approchées trouvées à l'instant t, doivent être de même longueur que space
    t (float) : Temps
    x0,xmax (float) : Borne inférieur et supérieur de l'intervalle de définition de l'espace
    ymin,ymax (float) : Valeur minimale et maximale de toute les solutions
    i (int) : Sera inclut dans le nom du fichier du graphe
    rank (int) : Rang du coeur qui fait tourner le programme
  """
  
  SolExa = Sol_exa(t,space,xmax)
  plt.plot(space,SolExa,label="Solution exacte sans réaction")
  plt.plot(space,X1,label="Approximation sans réaction")
  plt.plot(space,X2,label="Approximation avec réaction u(1-u)")
  plt.text(x0 + (xmax-x0)/10,ymax, "t=" + str("%.3f"% t) + "s", horizontalalignment = 'center', verticalalignment = 'center')
  plt.title ("Simulation de l'equation de réaction-diffusion avec condition de Dirichlet")
  plt.xlabel ( 'Espace')
  plt.ylabel ( 'Concentration')
  plt.ylim(ymin,ymax + 0.1*abs(ymax))
  plt.xlim(x0,xmax)
  plt.legend()
  plt.savefig('edp' + str(rank) + "0"*(4-len(str(i))) + str(i) + '.png', transparent=False)
  plt.clf()

def Film(matrix1,matrix2,pas_t,times,space,x0,xmax,rank,size) :
  """
  Créer un film qui permet de visualiser les résultats obtenue.

  Arguments:
    matrix1,matrix2 (array(n,m)) : Solutions approchées trouvées à tout les instants t
    pas_t (int) : Nombre de points utilisés pour la subdivision du temps
    times (array(1,n)) : Intervalle du temps
    space (array(1,m)) : Intervalle [x0,xmax]
    x0,xmax (float) : Borne inférieur et supérieur de l'intervalle de définition de l'espace
    rank (int) : Rang du coeur qui fait tourner le programme
    size (int) : Nombre de coeurs qui font tourner le programme en même temps
  """
  #Calcul le minimum et le maximum de la solution trouvé 
  y_min = min(matrix1.min(),matrix2.min())
  y_max = max(matrix1.max(),matrix2.max())
  
  #Crée les plots pour chaque pas de temps 
  for i in range((rank*pas_t)//size,((rank + 1)*pas_t)//size) :
    Grap(space,matrix1[i],matrix2[i],times[i],x0,xmax,y_min,y_max,i,rank)
  
  #Converti les images des plots en .mp4
  cmd = 'convert edp' + str(rank) + '*.png film-edp-be' + str(rank) + '.mp4'
  os.system(cmd)
  
  #Supprime les images
  for i in range((rank*pas_t)//size,((rank + 1)*pas_t)//size) :
    os.remove('edp' + str(rank) + "0"*(4-len(str(i))) + str(i) + '.png') 
  
  comm.allgather(1) #Permet de lancer l'étape suivante quand tous les coeurs ont fini la précédente
  
  if rank==0 :
    #Assemble les .mp4 obtenues avec chaque coeur
    cmd1 = 'convert film-edp-be*.mp4 edp-reac-be.mp4'
    os.system(cmd1)
  
    #Supprime les .mp4 intermédiaires
    for i in range(size) :
      os.remove('film-edp-be' + str(i) + '.mp4')


def CI(x,xmax) :
  """
  Fonction qui calcule les valeurs pour la condition initiale

  Arguments:
    x (array(1,n)) : Intervalle ou sous-intervalle de l'espace
    xmax (float) : Borne supérieur de l'intervalle de définition de l'espace
    
  Retour:
    array(1,n) : Solutions obtenues
  """
  return np.sin(np.pi*x/xmax)
  
def Reac(u) :
  """
  Fonction qui calcule les valeurs pour la réaction

  Arguments:
    u (array(1,n)) : Solutions à un pas de temps donné
    
  Retour:
    array(1,n) : Valeurs de la réaction
  """
  return u*(1-u)

#Prise du temps initial
start_time = time.time()

#Initiation du parallelisme
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
  
#Définition du temps
t0 = 0
tmax = 1
pas_t = 500
delta_t = (tmax-t0)/(pas_t)
times = np.linspace(t0,tmax,pas_t + 1)

#Définition de l'espace
x0 = 0
xmax = 2
pas_x = 19

#Pour la suite du programme, il faut que (pas_x + 1) soit divisible par size, sinon il y a un problème dans la création des matrices
if (pas_x + 1)%size != 0 :
   pas_x = pas_x + (size-(pas_x + 1)%size)
pas_x_s = pas_x//(size)
delta_x = (xmax-x0)/(pas_x)
space = np.linspace(x0,xmax,pas_x + 1)

#Définition des paramètres
k = 1
r = k*delta_t/delta_x**2

#Creation de la matrice qui va contenir les résultats
final_matrix1 = np.zeros((pas_t + 1,pas_x_s + 4))
final_matrix2 = np.zeros((pas_t + 1,pas_x_s + 4))

#Condition initiales
matrix1 = CI(space[rank*pas_x_s:rank*pas_x_s + pas_x_s + 4],xmax)
matrix2 = CI(space[rank*pas_x_s:rank*pas_x_s + pas_x_s + 4],xmax)
final_matrix1[0][:] = matrix1[:]
final_matrix2[0][:] = matrix2[:]

b1 = np.zeros(pas_x_s + 4)
b2 = np.zeros(pas_x_s + 4)

#Création des matrices qui vont permettre de disperser la matrice A
main  = np.zeros(pas_x_s + 4)
lower = np.zeros(pas_x_s + 3)
upper = np.zeros(pas_x_s + 3)
main[:] = 1 + 2*r
lower[:] = -r 
upper[:] = -r  

# Insertion des conditions aux bords
if rank == 0 :
  main[0] = 1
  upper[0] = 0

if rank == size-1 :
  main[pas_x_s + 3] = 1
  lower[pas_x_s + 2] = 0

#Dispersion de la matrice A
A = scipy.sparse.diags(
    diagonals = [main, lower, upper],
    offsets = [0, -1, 1], shape = (pas_x_s + 4, pas_x_s + 4),
    format = 'csr')
    
#print(rank,A.todense(),"\n")

#Calcul de la solution en parallèle avec condition de Dirichlet
for n in range(0, pas_t) :
  if 0 < rank :
  # Envoie matrix[2] et matrix[3] à rank-1
    comm.send(matrix1[2], dest = rank-1, tag = 1)
    comm.send(matrix1[3], dest = rank-1, tag = 2)
    comm.send(matrix2[2], dest = rank-1, tag = 3)
    comm.send(matrix2[3], dest = rank-1, tag = 4)

  # Recoit matrix[pas_x_s + 2] et matrix[pas_x_s + 3] de rank + 1
  if rank < size-1 :
    matrix1[pas_x_s + 2] = comm.recv(source = rank + 1, tag = 1)
    matrix1[pas_x_s + 3] = comm.recv(source = rank + 1, tag = 2)
    matrix2[pas_x_s + 2] = comm.recv(source = rank + 1, tag = 3)
    matrix2[pas_x_s + 3] = comm.recv(source = rank + 1, tag = 4)

  # Envoie matrix[pas_x_s] et matrix[pas_x_s+1] à rank + 1
  if rank < size-1 :
    comm.send(matrix1[pas_x_s], dest = rank + 1, tag = 5)
    comm.send(matrix1[pas_x_s + 1], dest = rank + 1, tag = 6)
    comm.send(matrix2[pas_x_s], dest = rank + 1, tag = 7)
    comm.send(matrix2[pas_x_s + 1], dest = rank + 1, tag = 8)

  # Recoit matrix[0] et matrix[1] de rank-1
  if 0 < rank :
    matrix1[0] = comm.recv(source = rank-1, tag = 5)
    matrix1[1] = comm.recv(source = rank-1, tag = 6)
    matrix2[0] = comm.recv(source = rank-1, tag = 7)
    matrix2[1] = comm.recv(source = rank-1, tag = 8)
    
  #Mise à jour des valeurs de la matrice b
  b1[:] = matrix1[:]
  b2[:] = matrix2[:] + delta_t*Reac(matrix2[:])
  
  #Application des conditions aux bords, ici Dirichlet pour rappel
  if rank == 0 :
    b1[0] = 0
    b2[0] = 0
  if rank == size-1 :
    b1[pas_x_s + 3] = 0
    b2[pas_x_s + 3] = 0
    
  #Calcul des nouvelles valeurs au pas de temps suivant
  matrix1[:] = scipy.sparse.linalg.spsolve(A, b1)
  matrix2[:] = scipy.sparse.linalg.spsolve(A, b2)
  
  #Stockage des valeurs obtenues dans la matrice finale
  final_matrix1[n][:] = matrix1[:]
  final_matrix2[n][:] = matrix2[:]
  
#Garde seulement les solutions nécessaires (suppression des chevauchements entre les différentes matrices)
if rank == 0 :
  final_matrix1b = final_matrix1[:,0:pas_x_s + 2]
  final_matrix2b = final_matrix2[:,0:pas_x_s + 2]
elif rank == size-1 :
  final_matrix1b = final_matrix1[:,2:pas_x_s + 4]
  final_matrix2b = final_matrix2[:,2:pas_x_s + 4]
else :
  final_matrix1b = final_matrix1[:,2:pas_x_s + 2]
  final_matrix2b = final_matrix2[:,2:pas_x_s + 2]

#Collecte et assemble les matrices de chaque coeur
gathered_list1 = comm.allgather(final_matrix1b)
gathered_list2 = comm.allgather(final_matrix2b)
gathered_matrix1 = np.concatenate(gathered_list1,axis = 1)  
gathered_matrix2 = np.concatenate(gathered_list2,axis = 1)  

#Création du .mp4 pour visualer la simulation
Film(gathered_matrix1,gathered_matrix2,pas_t + 1,times,space,x0,xmax,rank,size)

#Prise du temps final et mesure du temps écoulé
end_time = time.time()
elapsed_time = end_time-start_time
print("\nL'exécution du programme a pris", round(elapsed_time,4),"s sur le coeur ",rank)
