import matplotlib
matplotlib.use('Agg')
import math
import h5py
import numpy as np
import argparse
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import curve_fit
from scipy.optimize import differential_evolution

# load the command-line parameters
parser = argparse.ArgumentParser( description='Plot the gas slices and projections' )

parser.add_argument( '-s', action='store', required=True,  type=int, dest='idx_start',
                     help='first data index' )
parser.add_argument( '-e', action='store', required=True,  type=int, dest='idx_end',
                     help='last data index' )
args=parser.parse_args()

# take note
print( '\nCommand-line arguments:' )
print( '-------------------------------------------------------------------' )
for t in range( len(sys.argv) ):
   print( str(sys.argv[t]))
print( '' )
print( '-------------------------------------------------------------------\n' )


idx_start   = args.idx_start
idx_end     = args.idx_end

Div = 500 ## Number of data points in your rotation curve

def SearchIndex(x, A, N):
   i = 0
   j = N - 1
   while(i <= j):
      mid = int(i + (j - i)/2)
      if(A[mid] == x):     
         i = mid
         break 
      elif(A[mid] > x): 
         j = mid - 1
      else: i = mid + 1  
   return i  

delta_t = 140.59 ## delta_t between snapshots (Myr)
t = []
for idx in range(idx_start, idx_end+1):
   t.append(idx*delta_t)
t = np.array(t)
h_theory = np.zeros((Div, len(t)))
w = np.zeros((Div, len(t)))
mean_theory = np.zeros((4, len(t)))
i = 0
for idx in range(idx_start, idx_end+1):
   F = np.load('Data_Disk_%06d.npy'%idx)
   H = np.load('Heating_%06d.npz'%idx) 
   r = F[0]/3.08568e+21
   w[:,i] = r*F[4] ## weight = r*Sigma
   h_theory[:,i] = H['b']*(1e9*365*24*3600)/1e10
   i = i + 1
dt = delta_t*1e6*365*24*3600

h_theory[h_theory==-np.inf]=0

num_3 = SearchIndex( 3, r, Div )
num_5 = SearchIndex( 5, r, Div )
num_7 = SearchIndex( 7, r, Div )
num_9 = SearchIndex( 9, r, Div )
num_11 = SearchIndex( 11, r, Div )

for i in range (len(t)):
   mean_theory[0,i] = np.average(h_theory[num_3:num_5,i], weights=w[num_3:num_5])
   mean_theory[1,i] = np.average(h_theory[num_5:num_7,i], weights=w[num_5:num_7])
   mean_theory[2,i] = np.average(h_theory[num_7:num_9,i], weights=w[num_7:num_9])
   mean_theory[3,i] = np.average(h_theory[num_9:num_11,i], weights=w[num_9:num_11])


print('R(kpc)    heating rate (km^2/(s^2*Gyr))')
for i in range(4):
   print('%2d        %2.6f'%(i*2+4, np.mean(mean_theory[i,:]) ))


