#Save the following to WannierBandstruct.py:
import numpy as np
from scipy.interpolate import interp1d

#Read the MLWF cell map and Hamiltonian:    
cellMap = np.loadtxt("wannier.mlwfCellMap")[:,0:3].astype(np.int)
Hwannier = np.fromfile("wannier.mlwfH", dtype=np.complex128)
print(Hwannier)
nCells = cellMap.shape[0]
print('nCells', nCells)
nBands = int(np.sqrt(Hwannier.shape[0] / nCells))
print('nBands', nBands)
Hwannier = np.reshape(Hwannier, (nCells,nBands,nBands))

#Read the band structure k-points:
kpointsIn = np.loadtxt('bandstruct.kpoints', skiprows=2, usecols=(1,2,3))
print(kpointsIn)
nKin = kpointsIn.shape[0]
print('nKin', nKin)
#--- Interpolate to a 10x finer k-point path:
xIn = np.arange(nKin)
print('xIn:', len(xIn))
x = 0.1*np.arange(1+10*(nKin-1)) #same range with 10x density
print('x:', len(x))
kpoints = interp1d(xIn, kpointsIn, axis=0)(x)
print('len(kpoints)', len(kpoints))
#for k in range(len(kpoints)):
#    print (k, kpoints[k])
#print(kpoints)
nK = kpoints.shape[0]
print('nK', nK)
#Calculate band structure from MLWF Hamiltonian:
#--- Fourier transform from MLWF to k space:
Hk = np.tensordot(np.exp((2j*np.pi)*np.dot(kpoints,cellMap.T)), Hwannier, axes=1)
print('Hk', len(Hk))
#--- Diagonalize:
Ek,Vk = np.linalg.eigh(Hk)
for line in open('totalE.eigStats'): # Chemical potential in DFT calculation
    if line.startswith('mu'):
        mu = float(line.split()[2])
Ek = (Ek - mu)*27.21
#print(Ek)
#--- Save:
np.savetxt("wannier.eigenvals", Ek)
E_exp =np.array([-1.5596, 0.9174, -2.6055, -2.8624, -0.0183, 0.2936, 0.5688, -0.422, -0.8073, -1.2477, -1.6881, -2.2569, -1.8716,
                 -1.6147, 0.9358, 1.2844, -1.5596, -1.7982, -1.9817, -2.1101, 3.6158, -1.028, -2.3551, -3.2523, -5.0, -6.2523,
                 -6.1869, -4.9346, -2.8411,
                 -2.4579, -2.6075, -2.8131, -2.9907, -3.243, -3.3178, -3.4206, -3.4953,
                 -3.5794, -3.6542, -3.7103, -3.8318, -3.9439, -3.6916, -3.5607, -3.1869,
                 -2.9813, -3.9439, -4.3271, -4.7664, -4.7103, -4.7103, -4.6262, -4.757, -4.757, -4.7757, -4.8972, -5.0093, -5.2991,
                 -5.8598, -6.2056, -6.1589, -6.0561, -6.0841, -6.028, -6.0935, -6.0748, -6.028])
k_exp = np.array([1.5, 1.5, 1.5, 1.5, 1.3264, 1.365, 1.4035, 1.2781, 1.2395, 1.1913, 1.1383, 1.0707, 1.283, 1.3891, 1.5408, 1.6376,
                  1.5306, 1.6987, 1.8618, 2.0605, 3.1, 3.1092, 3.1092, 3.1092, 3.1, 3.1061, 3.529, 3.5015, 3.4892, 3.5474, 3.6394,
                  3.716, 3.9244, 4.0194, 4.1205, 4.2185, 4.3227, 4.4392, 4.5465, 4.7242, 4.5802,
                  4.3381, 3.9642, 3.8294, 3.6669, 3.5811, 3.6118, 3.6547, 3.7619, 3.9182, 4.24,
                  4.3902, 4.5281, 4.6384, 4.7487, 3.6792, 3.8048,
                  3.8508, 4.001, 3.7773, 3.86, 3.9765, 4.0377, 4.2063, 4.479, 4.8008, 4.9])
######Plot
#
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
#x = kpoints
fig, ax = plt.subplots()
##my_scatter = ax.scatter(k_exp, E_exp, s=80, facecolors='dot', edgecolors='b')
my_scatter = ax.scatter(k_exp, E_exp, color="red")

y = np.loadtxt('wannier.eigenvals')
#print(y.shape[2])
y = np.reshape(y, (nK,nBands,1))
#print('y = ',y[100][2])
bands = {i:[] for i in range(nBands)}
z = []
for n in range(nK):
    for i in range(nBands):
        bands[i].extend(y[n][i])
    z.append(n/100)
Ls = ['--', '-.', ':', '-', '--', '--', '-.', ':', '-', '--','--', '-.', ':', '-', '--', ':', '-.']  
#
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'darkorange', 'forestgreen', 'navy','seagreen', 'lime', 'peru', 'darkred', 'yellow']
for i in range(nBands):
    plt.plot(z, bands[i], color=colors[i], linewidth=3, ls=Ls[i])


##
plt.xlabel('k-ponits')
plt.xlim([0, 6.8])
plt.ylim(-7, 7)
plt.ylabel('E - Ef (eV)')
mpl.rcParams.update({'font.size': 12})#, {'font.weight': 'bold'})
plt.xticks([0.0, 1.5, 2.3, 3.1, 4.9, 6.8],
           [r"$\Gamma$", r"$X$", r"$W$", r"$L$", r"$\Gamma$", r"$K$"], fontweight="bold", fontsize=16)
###plt.legend(loc='lower center')
plt.rcParams['axes.labelweight'] = 'bold'
#plt.rcParams['title.labelweight'] = 'bold'
plt.title('Band Structur', fontweight="bold")

#plt.axvline(x=1.6,x=2.4,x=3.2,x=5.0)
xb =[1.5, 2.3, 3.1, 4.9]

for xc in xb:
    plt.axvline(x=xc, ls=':', mfc='k', linewidth=1)
plt.axhline(y = 0, ls='-.', mfc='k', linewidth=1)

plt.savefig('bands.jpg', dpi=1000)


