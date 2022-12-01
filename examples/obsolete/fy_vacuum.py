import sys

sys.path.append("/home/salmon/workspace/fytok/python")
sys.path.append("/home/salmon/workspace/SpDev/SpCommon")
sys.path.append("/home/salmon/workspace/SpDev/SpDB")

import collections
import math

import matplotlib.pyplot as plt
from spdm.data.Collection import Collection
from spdm.util.logger import logger
from spdm.numlib import scipy,np

db = Collection("east+mdsplus:///home/salmon/public_data/~t/",default_tree_name="efit_east")
entry = db.open(shot=55555).entry

vessel_inner_points= np.array([entry.wall.description_2d.vessel.annular.outline_inner.r.__value__(),    
                                          entry.wall.description_2d.vessel.annular.outline_inner.z.__value__()]).transpose([1,0]) 

vessel_outer_points= np.array([entry.wall.description_2d.vessel.annular.outline_outer.r.__value__(),    
                                          entry.wall.description_2d.vessel.annular.outline_outer.z.__value__()]).transpose([1,0])  

limiter_points =  np.array([entry.wall.description_2d[0].limiter.unit[0].outline.r.__value__(),
                                 entry.wall.description_2d[0].limiter.unit[0].outline.z.__value__()]).transpose([1,0]) 
isoflux=[
    (1.93, 0.15,    2.24, 0.52    ),   # isoflux03
    (1.99,0    ,    2.40, 0        ),     # isoflux01
    (1.93,-0.15,    2.24, -0.52    ),   # isoflux09
    (1.72, 0.24,    1.35, 0.45    ),   # isoflux04
    (1.66,0    ,    1.35, 0        ),     # isoflux06
    (1.72,-0.24 ,   1.35, -0.45    ),    # isoflux08

    ]  

COIL=collections.namedtuple("COIL", "label r z current turns")
def psi(r, z, coils):
    def green_function(Rc, Zc, R, Z):
        k2   =  4.0*Rc*R/((Rc+R)*(Rc+R)+(Zc-Z)*(Zc-Z))          
        return   math.sqrt(R*Rc)/(2.0*math.pi)* ((2.0-k2) *scipy.special.ellipk(k2) - 2.0* scipy.special.ellipe(k2))/math.sqrt(k2)  
    return sum([green_function(coil.r, coil.z, r, z)*coil.current for coil in coils])
itime=30000

pf_coils=[] 
rpos=[]
zpos=[]
 
itime=40000
for coil in entry.pf_active.coil: 
  
    # coil=entry.pf_active.coil[idx+1]
    rect=coil.element[0].geometry.rectangle.__value__()
    rpos.append(rect.r-rect.width/2.0)
    rpos.append(rect.r+rect.width/2.0)
    zpos.append(rect.z-rect.height/2.0)
    zpos.append(rect.z+rect.height/2.0)  
    pf_coils.append(COIL(str(coil.name),
     float(rect.r),float(rect.z), coil.current.data.__value__()[itime],int(coil.element[0].turns_with_sign) ))
    # pf_coils.append(COIL(rect.r*1.0,rect.z*1.0,Ic[idx] ) )
    # print((idx,coil.name.__value__()))

rmin=min(rpos)
rmax=max(rpos)
zmin=min(zpos)
zmax=max(zpos)



NX=100
NY=100
X, Y  = np.meshgrid(np.linspace(rmin, rmax, NX),np.linspace(zmin, zmax, NY))
Z=np.ndarray([NX,NY])
for ix in range(NX):
    for iy in range(NY):
        Z[ix,iy]=psi(X[ix,iy],Y[ix,iy],pf_coils)

fg=plt.figure()

plt.gca().add_patch(plt.Polygon(limiter_points , fill=False,closed=True))
plt.gca().add_patch(plt.Polygon(vessel_outer_points , fill=False,closed=True))
plt.gca().add_patch(plt.Polygon(vessel_inner_points , fill=False,closed=True))

for coil  in entry.pf_active.coil:
    rect=coil.element[0].geometry.rectangle.__value__() 
    plt.text(float(rect.r),float(rect.z),str(coil.name))
    plt.gca().add_patch(plt.Rectangle((rect.r-rect.width/2.0, rect.z-rect.height/2.0), rect.width, rect.height, fill=False))
 
plt.contour(X,Y,Z,levels=140,linewidths=0.2)

# plt.contour(
#     entry.equilibrium.time_slice[10].profiles_2d.grid.dim1.__value__(),
#     entry.equilibrium.time_slice[10].profiles_2d.grid.dim2.__value__(),
#     entry.equilibrium.time_slice[10].profiles_2d.psi.__value__(),
#     levels= 20,
#     linewidths=0.5)

for r0,z0,r1,z1 in isoflux:
    plt.plot([r0,r1],[z0,z1])
plt.axis('scaled')
plt.show()
