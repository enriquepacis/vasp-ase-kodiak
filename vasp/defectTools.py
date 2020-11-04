#author: Ethan Dickey
from ase import Atoms
from collections import Counter
import os
from vasp import POTCAR
 

#Calculates the number of total electrons in the system described by the Atoms object,
# including an optional defect charge (default 0)
#
# **Note: +1 defectCharge = 1 less electron (etc)**
#
#@param atoms the Atoms object which contains the chemical symbols (atoms.get_chemical_symbols()),
#             including the numeric counts of each one
#@param defectCharge an optional charge amount to add/subtract (+charge = -electrons)
#@return the total number of atoms in the system
def calcNElect(atoms, defectCharge = 0):
    print("atoms: ", atoms.get_chemical_symbols())
    #map atoms to unique symbols with counts
    elmCount = dict(Counter(atoms.get_chemical_symbols()))
    print("ElmCount: ", elmCount)

    numElectrons = 0
    #get the path for each element's POTCAR
    pppath = os.environ['VASP_PP_PATH'] + "/potpaw_PBE/"
    for el, count in elmCount.items():
        path = os.path.join(pppath, el + "/POTCAR")
        #pull the zval from the potcar for each element
        z = POTCAR.get_ZVAL(path)
        print("Z for " + el + ": " + str(z))
        numElectrons += z*count
        
    print("numElectrons = ", numElectrons)
    numElectrons += -1*defectCharge
    
    return numElectrons
       
