import matplotlib.pyplot as plt
import numpy as np
from . import vasp
from .monkeypatch import monkeypatch_class

@monkeypatch_class(vasp.Vasp)
def plot_electrostatic_potential(self,
                                 x=None,
                                 y=None,
                                 label=None,
                                 color=None,
                                 show=False,
                                 legend=False,
                                 ref_fermi=False):
    atoms = self.get_atoms()
    X, Y, Z, lp = self.get_local_potential()
    nx, ny, nz = lp.shape

    uc = atoms.get_cell()

    if x is None and y is None:
        avg_ep = [np.average(lp[:, :, z]) for z in range(nz)]
        ylabel = 'x-y averaged electrostatic potential'
    elif y is None and x is not None:
        # Assume xaxis goes from 0 to uc
        f = x / uc[0][0]
        x = int(np.floor(nx * f))
        avg_ep = [np.average(lp[x, :, z]) for z in range(nz)]
        ylabel = 'y averaged electrostatic potential'
        
    elif x is None and y is not None:
        f = y / uc[1][1]
        y = int(np.floor(nx * f))        
        avg_ep = [np.average(lp[:, y, z]) for z in range(nz)]
        ylabel = 'x averaged electrostatic potential'                
    else:
        avg_ep = [np.average(lp[x, y, z]) for z in range(nz)]
        ylabel = 'electrostatic potential'        
    xaxis = np.linspace(0, uc[2][2], nz)
    ef = self.get_fermi_level()

    kwargs = {}
    if label is not None:
        kwargs['label'] = label
        
    if color is not None:
        kwargs['color'] = color

    if ref_fermi:
        plt.plot(xaxis, np.array(avg_ep) - ef, **kwargs)
        ylabel += ' $-E_{fermi}$'
    else:
        plt.plot(xaxis, np.array(avg_ep), **kwargs)
        kwargs.pop('label', None)
        plt.plot([min(xaxis), max(xaxis)], [ef, ef], '--', **kwargs)

    if legend:
        plt.legend(loc='best')
    plt.xlabel('z ($\AA$)')
    plt.ylabel(ylabel)
    
    if show:
        plt.show()
    return
