"""Calculate bandstructure diagrams in jasp"""
from .vasp import Vasp
from .monkeypatch import monkeypatch_class
from ase.dft import DOS

import os, subprocess
import numpy as np
# subprocess is used in my "hack" for get_bandstructure_v02

# turn off if in the queue.
# # noqa means for pep8 to ignore the line.
if 'PBS_O_WORKDIR' in os.environ:
    import matplotlib  # noqa
    matplotlib.use('Agg')

import matplotlib.pyplot as plt  # noqa


@monkeypatch_class(Vasp)
def get_bandstructure(self,
                      kpts_path=None,
                      kpts_nintersections=10,
                      show=False):
    """Calculate band structure along :param kpts_path:
    :param list kpts_path: list of tuples of (label, k-point) to
      calculate path on.
    :param int kpts_nintersections: is the number of points between
      points in band structures. More makes the bands smoother.

    returns (npoints, band_energies, fighandle)

    """
    self.update()
    self.stop_if(self.potential_energy is None)

    kpts = [k[1] for k in kpts_path]
    labels = [k[0] for k in kpts_path]

    dos = DOS(self, width=0.2)
    d = dos.get_dos()
    e = dos.get_energies()

    ef = self.get_fermi_level()

    # run in non-selfconsistent directory

    wd = os.path.join(self.directory, 'bandstructure')

    if not os.path.exists(wd):
        self.clone(wd)

        calc = Vasp(wd)
        calc.set(kpts=kpts,
                 kpts_nintersections=kpts_nintersections,
                 reciprocal=True,
                 nsw=0,  # no ionic updates required
                 isif=None,
                 ibrion=None,
                 icharg=11)

        calc.update()

        if calc.potential_energy is None:
            return None, None, None

    else: # I don't think this will work unless the calculation is complete!

        fig = plt.figure()
        with open(os.path.join(wd, 'EIGENVAL')) as f:
            # skip 5 lines
            f.readline()
            f.readline()
            f.readline()
            f.readline()
            f.readline()
            unknown, npoints, nbands = [int(x) for x in f.readline().split()]

            f.readline()  # skip line
            
            band_energies = [[] for i in range(nbands)]

            for i in range(npoints):
                x, y, z, weight = [float(x) for x in f.readline().split()]

                for j in range(nbands):
                    fields = f.readline().split()
                    id, energy = int(fields[0]), float(fields[1])
                    band_energies[id - 1].append(energy)
                f.readline()  # skip line

        ax1 = plt.subplot(121)
        for i in range(nbands):
            plt.plot(list(range(npoints)), np.array(band_energies[i]) - ef)

        ax = plt.gca()
        ax.set_xticks([])  # no tick marks
        plt.xlabel('k-vector')
        plt.ylabel('Energy (eV)')

        nticks = len(labels) / 2 + 1
        ax.set_xticks(np.linspace(0, npoints, nticks))
        L = []
        L.append(labels[0])
        for i in range(2, len(labels)):
            if i % 2 == 0:
                L.append(labels[i])
            else:
                pass
        L.append(labels[-1])
        ax.set_xticklabels(L)
        plt.axhline(0, c='r')

        plt.subplot(122, sharey=ax1)
        plt.plot(d, e)
        plt.axhline(0, c='r')
        plt.ylabel('energy (eV)')
        plt.xlabel('DOS')

        plt.subplots_adjust(wspace=0.26)
        if show:
            plt.show()
        return (npoints, band_energies, fig)


@monkeypatch_class(Vasp)
def get_bandstructure_v02(self,
                      kpts_path=None,
                      kpts_nintersections=None,
                      show=False):
    """Calculate a hybrid band structure along :param kpts_path:
    :param list kpts_path: list of tuples of (label, k-point) to
      calculate path on.
    :param int kpts_nintersections: is the number of points between
      points in band structures. More makes the bands smoother.

    returns (npoints, band_energies, fighandle)

    This is designed to provide a function to calculate the band structure
    according to the procedure 2 provided by the VASP Wiki:
    https://www.vasp.at/wiki/index.php/Si_bandstructure#Procedure_2:_0-weight_.28Fake.29_SC_procedure_.28PBE_.26_Hybrids.29
    """

    print('Commencing the hybrid caluclation procedure.')

    """
    Our first job will be to copy the working directory to a subdirectory
    """

    self.update()
    self.stop_if(self.potential_energy is None)

    kpts = [k[1] for k in kpts_path]
    labels = [k[0] for k in kpts_path]

    # Calculate the zero-weight k-points
    kx = np.linspace(kpts[0][0], kpts[1][0], kpts_nintersections)
    ky = np.linspace(kpts[0][1], kpts[1][1], kpts_nintersections)
    kz = np.linspace(kpts[0][2], kpts[1][2], kpts_nintersections)

    dos = DOS(self, width=0.2)
    d = dos.get_dos()
    e = dos.get_energies()

    ef = self.get_fermi_level()

    '''
    By now, our initial DFT calculation should be complete. It should have
    produced an IBZKPT file.
    '''
    wd = os.path.join(self.directory, 'bandstructure')

    if not os.path.exists(wd):
        # I think: clone works if no subsequent calcs have been run
        self.clone(wd)
                      
        # I think: Vasp() works if no subsequent calcs have been run
        calc = Vasp(wd)
    
        '''
        Now we create a file "KPOINTS_HSE_bands" update the KPOINTS file by doing the following:
        (1) Use the IBZKPT file for the KPOINTS file
        (2) Add desired zero-weight k-points
        (3) Update the number of total k-points
        
        Here's a link to a video of a seminar where the additional k-points are
        discussed: https://youtu.be/OQhRYzWAGfk?t=2389
        - These points correspond to positions along the k-point path we desire
        '''
        IBZKPT = os.path.join( calc.directory, 'IBZKPT' ) 
        with open(IBZKPT, 'rt') as infile:
            infile.readline()
            nk_old = int(infile.readline().split()[0])
            # print(nk_old)
            print(f'Found {nk_old} k-points in IBZKPT.')
            print(f'There are {kpts_nintersections} additional zero-weight k-points.')
            print('Reading the original k-points...')
            original_k_lines = []
            for idx in range(0, nk_old+1):
                original_k_lines.append(infile.readline())
            # print(': {0}'.format(original_k_lines[idx])) 

        total_k_points = nk_old + kpts_nintersections
        
        print(f'There are a total of {total_k_points} k-points.')

        # Make lines for the original k-points
        HSE_lines = ['Explicit k-point list\n', ' '*6 + f'{total_k_points}\n']
        for line in original_k_lines:
            HSE_lines.append(line)

        # Make lines for the new k points
        for idx in range(0, kpts_nintersections):
            line_str = '{0:15.12f} {1:15.12f} {2:15.12f} 0.0\n'.format(kx[idx],
                                                                       ky[idx],
                                                                       kz[idx])
            HSE_lines.append(line_str)

        # for line in HSE_lines:
        #    print(line)
        
        # Write the 'KPOINTS_HSE_bands' file
        tgt = os.path.join(calc.directory, 'KPOINTS_HSE_bands')
        with open(tgt, 'w') as outfile:
            for line in HSE_lines:
                outfile.write(line)

        """
        Refer to:
        https://www.vasp.at/wiki/index.php/Si_bandstructure#Hybrid_calculation_using_a_suitably_modified_KPOINTS_file
        
        Here, the VASP example suggests adding the following to the INCAR file:
        ## HSE
        LHFCALC = .TRUE. ; HFSCREEN = 0.2 ; AEXX = 0.25
        ALGO = D ; TIME = 0.4 ; LDIAG = .TRUE.
        
        Notes: 
    
        validate.py does not appear to have a case for LHFCALC, HFSCREEN, or
        AEXX. This might necessitate (for now), the simple and dirty hack of
        simply appending text to the INCAR file. This, then, means another
        dirty hack of running VASP in the directory in question apart from
        ASE.

        LHFCALC (https://www.vasp.at/wiki/index.php/LHFCALC)
           specifies whether HF/DFT hybrid functional-type calculations are
           performed.
           Default: False

        HFSCREEN (https://www.vasp.at/wiki/index.php/HFSCREEN)
           specifies the range-separation parameter in range separated hybrid
           functionals.

        AEXX (https://www.vasp.at/wiki/index.php/AEXX)
           Default: AEXX = 0.25 if LHFCALC=.TRUE., 0 otherwise

        The ALGO (https://www.vasp.at/wiki/index.php/ALGO) Wiki page does not
           list a "D" property. There is a "Damped" setting, however.

        The LDIAG (https://www.vasp.at/wiki/index.php/LDIAG) page says that
           TRUE is the default value, so I'm going to refrain from adding this.
        """
        HSE_settings = 'LHFCALC = .TRUE. ; HFSCREEN = 0.2; ALGO = D\nICHARG = 11'

        tgt = os.path.join(calc.directory, 'INCAR')
        with open(tgt, 'a') as infile:
            infile.write(HSE_settings)

        # Create and run a subprocess that invokes 
        CWD = os.getcwd()
        VASPDIR = calc.directory
        from .vasprc import VASPRC
        module = VASPRC['module']
        script = """#!/bin/bash
module load {module}

source ~/.bashrc # added by EPB - slight issue with "module load intel"

cd {CWD}
cd {VASPDIR}  # this is the vasp directory

runvasp.py     # this is the vasp command
#end""".format(**locals())

        jobname = 'HSE_BandCalc'
        cmdlist = ['{0}'.format(VASPRC['queue.command'])]
        cmdlist += ['-o', VASPDIR]
        cmdlist += [option for option in VASPRC['queue.options'].split()]
        cmdlist += ['-N', '{0}'.format(jobname),
                    '-l', 'walltime={0}'.format(VASPRC['queue.walltime']),
                    '-l', 'nodes={0}:ppn={1}'.format(VASPRC['queue.nodes'],
                                                     VASPRC['queue.ppn']),
                    '-l', 'mem={0}'.format(VASPRC['queue.mem']),
                    '-M', VASPRC['user.email']]
        p = subprocess.Popen(cmdlist,
                             stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE,
                             universal_newlines=True)

        out, err = p.communicate(script)         

    """
    calc.set(nsw=0,  # no ionic updates required
       isif=None,
       ibrion=None,
       icharg=11,
                 )
    """
    return None, None, None
    

    '''
    # run in non-selfconsistent directory

    wd = os.path.join(self.directory, 'bandstructure')

    if not os.path.exists(wd):
        self.clone(wd)

        calc = Vasp(wd)
        calc.set(kpts=kpts,
                 kpts_nintersections=kpts_nintersections,
                 reciprocal=True,
                 nsw=0,  # no ionic updates required
                 isif=None,
                 ibrion=None,
                 icharg=11)

        calc.update()

        if calc.potential_energy is None:
            return None, None, None

    else: # I don't think this will work unless the calculation is complete!

        fig = plt.figure()
        with open(os.path.join(wd, 'EIGENVAL')) as f:
            # skip 5 lines
            f.readline()
            f.readline()
            f.readline()
            f.readline()
            f.readline()
            unknown, npoints, nbands = [int(x) for x in f.readline().split()]

            f.readline()  # skip line
            
            band_energies = [[] for i in range(nbands)]

            for i in range(npoints):
                x, y, z, weight = [float(x) for x in f.readline().split()]

                for j in range(nbands):
                    fields = f.readline().split()
                    id, energy = int(fields[0]), float(fields[1])
                    band_energies[id - 1].append(energy)
                f.readline()  # skip line

        ax1 = plt.subplot(121)
        for i in range(nbands):
            plt.plot(list(range(npoints)), np.array(band_energies[i]) - ef)

        ax = plt.gca()
        ax.set_xticks([])  # no tick marks
        plt.xlabel('k-vector')
        plt.ylabel('Energy (eV)')

        nticks = len(labels) / 2 + 1
        ax.set_xticks(np.linspace(0, npoints, nticks))
        L = []
        L.append(labels[0])
        for i in range(2, len(labels)):
            if i % 2 == 0:
                L.append(labels[i])
            else:
                pass
        L.append(labels[-1])
        ax.set_xticklabels(L)
        plt.axhline(0, c='r')

        plt.subplot(122, sharey=ax1)
        plt.plot(d, e)
        plt.axhline(0, c='r')
        plt.ylabel('energy (eV)')
        plt.xlabel('DOS')

        plt.subplots_adjust(wspace=0.26)
        if show:
            plt.show()
        return (npoints, band_energies, fig)
    '''
