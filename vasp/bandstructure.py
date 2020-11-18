"""Calculate bandstructure diagrams in jasp"""
from .vasp import Vasp
from .monkeypatch import monkeypatch_class
from ase.dft import DOS

import os, shutil
import subprocess as sp
import numpy as np
# subprocess is used in my "hack" for get_bandstructure_v02

# turn off if in the queue.
# # noqa means for pep8 to ignore the line.
if 'PBS_O_WORKDIR' in os.environ:
    import matplotlib  # noqa
    matplotlib.use('Agg')

import matplotlib.pyplot as plt  # noqa


def makeBandPlot( EIGFILE, labels, ef=0, ylim=None, d=None, e = None,
                  show=False):
    # This is John Kitchin's original band structure visualization
    fig = plt.figure()
    with open( EIGFILE ) as f:
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

    if d is not None:
        ax1 = plt.subplot(121)
    for i in range(nbands):
        plt.plot(list(range(npoints)), np.array(band_energies[i]) - ef)

    ax = plt.gca()
    ax.set_xticks([])  # no tick marks
    if ylim is not None:
        plt.ylim(ylim)
    plt.xlabel('k-vector')
    plt.ylabel('Energy (eV)')

    nticks = int(len(labels) / 2 + 1)
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

    if d is not None and e is not None:
        plt.subplot(122, sharey=ax1)
        plt.plot(d, e)
        plt.axhline(0, c='r')
        plt.ylabel('energy (eV)')
        plt.xlabel('DOS')
        if ylim is not None:
            plt.ylim(ylim)

        plt.subplots_adjust(wspace=0.26)

    if show:
        plt.show()

    return (npoints, band_energies, fig)


@monkeypatch_class(Vasp)
def get_bandstructure(self,
                      kpts_path=None,
                      kpts_nintersections=10,
                      show=False,
                      ylim=None,
                      outdir=None,
                      outfile=None):
    """Calculate band structure along :param kpts_path:
    :param list kpts_path: list of tuples of (label, k-point) to
      calculate path on.
    :param int kpts_nintersections: is the number of points between
      points in band structures. More makes the bands smoother.

    returns (npoints, band_energies, fighandle, Energy gap, Convection band minimum, valance band maximum)

    """
    prevdir = os.getcwd()#store for later use

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
            return None, None, None, None, None, None

    else: # I don't think this will work unless the calculation is complete!


        archiveKpts = os.path.join(wd, 'KPOINTS_old')
        originalKpts = os.path.join(wd, 'KPOINTS')
        if not os.path.exists(archiveKpts): # archive the original KPOINTS file
            shutil.copy(originalKpts, archiveKpts)

        with open(archiveKpts, 'r') as kpts_infile: # get lines of old KPOINTS
            KPtsLines = kpts_infile.readlines()

        # Append labels for high-symmetry points to lines of the KPOINTS file
        reject_chars = '$'
        for idx, label in enumerate(labels):
            newlabel = label
            for ch in reject_chars:
              newlabel = newlabel.replace(ch, '')

            KPtsLines[4+idx] = KPtsLines[4+idx][:-1] + f' ! {newlabel}\n'

        # Write a new version of the k-points file
        with open(originalKpts, 'w') as kpts_outfile:
            for line in KPtsLines:
                kpts_outfile.write(line)

        EIGFILE = os.path.join(wd, 'EIGENVAL')

        npoints, band_energies, fig = makeBandPlot( EIGFILE, labels,
                                                    ef=ef, ylim=ylim,
                                                    d=d, e=e, # DOS data
                                                    show=show)


        # Run sumo-bandplot, if it exists
        check_sumo = sp.run(['which', 'sumo-bandplot'], capture_output=True)
        found_sumo = not check_sumo.stdout == b''
        found_band_data = os.path.exists(os.path.join(wd, 'vasprun.xml'))


        if found_sumo and found_band_data:
            os.chdir(wd)
            sumo_cmd = ['sumo-bandplot']
            run_sumo = sp.run(sumo_cmd, capture_output=True)
            if outfile is not None:
                shutil.copy('band.pdf', outfile)
            else:
                outfile = 'band.pdf'
            if outdir is not None:
                target = os.path.join(outdir, outfile)
                shutil.copy(outfile, target)
            else:
                target = os.path.join(os.getcwd(), outfile)

            Egap, Ecbm, Evbm = read_bandstats()
            
        else:
            print('sumo-bandplot not found. No band plot generated.')
            target = None
            Egap, Ecbm, Evbm = None, None, None

        #reset dir
        os.chdir(prevdir)

        return (npoints, band_energies, fig, Egap, Ecbm, Evbm)


@monkeypatch_class(Vasp)
def get_supercell_bandstructure(self,
                                kpts_path=None,
                                kpts_nintersections=None,
                                supercell_size=[1,1,1],
                                unit_cell=[[1.0,0,0], [0,1.0,0], [0.0,0.0,1.0]],
                                show=False, imagedir=None, imageprefix=None,
                                eref=None, ymin=None, ymax=None,
                                lsorb=False):
    """Calculate band structure along :param kpts_path:
    :param list kpts_path: list of tuples of (label, k-point) to
      calculate path on.
    :param int kpts_nintersections: is the number of points between
      points in band structures. More makes the bands smoother.
    :param list supercell_size: this lists the size of the supercell
      [nx ny nz] as multiples of the unit cell in the ordinal directions
    :param list unit_cell: list of unit cell vectors

    returns (npoints, band_energies, fighandle)

    """
    self.update()
    self.stop_if(self.potential_energy is None)

    # Determine whether the colinear (vasp_std)  or non-colinear (vasp_ncl) is
    #    to be used
    if self.parameters.get('lsorbit'):
        runfile = 'runvasp_ncl.py'
    else:
        runfile = 'runvasp.py'

    '''
       I'm following the procedure for creating a k-path in the supercell
           https://github.com/QijingZheng/VaspBandUnfolding#band-unfolding-1
    '''
    # The tranformation matrix between supercell and primitive cell.
    M = [[1.0*supercell_size[0], 0.0, 0.0],
         [0.0, 1.0*supercell_size[1], 0.0],
         [0.0, 0.0, 1.0*supercell_size[2]]]

    print(M)

    from unfold import find_K_from_k, make_kpath, removeDuplicateKpoints

    # Extract unit-cell k-points from kpts_path
    uc_k_pts_bare = [ k[1] for k in kpts_path]
    nseg = 30 # points per segment
    kpath_uc = make_kpath(uc_k_pts_bare, nseg=nseg) # interpolated uc k-path

    # Map UC (=PC) k-points to SC k-points
    Kpath_sc = []
    for k_uc in kpath_uc:
        K, g = find_K_from_k(k_uc, M)
        Kpath_sc.append(K) # add a weight to K

    Kpath = removeDuplicateKpoints(Kpath_sc) # this is a numpy.ndarray

    # I convert this to a list because the Vasp wrapper prefers a kpts list
    # Also, the Vasp wrapper wants a weight for each K point
    Kpath_list = [list(np.append(K, 1.0)) for K in Kpath]

    # Extract (unit cel) labels from kpts_path
    labels = [k[0] for k in kpts_path]

    dos = DOS(self, width=0.2)
    d = dos.get_dos()
    e = dos.get_energies()

    ef = self.get_fermi_level()
    print(f'Fermi energy: {ef:8.3f} eV')

    # run in non-selfconsistent directory
    wd = os.path.join(self.directory, 'bandstructure')

    if not os.path.exists(wd):
        self.clone(wd)

        calc = Vasp(wd)
        # don't set kpts_nintersections - avoid line mode
        calc.set(kpts=Kpath_list,
                 reciprocal=True,
                 nsw=0,  # no ionic updates required
                 ismear=0, sigma=0.1,
                 isif=None,
                 ibrion=None,
                 icharg=11, lorbit=12)
        
        calc.update() # updates the calculation files
        
        # The manual calculation might be unnecessary because
        # of the calc.update()
        '''
        CWD = os.getcwd()
        VASPDIR = calc.directory
        from .vasprc import VASPRC
        module = VASPRC['module']
        script = """#!/bin/bash
module load {module}

source ~/.bashrc # added by EPB - slight issue with "module load intel"

cd {CWD}
cd {VASPDIR}  # this is the vasp directory

{runfile}     # this is the vasp command
#end""".format(**locals())

        jobname = 'SCBands'
        cmdlist = ['{0}'.format(VASPRC['queue.command'])]
        cmdlist += ['-o', VASPDIR]
        cmdlist += [option for option in VASPRC['queue.options'].split()]
        cmdlist += ['-N', '{0}'.format(jobname),
                    '-l', 'walltime={0}'.format(VASPRC['queue.walltime']),
                    '-l', 'nodes={0}:ppn={1}'.format(VASPRC['queue.nodes'],
                                                     VASPRC['queue.ppn']),
                    '-l', 'mem={0}'.format(VASPRC['queue.mem']),
                    '-M', VASPRC['user.email']]
        p = sp.Popen(cmdlist,
                     stdin=sp.PIPE,
                     stdout=sp.PIPE,
                     stderr=sp.PIPE,
                     universal_newlines=True)

        out, err = p.communicate(script)

        '''

        return None, None

    else: # I don't think this will work unless the calculation is complete!

        os.chdir(wd)

        if imagedir is None:
            imagedir = os.getcwd()
        if imageprefix is None:
            imageprefix = 'unfolded_bandstructure'

        from unfold import unfold

        # nseg = len(kpts_path) -1

        WaveSuper = unfold(M=M, wavecar='WAVECAR', lsorbit=lsorb)
        sw = WaveSuper.spectral_weight(kpath_uc)

        from unfold import EBS_cmaps, EBS_scatter
        e0, sf = WaveSuper.spectral_function(nedos=4000)

        if eref is None:
            eref = ef
        if ymin is None:
            ymin = ef - 5
        if ymax is None:
            ymax = ef + 5
        print('The unit cell vectors are:')
        print(unit_cell)
        # Show the effective band structure with a scatter plot
        EBS_scatter(kpath_uc, unit_cell, sw, nseg=nseg, eref=eref,
                    ylim=(ymin, ymax), 
                    factor=5,
                    kpath_label=labels)

        scatterplot = os.path.join(imagedir, imageprefix) + '_scatter_plot.png' 
        plt.savefig(scatterplot)

        plt.close('all')
        # or show the effective band structure with colormap
        EBS_cmaps(kpath_uc, unit_cell, e0, sf, nseg=nseg, eref=eref,
                  show=False,
                  ylim=(ymin, ymax),
                  kpath_label=labels)

        colormap = os.path.join(imagedir, imageprefix) + '_colormap.png'  
        plt.savefig(colormap)

        return scatterplot, colormap



@monkeypatch_class(Vasp)
def get_supercell_bandstructure_ppc(self,
                      kpts_path=None,
                      kpts_nintersections=None,
                      supercell_size=[1,1,1],
                      unit_cell=[[1.0,0,0], [0,1.0,0], [0.0,0.0,0.0]],
                      show=False):
    """Calculate band structure along :param kpts_path:
    :param list kpts_path: list of tuples of (label, k-point) to
      calculate path on.
    :param int kpts_nintersections: is the number of points between
      points in band structures. More makes the bands smoother.
    :param list supercell_size: this lists the size of the supercell
      [nx ny nz] as multiples of the unit cell in the ordinal directions
    :param list unit_cell: list of unit cell vectors

    This version uses PyProcar for k-path preparation and unfolding.
       See https://romerogroup.github.io/pyprocar/index.html

    
    
    returns (npoints, band_energies, fighandle)

    """
    self.update()
    self.stop_if(self.potential_energy is None)

    M = [[1.0*supercell_size[0], 0.0, 0.0],
         [0.0, 1.0*supercell_size[1], 0.0],
         [0.0, 0.0, 1.0*supercell_size[2]]]


    dos = DOS(self, width=0.2)
    d = dos.get_dos()
    e = dos.get_energies()

    ef = self.get_fermi_level()

    kpts = [k[1] for k in kpts_path]
    labels = [k[0] for k in kpts_path]

    # by now, the self-consistent calculation is complete
    # run in non-selfconsistent directory
    wd = os.path.join(self.directory, 'bandstructure')

    
    if not os.path.exists(wd):
        self.clone(wd)

        calc = Vasp(wd)

        # this next line actually writes a K-points file, but we're
        # going to use pyprocar to overwrite it
        calc.set(kpts=kpts, kpts_nintersections=10,
                 reciprocal=True,
                 nsw=0,  # no ionic updates required
                 isif=None,
                 ibrion=None,
                 icharg=11, lorbit=12)

        os.remove( os.path.join(wd, 'KPOINTS') )

        # Let's try generating the default k-points path
        # Now I need to learn how to set the k-path using pyprocar
        #   see: 
        import pyprocar as ppc
        ppc.kpath(os.path.join(wd, 'POSCAR'),
                  os.path.join(wd, 'KPOINTS'),
                  supercell_matrix=np.diag( supercell_size ))
        

        # calc.update()
        # we'll just launch VASP - skip the calc.update()

        # Create and run a subprocess that invokes 
        if self.parameters.get('lsorbit'):
            runfile = 'runvasp_ncl.py'
        else:
            runfile = 'runvasp.py'
        CWD = os.getcwd()
        VASPDIR = calc.directory
        from .vasprc import VASPRC
        module = VASPRC['module']
        script = """#!/bin/bash
module load {module}

source ~/.bashrc # added by EPB - slight issue with "module load intel"

cd {CWD}
cd {VASPDIR}  # this is the vasp directory

{runfile}     # this is the vasp command
#end""".format(**locals())

        jobname = 'SCBands'
        cmdlist = ['{0}'.format(VASPRC['queue.command'])]
        cmdlist += ['-o', VASPDIR]
        cmdlist += [option for option in VASPRC['queue.options'].split()]
        cmdlist += ['-N', '{0}'.format(jobname),
                    '-l', 'walltime={0}'.format(VASPRC['queue.walltime']),
                    '-l', 'nodes={0}:ppn={1}'.format(VASPRC['queue.nodes'],
                                                     VASPRC['queue.ppn']),
                    '-l', 'mem={0}'.format(VASPRC['queue.mem']),
                    '-M', VASPRC['user.email']]
        p = sp.Popen(cmdlist,
                     stdin=sp.PIPE,
                     stdout=sp.PIPE,
                     stderr=sp.PIPE,
                     universal_newlines=True)

        out, err = p.communicate(script)
        '''

        return None, None, None

        # if calc.potential_energy is None:
        #    return None, None, None
     

    else: # I don't think this will work unless the calculation is complete!

        os.chdir(wd)

        import pyprocar

        pyprocar.unfold(
            fname='PROCAR',
            poscar='POSCAR',
            outcar='OUTCAR',
            supercell_matrix=np.diag(supercell_size),
            ispin=None, # None for non-spin polarized calculation. For spin polarized case, ispin=1: up, ispin=2: down
            efermi=None,
            shift_efermi=True,
            elimit=(-2, 2), # kticks=[0, 36, 54, 86, 110, 147, 165, 199], knames=['$\Gamma$', 'K', 'M', '$\Gamma$', 'A', 'H', 'L', 'A'],
            print_kpts=False,
            show_band=True,
            width=4,
            color='blue',
            savetab='unfolding.csv',
            savefig='unfolded_band.png',
            exportplt=False)

        return None, None, None
        '''
        from unfold import unfold

        WaveSuper = unfold(M=M, wavecar='WAVECAR')
        sw = WaveSuper.spectral_weight(kpath_uc)

        from unfold import EBS_cmaps
        e0, sf = WaveSuper.spectral_function(nedos=4000)
        # or show the effective band structure with colormap
        EBS_cmaps(kpath_sc, unit_cell, e0, sf, nseg=nseg,#  eref=-4.01,
                  show=False) #,
                  # ylim=(-3, 4))
        '''

        plt.savefig('unfolded_bandstructure.png')


        '''
        # In the fol
        archiveKpts = os.path.join(wd, 'KPOINTS_old')
        originalKpts = os.path.join(wd, 'KPOINTS')
        if not os.path.exists(archiveKpts): # archive the original KPOINTS file
            shutil.copy(originalKpts, archiveKpts)

        with open(archiveKpts, 'r') as kpts_infile: # get lines of old KPOINTS
            KPtsLines = kpts_infile.readlines()

        # Append labels for high-symmetry points to lines of the KPOINTS file
        reject_chars = '$'
        for idx, label in enumerate(labels):
            newlabel = label
            for ch in reject_chars:
              newlabel = newlabel.replace(ch, '')

            KPtsLines[4+idx] = KPtsLines[4+idx][:-1] + f' ! {newlabel}\n'

        # Write a new version of the k-points file
        with open(originalKpts, 'w') as kpts_outfile:
            for line in KPtsLines:
                kpts_outfile.write(line)
    
        # This is John Kitchin's original band structure visualization
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
                          show=False,
                          outdir=None,
                          outfile=None):
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

    """
    Our first job will be to copy the working directory to a subdirectory.
    This will happen on the first invocation of get_bandstructure_v02()
    """

    self.update()
    self.stop_if(self.potential_energy is None)

    kx, ky, kz, labels, label_idx = interpolate_kpath(kpts_path,
                                                      kpts_nintersections)
    
    num_new_k_points = len(kx)

    '''
    kx = np.linspace(kpts[0][0], kpts[1][0], kpts_nintersections)
    ky = np.linspace(kpts[0][1], kpts[1][1], kpts_nintersections)
    kz = np.linspace(kpts[0][2], kpts[1][2], kpts_nintersections)
    '''

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
        # This branch should be taken on the first invocation of
        # get_bandstructure_v02
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

        # Deleting the old KPOINTS file is probably a bad thing
        # old_kpts = os.path.join( calc.directory, 'KPOINTS')
        # os.remove(old_kpts)

        # Read the automatically-generated k-points from IBZKPT
        IBZKPT = os.path.join( calc.directory, 'IBZKPT' ) 
        with open(IBZKPT, 'rt') as infile:
            infile.readline()
            nk_old = int(infile.readline().split()[0])
            # print(nk_old)
            print(f'Found {nk_old} k-points in IBZKPT.')
            print(f'There are {num_new_k_points} additional zero-weight k-points.')
            print('Reading the original k-points...')
            original_k_lines = []
            for idx in range(0, nk_old+1):
                original_k_lines.append(infile.readline())
            # print(': {0}'.format(original_k_lines[idx])) 

        total_k_points = nk_old + num_new_k_points
        
        print(f'There are a total of {total_k_points} k-points.')

        # Obtain text lines for the original k-points
        HSE_lines = ['Explicit k-point list\n', ' '*6 + f'{total_k_points}\n']
        for line in original_k_lines:
            HSE_lines.append(line)

        # Make text lines for the new k points
        for idx in range(0, num_new_k_points):
            line_str = '{0:15.12f} {1:15.12f} {2:15.12f} 0.0\n'.format(kx[idx],
                                                                       ky[idx],
                                                                       kz[idx])
            HSE_lines.append(line_str)

        # for line in HSE_lines:
        #    print(line)
        
        # Write the 'KPOINTS_HSE_bands' file
        tgt = os.path.join(calc.directory, 'KPOINTS')# 'KPOINTS_HSE_bands')
        with open(tgt, 'w') as KPTSfile:
            for line in HSE_lines:
                KPTSfile.write(line)

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
        if self.parameters.get('lsorbit'):
            runfile = 'runvasp_ncl.py'
        else:
            runfile = 'runvasp.py'

        CWD = os.getcwd()
        VASPDIR = calc.directory
        from .vasprc import VASPRC
        module = VASPRC['module']
        script = """#!/bin/bash
module load {module}

source ~/.bashrc # added by EPB - slight issue with "module load intel"

cd {CWD}
cd {VASPDIR}  # this is the vasp directory

{runfile}     # this is the vasp command
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
        p = sp.Popen(cmdlist,
                     stdin=sp.PIPE,
                     stdout=sp.PIPE,
                     stderr=sp.PIPE,
                     universal_newlines=True)

        out, err = p.communicate(script)

        return None

    else:
        """
        This should happen on the second invocation of get_bandstructure_v02.
        Here, we will add labels for the zero-weight k-points.
        """

        # Read the automatically-generated k-points from IBZKPT
        IBZKPT = os.path.join( wd, 'IBZKPT' ) 
        with open(IBZKPT, 'rt') as infile:
            infile.readline()
            nk_old = int(infile.readline().split()[0])
            # print(nk_old)
            print(f'Found {nk_old} k-points in IBZKPT.')

        kx, ky, kz, spec_k_labels, label_idx = interpolate_kpath(kpts_path,
                                                                 kpts_nintersections)

        num_new_k_points = len(kx)

        import shutil

        archive_KPOINTS = os.path.join(wd, 'KPOINTS_original')
        if not os.path.exists(archive_KPOINTS):
            shutil.copy(os.path.join(wd, 'KPOINTS'),
                        archive_KPOINTS)

        with open(archive_KPOINTS, 'r') as infile:
            KPTS_Lines  = infile.readlines()

        for idx, label in enumerate(spec_k_labels):
            label = label.replace('$', '')
            newk_idx = label_idx[idx]
            line_with_label = KPTS_Lines[3+nk_old+newk_idx][:-1] + f' {label}\n'
            KPTS_Lines[3+nk_old+newk_idx] = line_with_label

        with open(os.path.join(wd, 'KPOINTS'), 'w') as KPOINTSfile:
            for line in KPTS_Lines:
                KPOINTSfile.write(line)


        # Run sumo-bandplot, if it exists
        check_sumo = sp.run(['which', 'sumo-bandplot'], capture_output=True)
        found_sumo = not check_sumo.stdout == b''
        found_band_data = os.path.exists(os.path.join(wd, 'vasprun.xml'))


        if found_sumo and found_band_data:
            os.chdir(wd)
            sumo_cmd = ['sumo-bandplot']
            run_sumo = sp.run(sumo_cmd, capture_output=True)
            if outfile is not None:
                shutil.copy('band.pdf', outfile)
            else:
                outfile = 'band.pdf'
            if outdir is not None:
                target = os.path.join(outdir, outfile)
                shutil.copy(outfile, target)
            else:
                target = os.path.join(os.getcwd(), outfile)

            Egap, Ecbm, Evbm = read_bandstats()
            
        else:
            print('sumo-bandplot not found. No band plot generated.')
            target = None
            Egap, Ecbm, Evbm = None, None, None


    return {'file': target, 'Egap': Egap,
            'Ecbm': Ecbm, 'Evbm': Evbm}


def interpolate_kpath(kpts_path, kpts_nintersections=10):
    '''
    This obtains k-points from a k-point path specified by several end-points.
    Points are interpolated along the path.
    '''

    kpts = [k[1] for k in kpts_path]
    labels = [k[0] for k in kpts_path]

    # Calculate the zero-weight k-points
    num_segments = int(len(kpts_path)/2)
    label_idx = []
    reduced_labels = []
    for seg_idx in range(0, num_segments):
        # print('Analyzing segment {0} of {1}'.format(seg_idx+1, num_segments) )
        ki, kf = kpts[2*seg_idx], kpts[2*seg_idx + 1]

        if len(kpts_path[2*seg_idx]) == 3:
            nk = kpts_path[2*seg_idx][2]
        else:
            nk = kpts_nintersections

        # Interpolate points for each segment of the k-point path
        segx = np.linspace(ki[0], kf[0], nk)
        segy = np.linspace(ki[1], kf[1], nk)
        segz = np.linspace(ki[2], kf[2], nk)

        # handle the first segment as a special case
        if seg_idx == 0:
            kx, ky, kz = segx, segy, segz
            num_new_k_points = nk
            label_idx = [0, num_new_k_points -1]
            reduced_labels = [labels[0], labels[1]]
            points_added = nk
        else:
            # In this case, the new segment start conincides with the last
            #    segment ending point. We discard the new segment start
            if labels[2*seg_idx] == labels[2*seg_idx-1]:
                seg_points = nk - 1
                kx = np.append(kx, segx[1:])
                ky = np.append(ky, segy[1:])
                kz = np.append(kz, segz[1:])
                num_new_k_points += seg_points
                # Add index for new segment end point
                label_idx.append( num_new_k_points-1 )
                # Add label for new segment end point
                reduced_labels.append( labels[2*seg_idx+1] )

            # In this case, the new segment start does not coincide with the
            #    last segment ending point. We do not discard the new segment
            #    start
    

            else:
                seg_points = nk
                kx = np.append(kx, segx)
                ky = np.append(ky, segy)
                kz = np.append(kz, segz)
                # Add index for new segment start point
                label_idx.append( num_new_k_points )
                # Add label for new segment starting point
                reduced_labels.append( labels[2*seg_idx] )
                num_new_k_points += seg_points
                # Add index for new segment end point
                label_idx.append( num_new_k_points - 1 )
                # Add label for new segment end point
                reduced_labels.append( labels[2*seg_idx+1] )
        
    # print(f'There a total of {num_new_k_points} points in the path.')

    return kx, ky, kz, reduced_labels, label_idx


def read_bandstats():
   """
   Uses sumo-bandstats to obtain the band gap, valence band minimum, and
   valence band maximum.
   """

   read_stats = sp.run(['sumo-bandstats'], capture_output=True)
   
   # It's not right that we get data from stderr. It should be stdout.
   # This is a bug in sumo 1.10 (https://sumo.readthedocs.io/en/latest/sumo-bandplot.html)
   statlines = read_stats.stderr.decode(encoding='UTF-8').split('\n')
   for line in statlines:
      print(line)

   if len(statlines) > 5:
      add = 0
      #if there's an "Indirect band gap" line at the beginning of the ouptut -- diamond fail
      if statlines[0].split(' ')[0] == "Indirect":
         add = 1
      Evbm = float(list(filter(None, statlines[6+add].split(' ')))[1])
      Ecbm = float(list(filter(None, statlines[13+add].split(' ')))[1])

      Egap = Ecbm - Evbm

   else:
      Egap, Evbm, Ecbm = None, None, None

   # ef = '7.3f'
   # print(f'Gap = {Egap:{ef}} eV; Ecbm = {Ecbm:{ef}} eV; Evbm = {Evbm:{ef}} eV')

   return Egap, Ecbm, Evbm
