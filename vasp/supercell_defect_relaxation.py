from ase import Atoms
from ase import Atom
from ase import neighborlist
from ase import io
from ase import optimize

# from ase.utils import natural_cutoffs
from ase.neighborlist import natural_cutoffs
from ase.build import find_optimal_cell_shape
from ase.build import make_supercell
from ase.constraints import FixAtoms

import numpy as np
import sys

from gpaw import GPAW, PW, FermiDirac

def makeSupercellPristine_rep(cellPrim, M, N, K):
	# overloaded
	# simple repetition of primitive cells
	# primatime cell: cellPrim
	# Supercell: M x N x K
	supercellPristine = cellPrim.repeat((M, N, K))
	return supercellPristine

def makeSupercellPristine_bui(cellPrim, numCells, shape = 'sc'):
	# overloaded
	# use ase.build.find_optimal_cell_shape and ase.biuld.make_supercell
	# primitive cell: cellPrim
	# number of cells: numCells
	# ideal shape: shape
	P = find_optimal_cell_shape(cellPrim.cell, numCells, shape)
	supercellPristine = make_supercell(cellPrim, P)
	return supercellPristine

def add_defect(supercellPristine, numLayers, type = 1, location = 'origin', substitutionDefect = None, neighborOffset = 0, substitutionNeighbor = None):
	# the neighborOffset is which neighbor type 3 defect take place, default 0
	# type 1: single atom vacancy
	# type 2: single atom substitution
	# type 3: single atom vacancy and single neighbor substitution
	if location == 'center':
		defectIndex = finding_center_atom(supercellPristine)
	elif location == 'origin':
		defectIndex = 0
	else:
		pass

	try:
		maskIndices = generate_mask_indices_neightbor_list(supercellPristine, defectIndex, numLayers)
	except:
		print('Error: defectIndex not provided')
		sys.exit(1)

	# in function test substituting the atoms within mask
	# in_function_test_substitute_atoms(supercellPristine, maskIndices)

	constraint = FixAtoms(mask = [atom.index not in maskIndices for atom in supercellPristine])
	supercellPristine.set_constraint(constraint)

	if type == 1:
		add_defect_helper(supercellPristine, defectIndex)
	elif type == 2:
		add_defect_helper(supercellPristine, defectIndex, substitutionDefect)
	elif type == 3:
		neighborIndex = find_neighbor_index(supercellPristine, defectIndex, neighborOffset)
		add_defect_helper(supercellPristine, defectIndex, None, neighborIndex, substitutionNeighbor)
	else:
		pass

	return supercellPristine # after this point this should be used as defect supercell instead of supercellPristine

def add_defect_helper(supercellPristine, defectIndex, substitutionDefect = None, neighborIndex = -1, substitutionNeighbor = None):
	# not overloaded
	# add defect at defectIndex, default index 0
	# select type of defect and location of defect
	# pristine supercell: supercellPristine
	# defect index: defectIndex
	# neighbor index: neighborIndex
	# defect substitution: substitutionDefect
	# neighbor substitution: substitutionNeighbor
	# if substitutionDefect is None:
	# 	supercellPristine.pop(defectIndex)
	# else:
	# 	supercellPristine[defectIndex].symbol = substitutionDefect

	# if neighborIndex < 0:
	# 	pass
	# else:
	# 	if substitutionNeighbor is None:
	# 		supercellPristine.pop(neighborIndex)
	# 	else:
	# 		supercellPristine[neighborIndex].symbol = substitutionNeighbor

	# popping atom from atoms object will change the index of other atom in the atoms object
	# the solution is do the substitution first, then do the popping. For future reference, popping should be done in the decending order of indices if popping more than one atom

	if substitutionDefect is not None:
		supercellPristine[defectIndex].symbol = substitutionDefect

	if (substitutionNeighbor is not None) & (neighborIndex >= 0):
		supercellPristine[neighborIndex].symbol = substitutionNeighbor

	if substitutionDefect is None:
		supercellPristine.pop(defectIndex)

def generate_mask_indices_neightbor_list(supercellPristine, defectIndex, numLayers):
	# this function should be used before the defect was inserted
	# in type 1 and type 3 defect situation, the defect atom was removed, won't be able to calculate neightbor based an vacancy
	
	# generate and update the neightborList
	cutOff = natural_cutoffs(supercellPristine, 1)
	nl = neighborlist.NeighborList(cutOff, 0.3, False, True, True)
	nl.update(supercellPristine)

	# # takes out the symmetric connectivity matrix
	# matrix = nl.get_connectivity_matrix()

	# sets of indices
	totalSet = []
	newSet = []
	newNewSet = []

	totalSet.append(defectIndex)
	newSet.append(defectIndex)

	# print totalSet
	# print newSet
	# print "======="

	# grow starting from the defect, N layers
	for layer in range(numLayers):
		# for idx in newSet:
		# 	indices, offsets = nl.get_neighbors(idx)
		# 	newSet.remove(idx)
		# 	for idxx in indices:
		# 		if idxx not in totalSet:
		# 			totalSet.add(idxx)
		# 			newSet.add(idxx)
		# 		else:
		# 			pass
		while newSet:
			idx = newSet.pop()
			indices, offsets = nl.get_neighbors(idx)
			for idxx in indices:
				if idxx not in totalSet:
					totalSet.append(idxx)
					newNewSet.append(idxx)
				else:
					pass
		newSet = newNewSet
		newNewSet = []

		# print totalSet
		# print newSet
		# print "======="
	return totalSet

def finding_center_atom(cell):
	minimum = 2.0
	index = 0
	scaled_positions = cell.get_scaled_positions()
	natom = len(cell)
	for idx in range(natom):
		disp_x = np.abs(scaled_positions[idx][0] - 0.5)
		disp_y = np.abs(scaled_positions[idx][1] - 0.5)
		disp_z = np.abs(scaled_positions[idx][2] - 0.5)
		disp = np.sqrt(disp_x * disp_x + disp_y * disp_y + disp_z * disp_z)
		if disp < minimum:
			index = idx
			minimum = disp
	return index

def find_neighbor_index(cell, defectIndex, neighborOffset):
	# very similar to generating the mask indices
	cutOff = natural_cutoffs(cell, 1)
	nl = neighborlist.NeighborList(cutOff, 0.3, False, False, False)
	nl.update(cell)

	index_nei, offset_nei = nl.get_neighbors(defectIndex)

	# if index_nei.any():
	# 	return index_nei[0 + neighborOffset]
	# else:
	# 	return defectIndex + 1

	try:
		return index_nei[0 + neighborOffset]
	except IndexError:
		print('Warning: The defect atom does not have enough neighbor for neighborOffset = ', neighborOffset)
		print('Warning: Falling back to first neighbor')

		try:
			return index_nei[0]
		except:
			print('Warning: The defect atom does not have any available neighbor')
			print('Warning: Falling further back to next atom')

			return defectIndex + 1

# substitute masked atoms with test atoms
# depreciated after use

# def in_function_test_substitute_atoms(test_subject_cell, indices, substitution = 'Cu'):
# 	for idx in indices:
# 		test_subject_cell[idx].symbol = substitution

def supercell_ionic_relaxation(superCellDefect, optimizer_type = 'QuasiNewton',
                               ladder_begin = 0, ladder_end = 2,
                               charge=0):
	# structure relaxation witht he calculator already existing in the atoms object
	# returns the potential energy of the supercell before and after the relaxation as a tuple
	# 	usage: BR, AR = supercell_ionic_relaxation(defect_supercell)
	# if log ==  False, both potential energy will be 0

	# added support for different types of optimizer
	# local optimizer:  QuasiNewton, BFGS, LBFGS, GPMin, MDMin and FIRE.
	# preconditioned optimizer: to be added
	# global optimizer: to be added 

	# added support for the jacob's ladder
	# the ladder: 
	#			LDA,
	# 			PBE,
	# 			revPBE
	#			RPBE,
	#			PBE0,
	#			B3LYP.

	# ladder_begin, beginning step of the ladder, included
	# ladder_end, ending step of the ladder, NOT INCLUDED

	potential_energy_BR = 0.0
	potential_energy_AR = 0.0

	ladder = ['LDA', 'PBE', 'revPBE', 'RPBE', 'PBE0', 'B3LYP']
	# calc = superCellDefect.get_calculator()
	# calc_backup = calc.get_xc_functional()

	# if log:
	# 	io.write('defect_supercell_before_relaxation.cube', superCellDefect)
	# 	potential_energy_BR = superCellDefect.get_potential_energy()
	# else:
	# 	pass

	if ladder_end > len(ladder):
		ladder_end = len(ladder)

	# loop climbing the ladder
	for calc_step in ladder[ ladder_begin : ladder_end ]:
		# print('=====================', file = log_file)
		# print('starting:', file = log_file)
		# print(calc_step, file = log_file)

		calc = GPAW(mode='fd',
		            kpts={'size': (2, 2, 2), 'gamma': False},
		            xc=calc_step,
		            charge=charge,
		            occupations=FermiDirac(0.01)
		            )

		# calc.set(xc = calc_step)
		superCellDefect.set_calculator(calc)

		if optimizer_type == 'QuasiNewton':
			relax = optimize.QuasiNewton(superCellDefect)
		elif optimizer_type == 'BFGS':
			relax = optimize.BFGS(superCellDefect)
		elif optimizer_type == 'LBFGS':	
			relax = optimize.BFGSLineSearch(superCellDefect)
		elif optimizer_type == 'GPMin':
			relax = optimize.GPMin(superCellDefect)
		elif optimizer_type == 'FIRE':
			relax = optimize.FIRE(superCellDefect)
		elif optimizer_type == 'MDMin':
			relax = optimize.MDMin(superCellDefect)
		else:
			print('optimizer not supported at the moment, falling back to QuasiNewton')
			relax = optimize.QuasiNewton(superCellDefect)

		relax.run(fmax = 0.05)

		potential_current = superCellDefect.get_potential_energy()
		# print('lattice energy after relaxation: %5.7f eV' % potential_current, file = log_file)
		# print('=====================', file = log_file)

	# put the original xc back to the calculator
	# calc.set(xc = calc_backup)

	# relax = QuasiNewton(superCellDefect)
	# relax.run(fmax=0.05)
	
	# if log:
	# 	io.write('defect_supercell_after_relaxation.cube', superCellDefect)
	# 	potential_energy_AR = superCellDefect.get_potential_energy()
	# else:
	# 	pass

	return potential_energy_BR, potential_energy_AR









