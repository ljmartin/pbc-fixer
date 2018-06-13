# pbc-fixer
Quickly equilibrates the cell size of new systems for simulations using periodic boundary conditions in OpenMM

## reason
OpenMM barostats scale the coordinates of all molecules, rather than the coordinates of the unit cell (like, say, NAMD). For generating new systems for simulation, often you use restraints on the protein atoms, which fight against the movements of the barostat.

This script takes a system that you have chopped out of a `PBDFixer.addMembrane` system, and reduces the system size enough that the sides of the water box are touching. The lipids that were originally poking out of the system are quickly equilibrated to prevent explosions. The result is a protein/lipid/water system at an appropriate density ready for simulation.  

It is currently set up for equilibrating triclinc cells, an appropriate selection (in VMD) for the starting system would be:
 `z>-30 and (protein or (water and same residue as x<40 and same residue as x>-40 and same residue as y<50 and same residue as y>-50) or (lipid and same residue as x<30 and same residue as x>-30 and same residue as y<40 and same residue as y>-40))`
 
## code 
 ```
 from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
from sys import stdout

# Load CHARMM files
psf = CharmmPsfFile('ionized.psf')
pdb = PDBFile('ionized.pdb')

#set box with 4 angstrom buffer:
maxx = max([i[0]*1/nanometer for i in pdb.getPositions()])
maxy = max([i[1]*1/nanometer for i in pdb.getPositions()])
maxz = max([i[2]*1/nanometer for i in pdb.getPositions()])
psf.setBox((maxx+0.4)*nanometer, (maxx+0.4)*nanometer, (maxz+0.4)*nanometer, 90*degrees, 90*degrees, 60*degrees)

# create system
params = CharmmParameterSet('par_all36m_prot.prm', 'toppar_water_ions.str', 'par_all36_lipid.prm')
system = psf.createSystem(params, nonbondedMethod=PME, nonbondedCutoff=0.8*nanometer, constraints=HBonds)

# for density calculation
systemMass = sum([system.getParticleMass(i[0]) for i in enumerate(pdb.topology.atoms())])

#hold all protein atoms:
force = CustomExternalForce("k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
force.addGlobalParameter("k", 1*kilocalories_per_mole/angstroms**2)
force.addPerParticleParameter("x0")
force.addPerParticleParameter("y0")
force.addPerParticleParameter("z0")
for i, z in zip(pdb.topology.atoms(), pdb.getPositions()):
    if i.residue.name in ['PHE', 'LYS', 'ARG', 'ILE', 'LEU', 'HIS', 'HSD', 'ALA', 'GLN', 'GLU', 'ASN', 'MET', 'TYR', 'SER', 'ASP', 'TRP', 'CYS', 'GLY', 'THR', 'PRO', 'VAL']:
        force.addParticle(int(i.index), [z[0], z[1], z[2]])
system.addForce(force)

#small timestep to stop explosions
integrator = LangevinIntegrator(10*kelvin, 1/picosecond, 0.0005*picoseconds)
platform = Platform.getPlatformByName('CUDA')
#platform = Platform.getPlatformByName('CPU')
#prop = {'CudaPrecision':'single', 'CudaDeviceIndex':'0'}

simulation = Simulation(pdb.topology, system, integrator, platform)
simulation.context.setPositions(pdb.positions)
state = simulation.context.getState(getEnergy=True)
print("initial energy = ", state.getPotentialEnergy())
simulation.minimizeEnergy()
state = simulation.context.getState(getEnergy=True)
print("energy after minimization = ", state.getPotentialEnergy())
simulation.context.setVelocitiesToTemperature(10*kelvin)
print('just set velocities')

simulation.reporters.append(DCDReporter('./equilibrate-pbc.dcd', 50))

##close cell in towards the boundary of water box, equilibrating lipids that bump against each other
for i in range(1,200):
    boxflag = False
    #get pbc values
    pbc = simulation.context.getState().getPeriodicBoxVectors()
    pbcx=new_pbcx = pbc[0][0]/nanometer
    pbcy=new_pbcy = pbc[1][1]/nanometer
    pbcz=new_pbcz = pbc[2][2]/nanometer
    systemVolume = pbcx * pbcy * pbcz*nanometer**3
    #get system size
    waterlist = [i[1] for i in zip(pdb.topology.atoms(), pdb.getPositions()) if i[0].residue.name == 'HOH']
    maxx = max([i[0]/nanometer for i in waterlist])
    maxy = max([i[1]/nanometer for i in waterlist])
    maxz = max([i[2]/nanometer for i in waterlist])
    #reduce size of the cell towards system size
    if pbcx >= maxx:
        new_pbcx = pbcx-0.05
        boxflag = True
    if pbcy >= maxy:
        new_pbcy = pbcy-0.05
        boxflag = True
    if pbcz >= maxz:
        new_pbcz = pbcz-0.05
        boxflag = True
    if boxflag is True:
        #set new triclinic cell:
        simulation.context.setPeriodicBoxVectors(Vec3(new_pbcx,0,0), Vec3((-0.5*new_pbcx),new_pbcy,0), Vec3(0,0,new_pbcz))
        print('fixing box.', i, end='\r')
    if boxflag is False:
        break
    #equilibrate the lipids that poke out of system
    simulation.step(50)

##now squish the cell until an appropriate density is achieved
pbc = simulation.context.getState().getPeriodicBoxVectors()
pbcx = pbc[0][0]/nanometer
pbcy = pbc[1][1]/nanometer
pbcz = pbc[2][2]/nanometer
systemVolume = pbcx * pbcy * pbcz*nanometer**3
density = (systemMass / systemVolume).value_in_unit(gram/item/milliliter)
while density <= 1.06:
    print('finished fixing box - fixing density which is: ', density, end='\r')
    simulation.context.setPeriodicBoxVectors(Vec3(pbcx*0.999,0,0), Vec3((-0.5*(pbcx*0.999)),pbcy*0.999,0), Vec3(0,0,pbcz*0.999))
    simulation.step(50)
    pbc = simulation.context.getState().getPeriodicBoxVectors()
    pbcx = pbc[0][0]/nanometer
    pbcy = pbc[1][1]/nanometer
    pbcz = pbc[2][2]/nanometer
    systemVolume = pbcx * pbcy * pbcz*nanometer**3
    density = (systemMass / systemVolume).value_in_unit(gram/item/milliliter)

integrator.setTemperature(300*kelvin)
simulation.reporters.append(StateDataReporter(stdout, 50, step=True, time=True, potentialEnergy=True, temperature=True, density=True, totalSteps=50000))
simulation.step(50000)
```
