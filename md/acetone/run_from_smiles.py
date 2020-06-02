import yaml, sys, os, time, logging
import matplotlib.pyplot as plt
from openforcefield.topology import Molecule, Topology
from openforcefield.typing.engines.smirnoff import ForceField, NonbondedMethod
from openforcefield.utils.toolkits import RDKitToolkitWrapper, AmberToolsToolkitWrapper
from simtk import openmm, unit
from rdkit import Chem
import mdtraj as mdt

#### packmol needs to be in PATH so distutils.spawn.find_executable can locate it
os.environ["PATH"] += os.pathsep + os.path.expanduser("~") + "/sw/packmol"
from openmoltools.packmol import pack_box

logging.basicConfig(level=logging.INFO, filename="log.log", format='%(asctime)s %(name)-12s  %(message)s', datefmt='%m-%d %H:%M', filemode="w")

logger = logging.getLogger(__name__)

#### USAGE:
#### python run_from_smiles.py SMILES-CODE

#### Expects ``mdconf.yml`` in the same directory.
 
def run_md(molecule, solvent_name="chloroform", confId=0):
    """
    Uses the PARSLEY forcefield to compute molecule ``molecule`` in a cubic box of solvent at STP.

    Details:
        - Particle mesh Ewald summation is used (1 nm cutoff)
        - Periodic boundary conditions are employed
        - Langevin thermostat is employed to regulate temperature
        - Box size is automatically scaled to the desired number of solvent molecules

    Args:
        molecule (openforcefield.topology.Molecule): desired molecule
        solvent_name (str): either ``chloroform`` or ``benzene``, for now
        confId (int): conformer ID for autogenerated molecular conformers, 0 seems fine by default

    Returns:
        Nothing, but ``.csv``, ``.hdf5``, and ``.pdb`` files are generated in the current directory.
    """

    #### Load in the appropriate Molecule object
    off_solute = molecule.to_topology()
    omm_solute = off_solute.to_openmm()
    mdt_solute = mdt.Topology.from_openmm(omm_solute)

    #### Build solvent Molecule object
    solvent, density, mw = None, None, None
    if solvent_name == "chloroform":
        solvent = Molecule.from_smiles("C(Cl)(Cl)Cl")
        density = 1.49
        mw = 119.38
    elif solvent_name == "benzene":
        solvent = Molecule.from_smiles("c1ccccc1")
        density = 0.879
        mw = 78.11
    else:
        raise ValueError(f"Unknown solvent {solvent_name}!")

    solvent.generate_conformers()
    off_solvent = solvent.to_topology()
    omm_solvent = off_solvent.to_openmm()
    mdt_solvent = mdt.Topology.from_openmm(omm_solvent)

    #### Calculate box side length
    num, length = None, None
    if "num" in config:
        num = config["num"]
        assert isinstance(num, int), "Need an integer number of solvent molecules."
        assert num > 0, "Need a positive number of solvent molecules."
        length = (1.6606 * num * mw / density) ** (1/3) # 1.6606 = 10^24 (Å**3 per mL) divided by Avogadro's number 
    elif "length" in config:
        length = config["length"]
        assert isinstance(length, (int, float)), "Need a numeric side length."
        assert length > 0, "Need a positive length."
        num = (length ** 3) * density / (mw * 1.6606)
        num = int(num)
    else:
        raise ValueError("Need ``length`` or ``num`` in config file!")

    logger.info(f"{num} solvent molecules in a cube with {length:.2f} Å sides.")

    #### Write solvent and solute to ``.pdb`` files for PACKMOL
    solute_pdb = "solute.pdb"
    with open(solute_pdb, "w+") as f: 
        openmm.app.pdbfile.PDBFile.writeFile(omm_solute, molecule.conformers[confId], f)

    solvent_pdb = "solvent.pdb"
    with open(solvent_pdb, "w+") as f: 
        openmm.app.pdbfile.PDBFile.writeFile(omm_solvent, solvent.conformers[0], f)

    #### Use ``openmoltools`` Python wrapper for PACKMOL to fill the box appropriately
    mdt_trajectory= pack_box([solute_pdb, solvent_pdb], [1, num], box_size=length)

    #### Convert back to ``openforcefield``
    omm_topology = mdt_trajectory.top.to_openmm()
    length = length / 10 # OpenMM uses nanometers for some stupid reason
    omm_topology.setPeriodicBoxVectors(((length, 0, 0), (0, length, 0), (0, 0, length)))
    off_topology = Topology.from_openmm(omm_topology, [Molecule.from_topology(off_solute), Molecule.from_topology(off_solvent)])

    logger.info(f"BOX VECTORS: {off_topology.box_vectors}")

    #### Set up the OpenMM system
    forcefield.get_parameter_handler('Electrostatics').method = 'PME'
    system = forcefield.create_openmm_system(off_topology)
    time_step = config["time_step"] * unit.femtoseconds
    temperature = config["temperature"] * unit.kelvin
    friction = 1 / unit.picosecond
    integrator = openmm.LangevinIntegrator(temperature, friction, time_step)
    
    #### Set up the simulation 
    simulation = openmm.app.Simulation(omm_topology, system, integrator)
    simulation.context.setPositions(mdt_trajectory.openmm_positions(0))

    pdb_reporter = openmm.app.PDBReporter('trj.pdb', config["pdb_freq"])
    hdf5_reporter = mdt.reporters.HDF5Reporter('trj.hdf5', config["hdf5_freq"])
    state_data_reporter = openmm.app.StateDataReporter(
        "data.csv",
        config["data_freq"],
        step=True,
        potentialEnergy=True,
        temperature=True,
        density=True
    )
#    simulation.reporters.append(pdb_reporter)
    simulation.reporters.append(hdf5_reporter)
    simulation.reporters.append(state_data_reporter)

    logger.info("Using Platform: " + simulation.context.getPlatform().getName())
    
    #### Clean up ``.pdb`` files
    os.remove(solute_pdb)
    os.remove(solvent_pdb)

    logger.info("Minimizing...")
    simulation.minimizeEnergy(maxIterations=25)

    logger.info("Running...")
    w_start = time.time()
    p_start = time.process_time()
    simulation.step(config["num_steps"])
    w_end = time.time()
    p_end = time.process_time()
    logger.info(f"Elapsed time {w_end-w_start:.2f} s (CPU: {p_end-p_start:.2f} s)")
    logger.info("Done")
 
if __name__=="__main__":
    forcefield = ForceField("openff_unconstrained-1.2.0.offxml")
    config = yaml.load(open("mdconf.yml", "r"), yaml.Loader)
    molecule = Molecule.from_smiles(sys.argv[1])
    molecule.generate_conformers()

    if len(sys.argv) > 2:
        run_md(molecule, sys.argv[2])
    else:
        logger.info("Using chloroform as default solvent...")
        run_md(molecule)