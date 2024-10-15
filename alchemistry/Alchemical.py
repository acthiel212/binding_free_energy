from intspan import intspan

def apply_lambdas(context, alchemical_atoms, vdwForce, vdw_lambda, multipoleForce, elec_lambda):
    # Parse alchemical_atoms input
    alchemical_atoms = list(intspan(alchemical_atoms))
    # OpenMM atom index starts at zero while FFX starts at 1. This allows the flags between FFX and OpenMM to match
    alchemical_atoms = [i - 1 for i in alchemical_atoms]
    print(f"Alchemical atoms: {alchemical_atoms}")

    context.setParameter("AmoebaVdwLambda", vdw_lambda)
    vdwForce.setAlchemicalMethod(2)  # 2 == Annihilate, 1 == Decouple
    for i in alchemical_atoms:
        params = vdwForce.getParticleParameters(i)
        vdwForce.setParticleParameters(i, params[0], params[1], params[2], params[3], True, params[5])
    # Update force parameters
    vdwForce.updateParametersInContext(context)

    # Apply alchemical scaling to the multipole force
    for i in alchemical_atoms:
        params = multipoleForce.getMultipoleParameters(i)
        charge = params[0] * elec_lambda
        dipole = [d * elec_lambda for d in params[1]]
        quadrupole = [q * elec_lambda for q in params[2]]
        polarizability = params[-1] * elec_lambda
        # Update multipole parameters (keeping other parameters unchanged)
        multipoleForce.setMultipoleParameters(i, charge, dipole, quadrupole, *params[3:-1], polarizability)
    # Update force parameters
    multipoleForce.updateParametersInContext(context)

    # Reinitialize the context to ensure changes are applied
    context.reinitialize(preserveState=True)

def setup_alchemical_forces(system):
    # Setup simulation context
    numForces = system.getNumForces()
    forceDict = {}
    for i in range(numForces):
        force_name = system.getForce(i).getName()
        forceDict[force_name] = i
    print(f"Force dictionary: {forceDict}")

    # Ensure that the specified forces exist
    required_forces = ['AmoebaVdwForce', 'AmoebaMultipoleForce']
    for force in required_forces:
        if force not in forceDict:
            raise ValueError(f"Required force '{force}' not found in the system.")

    vdwForce = system.getForce(forceDict.get('AmoebaVdwForce'))
    vdwForce.setForceGroup(1)
    multipoleForce = system.getForce(forceDict.get('AmoebaMultipoleForce'))
    multipoleForce.setForceGroup(1)
    return vdwForce, multipoleForce