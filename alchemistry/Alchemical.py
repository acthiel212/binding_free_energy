from intspan import intspan


# Apply lambda to alchemical forces that were setup in below method.
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


# Save Default Electrostatic params when applying lambda more than once.
def save_default_elec_params(multipoleForce, alchemical_atoms):
    # Parse alchemical_atoms input
    alchemical_atoms = list(intspan(alchemical_atoms))
    # OpenMM atom index starts at zero while FFX starts at 1. This allows the flags between FFX and OpenMM to match
    alchemical_atoms = [i - 1 for i in alchemical_atoms]
    params = []
    for i in alchemical_atoms:
        param = multipoleForce.getMultipoleParameters(i)
        params.append(param)
    return params


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


# Helper function to set lambda values and update forces in the context
def update_lambda_values(context, vdw_lambda, elec_lambda, vdwForce, multipoleForce, alchemical_atoms,
                         default_elec_params):
    # Parse alchemical_atoms input
    alchemical_atoms = list(intspan(alchemical_atoms))
    # OpenMM atom index starts at zero while FFX starts at 1. This allows the flags between FFX and OpenMM to match
    alchemical_atoms = [i - 1 for i in alchemical_atoms]
    context.setParameter("AmoebaVdwLambda", vdw_lambda)
    vdwForce.setAlchemicalMethod(2)  # 2 == Annihilate, 1 == Decouple
    for i in alchemical_atoms:
        [parent, sigma, eps, redFactor, isAlchemical, type] = vdwForce.getParticleParameters(i)
        vdwForce.setParticleParameters(i, parent, sigma, eps, redFactor, True, type)

    j = 0
    for i in alchemical_atoms:
        param = default_elec_params[j]
        charge, dipole, quadrupole, polarizability = param[0], param[1], param[2], param[-1]
        charge = charge * elec_lambda
        dipole = [d * elec_lambda for d in dipole]
        quadrupole = [q * elec_lambda for q in quadrupole]
        polarizability = polarizability * elec_lambda
        multipoleForce.setMultipoleParameters(i, charge, dipole, quadrupole, *default_elec_params[j][3:-1],
                                              polarizability)
        j += 1
    vdwForce.updateParametersInContext(context)
    multipoleForce.updateParametersInContext(context)
    context.reinitialize(preserveState=True)
