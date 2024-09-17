ex: 
output1: python BindingFreeEnergy.py --pdb_file temoa_g3-15-0000-0000.pdb --forcefield_file hostsG3.xml --nonbonded_method NoCutoff --vdw_lambda 0 --elec_lambda 0 --alchemical_atoms "0,2" --num_steps 30000 
output2: python BindingFreeEnergy.py --pdb_file temoa_g3-15-0000-0000.pdb --forcefield_file hostsG3.xml --nonbonded_method NoCutoff --vdw_lambda 0.4 --elec_lambda 0 --alchemical_atoms "0,2" --num_steps 30000

python BAR.py --traj_i output1.dcd --traj_ip1 output2.dcd --pdb_file temoa_g3-15-0000-0000.pdb --forcefield_file hostsG3.xml  --vdw_lambda_i 0 --elec_lambda_i 0 --vdw_lambda_ip1 0.4 --elec_lambda_ip1 0 --alchemical_atoms "0,2" --nonbonded_method NoCutoff
