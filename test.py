from rewards.system import RewardingSystem
s = RewardingSystem()
mols = [['CCH','CCOCCOCC'],['CCOCCOCC','C[C@@H]1CCN(C(=O)CCCc2ccccc2)C[C@@H]1O']]
ops = ['qed','logp','similarity','donor','sa','reduction_potential','smarts_filter','logs']

print(s.evaluate(ops,mols))