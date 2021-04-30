import cProfile
import DSGRN
import ramp_systems.ramp_system as ramp_system



def run():
    network_spec = "\
        TGFb : (~Ovol2)(~miR200) : E \n \
        Snail1 : (TGFb)(~miR34a) : E \n \
        miR200 : (~Snail1)(~Zeb1) : E \n \
        Ovol2 : (~Zeb1) \n \
        Zeb1 : (Snail1)(~miR200)(~Ovol2) : E \n \
        miR34a : (~Snail1)(~Zeb1) : E \
        "
    Network = DSGRN.Network(network_spec)
    pg = DSGRN.ParameterGraph(Network)
    pindex = 113854109 #parameter which contains 8 stable FPs
    parameter = pg.parameter(pindex)
    sampler = DSGRN.ParameterSampler(Network)
    num2sample = 100
    for i in range(num2sample):
        sample = sampler.sample(parameter)
        RS = ramp_system.get_ramp_system_from_parameter_string(sample,Network)
        hill_coefficients = RS.extreme_hill_coefficients_with_optimal_theta()
        slopes = RS.extreme_slopes_with_optimal_theta()

cProfile.run('run()')
