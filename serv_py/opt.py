import random,array
from timeit import default_timer as timer
from deap import base, creator, tools
import sys
eps=sys.float_info.epsilon
class opt_toobox():
    def checkBounds(self,min, max):
        def decorator(func):
            def wrapper(*args, **kargs):
                offspring = func(*args, **kargs)
                for child in offspring:
                    for i in range(len(child)):
                        if child[i] > max[i]:
                            if random.random()<0.8:
                                child[i] = max[i]- ((child[i] - max[i]) % (max[i]-min[i]))
                            else:
                                child[i] = max[i]-eps*random.randint(1,1e10) # eps ~ 1e-16 * e10 -> \us
                        elif child[i] < min[i]:
                            if random.random()<0.8:
                                child[i] = min[i]+ (( min[i]-child[i]) % (max[i]-min[i]))
                            else:
                                child[i] = min[i]+eps*random.randint(1,1e10) # eps ~ 1e-16 * e10 -> \us                            
                return offspring
            return wrapper
        return decorator
        
    def __init__(self,s_n):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        # creator.create("Individual", array.array, typecode="d", fitness=creator.FitnessMin)
        creator.create("Individual", list, fitness=creator.FitnessMin)
        
        self.toolbox = base.Toolbox()
        FLT_MIN_E, FLT_MAX_E = 0, 1
        FLT_MIN_K, FLT_MAX_K = 0, 100000
        FLT_MIN_V, FLT_MAX_V = 0, 100
        self.s_n = s_n
        self.toolbox.register("attr_flt", random.random)
        self.toolbox.register("attr_flt_k", random.uniform, FLT_MIN_K, FLT_MAX_K)
        self.toolbox.register("attr_flt_v", random.uniform, FLT_MIN_V, FLT_MAX_V)
        ind_type=()
        ind_range_max=()
        ind_range_min=()
        for _ in range(s_n):
            ind_type=ind_type+(self.toolbox.attr_flt,)
            ind_range_max=ind_range_max+(FLT_MAX_E,)
            ind_range_min=ind_range_min+(FLT_MIN_E,)
        for _ in range(s_n*(s_n-1)):
            ind_type=ind_type+(self.toolbox.attr_flt_k,)
            ind_range_max=ind_range_max+(FLT_MAX_K,)
            ind_range_min=ind_range_min+(FLT_MIN_K,)
        for _ in range(s_n):
            ind_type=ind_type+(self.toolbox.attr_flt_v,)      
            ind_range_max=ind_range_max+(FLT_MAX_V,) 
            ind_range_min=ind_range_min+(FLT_MIN_V,)
        # ind_type=(toolbox.attr_flt,toolbox.attr_flt,toolbox.attr_flt_k,toolbox.attr_flt_k, toolbox.attr_flt_v)
        self.toolbox.register("individual", tools.initCycle, creator.Individual,
                        ind_type, n=1)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
        self.toolbox.decorate("mate", self.checkBounds(ind_range_min, ind_range_max))
        self.toolbox.decorate("mutate", self.checkBounds(ind_range_min, ind_range_max))
    def run(self,stopflag,q,ind_num=0,NGEN=500,CXPB=0.35,MUTPB=0.4):
        self.ind_num=ind_num
        qO,qN=q
        if ind_num==0:
            self.ind_num=20*self.s_n*(self.s_n+1)
        running=True
        pop=self.toolbox.population(n=self.ind_num)
        for gen in range(NGEN):            
            # Select the next generation individuals
            selected = self.toolbox.select(pop, len(pop))
            # Clone the selected individuals
            # offspring = map(self.toolbox.clone, offspring)
            offspring = [self.toolbox.clone(ind) for ind in selected]
            # Apply crossover on the offspring
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < CXPB:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values
            # Apply mutation on the offspring
            for mutant in offspring:
                if random.random() < MUTPB:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values
            # # Evaluate the individuals with an invalid fitness
            # for oi in offspring:
            #     print(gen,"son send: ",oi.fitness.values)
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            count=len(invalid_ind)
            print("sent count: ",count)
            for i in range(count):
                if stopflag.value>=1:
                    running=False
                    break
                qO.put((i,invalid_ind[i]))
                # pop_dict[(gen,i)]=invalid_ind[i]
            # The population is entirely replaced by the offspring
            count_r=0
            for _ in range(count):
                # if stopflag.value>=1:
                #     running=False
                #     break
                ii , ind_fit = qN.get()
                count_r=count_r+1
                # print("recv idx: ",ii, " tot_recv: ",count_r," ind_fit: ",ind_fit) 
                invalid_ind[ii].fitness.values=(ind_fit,)
            
            if not running or stopflag.value>=1:
                break
            pop[:] = offspring
            bestf=9999999999999.9
            for oi in offspring:
                print(gen, " oi.fitness.values",oi.fitness.values)
                if (oi.fitness.valid):
                    if (oi.fitness.values[0])<bestf:
                        bestf=(oi.fitness.values[0])
            print(gen," , best: ", bestf)
        # connOpt.close()
        # print("connOpt.close()")

if __name__ == '__main__':
    from multiprocessing import Process, Value, Pipe
    otb=opt_toobox(2)
    connO, connI = Pipe()
    otb_p = Process(target=otb.run, args=((connO, connI),3,6))
    otb_p.start()
    connI.close()
    
    while 1:
        try:    
            i,_=connO.recv()
            connO.send((i,23.32))
        except EOFError as e:
            print(e)
            break
