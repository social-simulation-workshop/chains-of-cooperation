import argparse
import csv
import copy
import itertools
import os
import math
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
from scipy.stats import truncnorm

def get_truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


class ArgsModel(object):
    
    def __init__(self) -> None:
        super().__init__()
    
        self.parser = argparse.ArgumentParser()
        self.parser = self.add_decision_algo_param(self.parser)
        self.parser = self.add_production_algo_param(self.parser)
        self.parser = self.add_learning_algo_param(self.parser)
        self.parser = self.add_exp_param(self.parser)

    @staticmethod
    def add_decision_algo_param(parser: argparse.ArgumentParser):
        parser.add_argument("--M", type=float, default=5.0,
            help="slope parameter.")
        parser.add_argument("--thres_type", type=str, default="all_defector",
            help="type of distribution for agents' thresholds. \
                \"all_defector\" for all set to 1. \
                \"uni\" for uniformly or \"nor\" for normally distributed.")
        parser.add_argument("--net_group", type=str, default="parallel",
            help="the size of reference group. parallel, strong, weak, or serial.")
        parser.add_argument("--R_type", type=str, default="normal",
            help="type of distribution for resources. equal, normal, or lognormal.")
        parser.add_argument("--div", type=str2bool, nargs="?", const=True, default=False,
            help="divisible contribution for agents if set.")
        return parser
        
    @staticmethod
    def add_production_algo_param(parser: argparse.ArgumentParser):
        parser.add_argument("--X", type=float, default=0.,
            help="the const that determined the range of L. X \in \[-1, 1\]. \
                X = -1 for -1 <= L <= 0; \
                X = 1 for 0 <= L <= 1;")
        parser.add_argument("--J", type=float, default=0.5,
            help="jointness of supply. J \in \[0, 1\].")
        
        # distribution of I_j is not disccussed in the paper.
        parser.add_argument("--I_type", type=str, default="normal",
            help="normal distribution for agent's interest in public goods with mean=1.")
        parser.add_argument("--corr_RI", type=str, default="orthogonal",
            help="correlation between the distribution of resource and interests. \
                pos for r_RI = 1; neg for r_RI=1; orthogonal for independent dis.")
        return parser
    
    @staticmethod
    def add_learning_algo_param(parser: argparse.ArgumentParser):
        parser.add_argument("--E_type", type=str, default="normal",
            help="normal distribution for agent's learning rate parameter.")
        return parser

    @staticmethod
    def add_exp_param(parser: argparse.ArgumentParser):
        parser.add_argument("--N", type=int, default=100,
            help="# of agent.")
        parser.add_argument("--n_iter", type=int, default=200,
            help="# of iterations.")
        parser.add_argument("--figNo", type=int, default=1,
            help="the figure to replicate. 0 for custom parameters.")
        parser.add_argument("--seed", type=int, default=10455,
            help="random seed.")
        return parser

    @staticmethod
    def set_fig_param(args, figNo) -> dict:
        """ set essential parameters for each experiments """
        args.figNo = figNo
        if figNo == 1:
            args.N = 1000
            args.thres_type = "uni"
            args.div = False
            args.X = 0.
            args.J = 0.5

            args.R_type = "normal"
            args.I_type = "normal"
            args.E_type = "normal"
            args.corr_RI = "orthogonal"

            args1 = copy.deepcopy(args)
            args2 = args
            args1.net_group = "parallel"
            args2.net_group = "serial"
            return {"Parallel Choice": args1, "Serial Choice": args2}


    def get_args(self) -> dict:
        args = self.parser.parse_args()
        if args.figNo == 0:
            return {"custom": args}
        else:
            return self.set_fig_param(args, args.figNo)
    
    
    def get_fig_args(self, figNo:int) -> dict:
        args = self.parser.parse_args()
        return self.set_fig_param(args, figNo)


class Agent(object):
    _ids = itertools.count(0)

    def __init__(self, args) -> None:
        super().__init__()
        
        self.id = next(self._ids)

        self.thres = None
        self.M = args.M
        self.net = None
        self.R = None
        self.I = None
        self.E = None

        self.S = 0
        self.O = 0
        self.is_volunteer = False
    
    @staticmethod
    def draw(p):
        return True if np.random.uniform() < p else False
    
    def set_param(self, thres, R, I, E):
        self.thres, self.R, self.I, self.E = thres, R, I, E
    
    def to_volunteer(self, pi) -> None:
        #print("agent {} | pi = {:4f}".format(("  "+str(self.id))[-3:], pi))
        if self.thres == 1 and pi == 0:
            self.is_volunteer = False
        elif self.thres == 0 and pi == 1:
            self.is_volunteer = True
        else:
            p = 1/(1+math.exp(self.M * (self.thres - pi)))
            self.is_volunteer = self.draw(p)
    
    def get_net_pi(self):
        if self.net is None:
            raise TypeError("haven't initialized social net for the agent.")
        return sum([1 for ag in self.net if ag.is_volunteer]) / len(self.net)
    

class PublicGoodsGame(object):

    def __init__(self, args: argparse.ArgumentParser) -> None:
        super().__init__()
        Agent._ids = itertools.count(0)

        self.args = args
        print("Args: {}".format(args))

        self.ags = self.init_ags(args)
        self.global_pi = self._get_global_pi()

        self.total_R = sum([ag.R for ag in self.ags])
        self.global_R_ratio = self._get_global_R_ratio()

        self.R_ratio_list = list()
        self.R_ratio_list.append(self.global_R_ratio)

        self.L = 0
    
    @staticmethod
    def get_thres(thres_type: str, mean=0.5, sd=0.1):
        if thres_type == "all_defector":
            return 1.
        elif thres_type == "uni":
            return np.random.uniform()
        elif thres_type == "nor":
            return get_truncated_normal(mean=mean, sd=sd, low=0, upp=1).rvs()

    @staticmethod
    def get_RI(R_type, I_type, corr_RI, mean=1, sd=0.5):
        if R_type == "normal" and I_type == "normal":
            x1 = get_truncated_normal(mean=mean, sd=sd, low=0, upp=np.inf).rvs()
            x2 = get_truncated_normal(mean=mean, sd=sd, low=0, upp=np.inf).rvs()
            if corr_RI == "orthogonal":
                return float(x1), float(x2)
            elif corr_RI == "pos":
                return float(x1), float(x1)
            elif corr_RI == "neg":
                return float(x1), -float(x1)
        elif R_type == "equal" and I_type == "normal":
            x1 = get_truncated_normal(mean=mean, sd=sd, low=0, upp=np.inf).rvs()
            return 1., float(x1)
        else:
            raise TypeError("haven't coded distributions other than normal.")
    
    @staticmethod
    def get_E(E_type, mean=0.5, sd=0.1):
        if E_type == "normal":
            return get_truncated_normal(mean=mean, sd=sd, low=0, upp=1).rvs()
    
    @staticmethod
    def check_distribution(ags):
        R = np.array([ag.R for ag in ags])
        I = np.array([ag.I for ag in ags])
        E = np.array([ag.E for ag in ags])
        print("R: mean={:5f}; sd={:5f}".format(np.mean(R), np.std(R)))
        print("I: mean={:5f}; sd={:5f}".format(np.mean(I), np.std(I)))
        print("E: mean={:5f}; sd={:5f}".format(np.mean(E), np.std(E)))

    def init_ags(self, args: argparse.ArgumentParser) -> list:
        ags = list()
        for _ in range(args.N):
            ag = Agent(args)
            thres = self.get_thres(args.thres_type)
            R, I = self.get_RI(args.R_type, args.I_type, args.corr_RI)
            E = self.get_E(args.E_type)
            ag.set_param(thres, R, I, E)
            ags.append(ag)
        self.check_distribution(ags)
        return ags   

    def _get_global_pi(self):
        return sum([1 for ag in self.ags if ag.is_volunteer]) / self.args.N
    
    def _get_global_R_ratio(self):
        return sum([ag.R for ag in self.ags if ag.is_volunteer]) / self.total_R
    
    def get_ag_pi(self, ag:Agent):
        # paper: pi is "the participation rate"
        if self.args.net_group == "parallel":
            return 0.5
        elif self.args.net_group == "serial":
            return self.global_pi
        elif self.args.net_group in {"strong", "weak"}:
            return ag.get_net_pi()

    def simulate_iter(self, epsilon=10**-6):
        """ At iteration i:
        1. determine status of every agent based on pi_{i-1}.
        2. update global_pi and global_R_ratio
        3. calculate L; and find S_ij and O_ij for every agent j
        4. update T_ij for every agent j
        """
        # 1.
        for ag in self.ags:
            ag.to_volunteer(pi=self.get_ag_pi(ag))
        
        # 2.
        self.global_pi = self._get_global_pi()
        self.global_R_ratio = self._get_global_R_ratio()

        # 3.
        # paper: pi is "the rate of contribution"
        self.L = 1/(1+math.exp(10*(0.5-self.global_R_ratio))) - (1-self.args.X)/2
        S_max = -np.inf
        for ag in self.ags:
            S_ij = self.L*self.total_R*ag.I/(self.total_R**(1-self.args.J)) - int(ag.is_volunteer)*ag.R
            O_ij = ag.E * (2*S_ij - ag.S)
            S_max = max(S_max, abs(S_ij))
            ag.S = S_ij
            ag.O = O_ij
        
        # 4.
        for ag in self.ags:
            ag.O = ag.O / (3*S_max)
            ag.O += epsilon
            
            t_drop = ag.O * (1 - (1-ag.thres)**(1/abs(ag.O)))
            t_increase = ag.O * (1 - ag.thres**(1/abs(ag.O)))
            t_drop = 0 if np.isnan(t_drop) else t_drop
            t_increase = 0 if np.isnan(t_increase) else t_increase

            # threshold drops
            if ag.is_volunteer and ag.O>0:
                ag.thres -= t_drop
            elif not ag.is_volunteer and ag.O<0:
                ag.thres += t_drop
            # threshold increases
            elif ag.is_volunteer and ag.O<0:
                ag.thres -= t_increase
            elif not ag.is_volunteer and ag.O>0:
                ag.thres += t_increase
            
            ag.thres = min(ag.thres, 1.)
            ag.thres = max(ag.thres, 0.)

    def simulate(self, log_v=1):
        print("| iter   0 | pi = {:.4f}; R = {:.4f}; L = {:.4f}".format(self.global_pi, self.global_R_ratio, self.L))
        for iter in range(1, self.args.n_iter+1):
            self.simulate_iter()
            self.R_ratio_list.append(self.global_R_ratio) # pi: the rate of contribution
            if iter % log_v == 0:
                print("| iter {} | pi = {:.4f}; R = {:.4f}; L = {:.4f}".format(("  "+str(iter))[-3:], self.global_pi, self.global_R_ratio, self.L))
    
    def get_pi_list(self):
        return np.array(self.R_ratio_list)


class PlotLinesHandler(object):

    def __init__(self, xlabel, ylabel, ylabel_show,
        figure_size=(9, 9), output_dir=os.path.join(os.getcwd(), "imgfiles")) -> None:
        super().__init__()

        self.output_dir = output_dir
        self.title = "{}-{}".format(ylabel, xlabel)
        self.legend_list = list()

        plt.figure(figsize=figure_size, dpi=80)
        plt.title("{} - {}".format(ylabel_show, xlabel))
        plt.xlabel(xlabel)
        plt.ylabel(ylabel_show)
        ax = plt.gca()
        ax.set_ylim([0., 1.])

    def plot_line(self, data, legend, linewidth=1):
        self.legend_list.append(legend)
        plt.plot(np.arange(data.shape[-1]), data, linewidth=linewidth)

    def save_fig(self, title_param=""):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        plt.legend(self.legend_list)
        title_lg = "_".join(self.legend_list)
        fn = "_".join([self.title, title_lg, title_param]) + ".png"
        plt.savefig(os.path.join(self.output_dir, fn))
        print("fig save to {}".format(os.path.join(self.output_dir, fn)))




if __name__ == "__main__":
    parser = ArgsModel()
    
    ## figure 1
    args_dict = parser.get_fig_args(1)
    plot_line_hd = PlotLinesHandler(xlabel="Iteration", ylabel="pi",
                                    ylabel_show="Level of Contribution "+r"$\pi$")
    for exp_legend, exp_args in args_dict.items():
        np.random.seed(seed=exp_args.seed)
        game = PublicGoodsGame(exp_args)
        game.simulate()
        plot_line_hd.plot_line(game.get_pi_list(), exp_legend)
        param = "N_{}_T_{}".format(exp_args.N, exp_args.thres_type)
    plot_line_hd.save_fig(param)