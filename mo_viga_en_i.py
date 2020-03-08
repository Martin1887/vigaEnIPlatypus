#!/usr/bin/python3
from platypus import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from multiprocessing import cpu_count


class VigaEnI(Problem):
    # Constants of the problem
    E = 20000.0
    sigma_a = 16.0

    def I(self):
        return ((self.x3 * (self.x1 - 2 * self.x4)) ** 3) / \
            12 + 2 * self.x2 * self.x4 * \
            (4 * self.x4 ** 2 + 3 * self.x1 * (self.x1 - 2 * self.x4))

    def M_y(self):
        return (self.P / 2) * (self.L / 2)

    def M_z(self):
        return (self.Q / 2) * (self.L / 2)

    def z_y(self):
        return (1 / 6 * self.x1) * (self.x3 * (self.x1 - self.x4) ** 3 +
                                    2 * self.x2 * self.x4 *
                                    (4 * self.x4 ** 2 + 3 * self.x1 * (self.x1 - 2 * self.x4)))

    def z_z(self):
        return (1 / 6 * self.x2) * (self.x1 - self.x4) * self.x3 ** 3 + \
            2 * self.x4 * self.x2 ** 3

    # P = 600 kN, Q = 50 kN, L=200 cm in the specification
    def __init__(self, P=600, Q=500, L=200):
        self.P = P
        self.Q = Q
        self.L = L
        # 4 decission variables, 2 objective functions, 1 restriction
        # (in addition of range values of decission variables)
        super(VigaEnI, self).__init__(4, 2, 1)
        self.types[:] = [Real(10, 80), Real(
            10, 50), Real(0.9, 5), Real(0.9, 5)]
        self.constraints[:] = ">=0"

    def evaluate(self, solution):
        self.x1 = solution.variables[0]
        self.x2 = solution.variables[1]
        self.x3 = solution.variables[2]
        self.x4 = solution.variables[3]
        solution.objectives[:] = [
            2 * self.x2 * self.x4 + self.x3 * (self.x1 - 2 * self.x4),
            (self.P * self.L ** 3) / (48 * self.E * self.I())
        ]
        solution.constraints[:] = [
            self.sigma_a - ((self.M_y() / self.z_y()) +
                            (self.M_z() / self.z_z()))
        ]


if __name__ == "__main__":
    algorithms = [NSGAII,
                  (NSGAIII, {"divisions_outer": 4}, 'NSGAIII_4'),
                  (NSGAIII, {"divisions_outer": 12}, 'NSGAIII_12'),
                  (NSGAIII, {"divisions_outer": 24}, 'NSGAIII_24'),
                  (CMAES, {"epsilons": [0.05]}),
                  GDE3,
                  IBEA,
                  (MOEAD, {"weight_generator": normal_boundary_weights,
                           "divisions_outer": 4}, 'MOEAD_4'),
                  (MOEAD, {"weight_generator": normal_boundary_weights,
                           "divisions_outer": 12}, 'MOEAD_12'),
                  (MOEAD, {"weight_generator": normal_boundary_weights,
                           "divisions_outer": 24}, 'MOEAD_24'),
                  (OMOPSO, {"epsilons": [0.05]}),
                  SMPSO,
                  SPEA2,
                  (EpsMOEA, {"epsilons": [0.05]})]
    problems = [VigaEnI()]

    with ProcessPoolEvaluator(cpu_count()) as evaluator:
        results = experiment(algorithms, problems,
                             nfe=10000, evaluator=evaluator)

    # Pareto front per algorithm in 2D
    fig = plt.figure()
    for i, algorithm in enumerate(six.iterkeys(results)):
        result = results[algorithm]["VigaEnI"][0]

        ax = fig.add_subplot(2, 7, i+1)
        ax.set_axisbelow(True)
        ax.grid()
        ax.scatter([s.objectives[0] for s in result],
                   [s.objectives[1] for s in result])

        ax.set_xlabel('área (cm2)')
        ax.set_ylabel('deflexión (cm)')
        ax.set_title(algorithm)

    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10,
                        right=0.95, hspace=0.25, wspace=0.35)
    plt.show()

    # all results
    fig = plt.figure()
    plt.grid()

    point_type = ['r.', 'bx', 'g^', 'cs', 'm+', 'k1', 'vy',
                  'y.', 'rx', 'gv', 'ks', 'g+', 'r1', 'vb']
    for i, algorithm in enumerate(six.iterkeys(results)):
        result = results[algorithm]["VigaEnI"][0]

        plt.plot([s.objectives[0] for s in result],
                 [s.objectives[1] for s in result], point_type[i], label=algorithm)

    plt.xlabel('área (cm2)')
    plt.ylabel('deflexión (cm)')
    plt.title('Todos los algoritmos')
    plt.legend(loc='upper right')
    for s in result:
        label = 'x1={}, x2={}, x3={}, x4={}'\
            .format(s.variables[0], s.variables[1], s.variables[2], s.variables[3])
        plt.annotate(label, (s.objectives[0], s.objectives[1]))

    plt.show()
