import matplotlib.pyplot as plt
import numpy as np
import logging
from optimizer import Optimizer

logging.basicConfig(filename='output/run.log', level=logging.INFO)
logger = logging.getLogger("Optimizer")

get_next_pose = lambda initial, change: np.array(
    [
        initial[0] + change[0] * np.cos(initial[2]) - change[1] * np.sin(
            initial[2]),
        initial[1] + change[1] * np.cos(initial[2]) + change[0] * np.sin(
            initial[2]),
        initial[2] + change[2],
    ]
)

initial_poses = []
od_constraints = []
loop_constraints = []

data_file = open("../data/noisy.g2o")

for line in data_file.readlines():
    params = line[:-1].split(" ")

    if len(params) < 5:
        continue

    if params[0] == "VERTEX_SE2":
        initial_poses.append(
            np.array([float(params[2]), float(params[3]), float(params[4])]))
        # n * 3
    else:
        pose_start = int(params[1])
        pose_end = int(params[2])

        formatted_params = [
            [
                pose_start,
                pose_end,
            ],
            [
                float(params[3]),
                float(params[4]),
                float(params[5]),
            ],
        ]

        if pose_end == pose_start + 1:
            od_constraints.append(formatted_params)
        else:
            loop_constraints.append(formatted_params)

initial_poses = np.array(initial_poses)

constraints = []
constraints.extend(od_constraints)
constraints.extend(loop_constraints)

n_poses = initial_poses.shape[0]
n_constraints = len(constraints)


def f(state):
    poses = state.reshape(-1, 3)
    fixed_pose = initial_poses[0]
    f_array = []

    for constraint in constraints:
        pose_start = constraint[0][0]
        pose_end = constraint[0][1]

        diff = get_next_pose(poses[pose_start], constraint[1]) - poses[pose_end]
        f_array.extend(diff)

    # initial position
    f_array.extend(poses[0] - fixed_pose)

    f_array = np.array([f_array]).T

    return f_array


f_initial = f(initial_poses)

state_vector = initial_poses.reshape(-1, 1)

j_shape = (f_initial.shape[0], state_vector.shape[0])


def jac(state: np.ndarray):
    J = np.zeros(j_shape)
    for i, constraint in enumerate(constraints):
        pose_start = constraint[0][0]
        pose_end = constraint[0][1]
        theta_start = state[3 * pose_start + 2]

        # x constraint
        J[3 * i, 3 * pose_start] = 1
        J[3 * i, 3 * pose_end] = -1
        J[3 * i, 3 * pose_start + 2] = -constraint[1][0] * np.sin(theta_start) - \
                                       constraint[1][1] * np.cos(theta_start)

        # y constraint
        J[3 * i + 1, 3 * pose_start + 1] = 1
        J[3 * i + 1, 3 * pose_end + 1] = -1
        J[3 * i + 1, 3 * pose_start + 2] = -constraint[1][1] * np.sin(
            theta_start) + constraint[1][0] * np.cos(theta_start)

        # theta constraint
        J[3 * i + 2, 3 * pose_start + 2] = 1
        J[3 * i + 2, 3 * pose_end + 2] = -1

    # fixed poses
    J[-3, 0] = 1
    J[-2, 1] = 1
    J[-1, 2] = 1
    return J


J = jac(state_vector)


def draw(X, Y, THETA, name):
    ax = plt.subplot(111)
    ax.plot(X, Y, 'ro')
    plt.plot(X, Y, 'c-')

    for i in range(len(THETA)):
        x2 = 0.25 * np.cos(THETA[i]) + X[i]
        y2 = 0.25 * np.sin(THETA[i]) + Y[i]
        plt.plot([X[i], x2], [Y[i], y2], 'g->')

    plt.savefig(name)


def main():
    info_mat = np.eye(f_initial.shape[0])
    values = [(500, 700, 1000)]
    for od_info, loop_info, fix_info in values:
        plt.clf()
        n_od = 3 * len(od_constraints)
        n_loop = 3 * len(loop_constraints)

        info_mat[:n_od, :n_od] = od_info * np.eye(n_od)
        info_mat[n_od:n_od + n_loop, n_od:n_od + n_loop] = loop_info * np.eye(
            n_loop)
        info_mat[-3:, -3:] = fix_info * np.eye(3)
        state = state_vector.copy()
        solver = Optimizer(f, jac, state, info_mat, algo='LM', lm_lambda=1e-3,
                           n_iter=1000)
        solver.optimize()

        plt.plot(solver.losses)
        plt.savefig(f"output/{fix_info}_{loop_info}_{od_info}_loss.png")
        plt.clf()

        final_vector = solver.get_current()

        poses = final_vector.reshape(-1, 3)
        draw(poses[:, 0], poses[:, 1], poses[:, 2],
             f"output/{fix_info}_{loop_info}_{od_info}_path.png")
        plt.clf()

        logger.info(f"{fix_info}_{loop_info}_{od_info}: {solver.loss()}")


if __name__ == '__main__':
    main()
