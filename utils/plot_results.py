import os
from itertools import cycle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn
import yaml

plt.style.use('ggplot')




def print_failures(res_dir=f'../res/Results'):
    eval = dict()

    # for approach in ['RL']:
    for approach in ['Oracle']:
        res = pd.DataFrame()
        for run in sorted([r for r in os.listdir(f'{res_dir}/{approach}/test')
                           if '.DS_Store' != r], key=lambda x: int(x.replace('run', ''))):
            run_res = pd.read_excel(os.path.join(f'{res_dir}/{approach}/test', run, 'metrics.xlsx'))
            run_res = run_res[run_res['Steps'] == run_res['Steps'].max()]

            # Ensure only last element is retrieved (e.g. when the agent calls the stop action immediately)
            run_res = run_res.iloc[len(run_res) - 1]
            res = pd.concat([res, pd.DataFrame.from_records([run_res])])

        for method in set(res['Method']):

            if approach == 'Oracle':
                eval['Oracle'] = res[res['Method'] == method]
            else:
                eval[method] = res[res['Method'] == method]

    print(res[res['Objects'] == 4]['Distance to success'].mean())




def check_logs(res_dir=f'../res/MNISTExib-v0'):

    approach = 'Oracle'
    failures = {'steps': 0, 'objects': 0, 'states': 0, 'early_stop': 0}
    for run in sorted([r for r in os.listdir(f'{res_dir}/{approach}/test')
                       if '.DS_Store' != r], key=lambda x: int(x.replace('run', ''))):

        with open(os.path.join(f'{res_dir}/{approach}/test', run, 'log'), 'r') as f:
            run_log = f.readlines()
            if ((int(run[3:]) % 100) // 5) >= 16:  # 10 objects
                if 'Success rate:0.0' in ' '.join(run_log):
                    # too many steps
                    if len(run_log) == 107:
                        failures['steps'] += 1
                    elif 'stop()' in ' '.join(run_log):
                        failures['early_stop'] += 1
                    else:
                        print('Unidentified failure')
                        breakpoint()


def print_env_stats():
    count = 0
    maxd = 0
    all_und = []

    with open(f'../datasets/fine-tune/train/MNISTExib-v0.yaml') as f:
        train_configs = yaml.safe_load(f)


    for env_id, env in train_configs.items():
        typedigits = [d.split('-')[0] for d in env['_state']['objects']]
        if len(set(typedigits)) < len(env['_state']['objects']):
            count += 1
            all_und.append(len(env['_state']['objects']) - len(set(typedigits)))
            maxd = max(maxd, len(env['_state']['objects']) - len(set(typedigits)))

    print(f"#Envs with at least a pair of indistinguishable objects: {count}.")



def plot_rlvspal(metrics, res_dir=f'../res/Results', xlabel='Objects'):
    eval = dict()
    styles = cycle(['--', '-', ':', '-.', '--'])
    colors = cycle(['blue', 'green', 'purple', 'orange', 'brown'])
    palette = seaborn.color_palette("Paired")
    color_map = {
        "blue": palette[1],  # Default "blue" color in the pastel palette
        "orange": palette[7],  # Default "orange" color
        "green": palette[3],
        "purple": palette[9],
        "brown": palette[11],
    }

    # for approach in ['OLOM', 'Oracle', 'PPO', 'IMPALA', 'DRQN']:
    for approach in os.listdir(res_dir):
        res = pd.DataFrame()
        for run in sorted([r for r in os.listdir(f'{res_dir}/{approach}/test')
                           if '.DS_Store' != r], key=lambda x: int(x.replace('run', ''))):
            run_res = pd.read_excel(os.path.join(f'{res_dir}/{approach}/test', run, 'metrics.xlsx'))
            run_res = run_res[run_res['Steps'] == run_res['Steps'].max()]

            # Ensure only last element is retrieved (e.g. when the agent calls the stop action immediately)
            run_res = run_res.iloc[len(run_res) - 1]
            res = pd.concat([res, pd.DataFrame.from_records([run_res])])

        for method in sorted(list(set(res['Method']))):

            if approach.lower() == 'oracle':
                eval[approach] = res[res['Method'] == method]
            else:
                eval[method] = res[res['Method'] == method]

    # Plot metrics
    for metric in metrics:
        for method, res in eval.items():
            x = sorted(list(set(res[xlabel])))
            y_avg = [res[res[xlabel] == nobjs][metric].mean() for nobjs in x]
            y_var = np.array([res[res[xlabel] == nobjs][metric].std()**2 for nobjs in x])
            y_range = y_var
            if metric == 'Distance to success':
                y_range = np.sqrt(y_range)  # Use standard deviation

            c = color_map[next(colors)]

            if method == 'POUCT':
                method = 'OLOM'
            elif method == 'PPO':
                method = 'RPPO'

            plt.plot(x, y_avg, label=method, linestyle=next(styles), color=c, alpha=.8)

            # # Plot the standard deviation as a shaded area
            plt.fill_between(x,
                             [m - s for m, s in zip(y_avg, y_range)],
                             [m + s for m, s in zip(y_avg, y_range)],
                             color=c,
                             alpha=0.2)

        if metric in ['Success rate', 'SPL']:
            plt.ylim([-0.01, 1.01])
        if metric in ['Distance to success']:
            plt.ylim(bottom=-0.1)

        if metric == 'Success rate':
            plt.ylabel("SR")
        elif metric == 'Distance to success':
            plt.ylabel("DTS")
        elif metric == 'SPL':
            plt.ylabel("SPL")
        else:
            raise NotImplementedError

        plt.legend()
        plt.xlabel("#Objects")
        plt.savefig(f'{res_dir}/{metric}.png')
        plt.close()


def print_time(res_dir=f'../resOK/MNISTExib-v0'):
    approach = 'DRQN'
    res = pd.DataFrame()
    for run in sorted([r for r in os.listdir(f'{res_dir}/{approach}/test')
                       if '.DS_Store' != r], key=lambda x: int(x.replace('run', ''))):
        run_res = pd.read_excel(os.path.join(f'{res_dir}/{approach}/test', run, 'metrics.xlsx'))
        run_res = run_res[run_res['Steps'] == run_res['Steps'].max()]

        # Ensure only last element is retrieved (e.g. when the agent calls the stop action immediately)
        run_res = run_res.iloc[len(run_res) - 1]
        res = pd.concat([res, pd.DataFrame.from_records([run_res])])

    print(res['Time seconds'].mean())


def print_objs_time(res_dir):
    eval = dict()

    for approach in ['OLOM']:
        res = pd.DataFrame()
        for run in sorted([r for r in os.listdir(f'{res_dir}/{approach}/train')
                           if '.DS_Store' != r], key=lambda x: int(x.replace('run', ''))):

            if os.path.exists(os.path.join(f'{res_dir}/{approach}/train', run, 'metrics.xlsx')):
                run_res = pd.read_excel(os.path.join(f'{res_dir}/{approach}/train', run, 'metrics.xlsx'))

                # Ensure only last element is retrieved (e.g. when the agent calls the stop action immediately)
                run_res = run_res.iloc[len(run_res) - 1]
                res = pd.concat([res, pd.DataFrame.from_records([run_res])])

    for i in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
        print(f"\n\n# Objects: {i}")
        print(f"Time: {res[res['Number of environment objects'] == i]['Time seconds'].mean():.2f}")
        print(f"Memory (MB): {res[res['Number of environment objects'] == i]['Memory usage (MB)'].mean():.2f}")


def print_gap_tmp(res_dir):
    eval = dict()

    for approach in ['OLOM']:
        res = pd.DataFrame()
        for run in sorted([r for r in os.listdir(f'{res_dir}/{approach}/test')
                           if '.DS_Store' != r], key=lambda x: int(x.replace('run', ''))):

            if os.path.exists(os.path.join(f'{res_dir}/{approach}/test', run, 'metrics.xlsx')):
                run_res = pd.read_excel(os.path.join(f'{res_dir}/{approach}/test', run, 'metrics.xlsx'))

                # Ensure only last element is retrieved (e.g. when the agent calls the stop action immediately)
                run_res = run_res.iloc[len(run_res) - 1]
                res = pd.concat([res, pd.DataFrame.from_records([run_res])])

    print(f"Avg success rate: {res['Success rate'].mean():.2f}")
    print(f"Avg SPL: {res['SPL'].mean():.2f}")
    print(f"Avg DTS: {res['Distance to success'].mean():.2f}")

if __name__ == "__main__":
    SMALL_SIZE = 8
    MEDIUM_SIZE = 10
    BIG_SIZE = 12
    XL_SIZE = 15
    XXL_SIZE = 17

    plt.style.use('ggplot')
    plt.rc('font', size=BIG_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=BIG_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=XL_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=BIG_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=BIG_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=BIG_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIG_SIZE)  # fontsize of the figure title

    # plot_rlvspal(metrics=['Success rate', 'Distance to success', 'SPL'],
    #              res_dir=f'../res/simpleMNISTExib-v0',
    #              xlabel='Objects')

    # print_env_stats()

    # check_logs('../res/MNISTExib-v0')

    # print_failures('../res/MNISTExib-v0')

    # print_gap_tmp(res_dir=f'../res/simpleMNISTExib-v0_NIPS')
    # print_gap_tmp(res_dir=f'../res/simpleMNISTExib-v0_SAMEDIGITERRORS')

    print_gap_tmp(res_dir=f'../res/MNISTExib-v0_NIPS')
    print_gap_tmp(res_dir=f'../res/MNISTExib-v0_SAMEDIGITERRORS')


