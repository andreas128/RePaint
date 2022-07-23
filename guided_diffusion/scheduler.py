# Copyright (c) 2022 Huawei Technologies Co., Ltd.
# Licensed under CC BY-NC-SA 4.0 (Attribution-NonCommercial-ShareAlike 4.0 International) (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
#
# The code is released for academic research use only. For commercial use, please contact Huawei Technologies Co., Ltd.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This repository was forked from https://github.com/openai/guided-diffusion, which is under the MIT license

def get_schedule(t_T, t_0, n_sample, n_steplength, debug=0):
    if n_steplength > 1:
        if not n_sample > 1:
            raise RuntimeError('n_steplength has no effect if n_sample=1')

    t = t_T
    times = [t]
    while t >= 0:
        t = t - 1
        times.append(t)
        n_steplength_cur = min(n_steplength, t_T - t)

        for _ in range(n_sample - 1):

            for _ in range(n_steplength_cur):
                t = t + 1
                times.append(t)
            for _ in range(n_steplength_cur):
                t = t - 1
                times.append(t)

    _check_times(times, t_0, t_T)

    if debug == 2:
        for x in [list(range(0, 50)), list(range(-1, -50, -1))]:
            _plot_times(x=x, times=[times[i] for i in x])

    return times


def _check_times(times, t_0, t_T):
    # Check end
    assert times[0] > times[1], (times[0], times[1])

    # Check beginning
    assert times[-1] == -1, times[-1]

    # Steplength = 1
    for t_last, t_cur in zip(times[:-1], times[1:]):
        assert abs(t_last - t_cur) == 1, (t_last, t_cur)

    # Value range
    for t in times:
        assert t >= t_0, (t, t_0)
        assert t <= t_T, (t, t_T)


def _plot_times(x, times):
    import matplotlib.pyplot as plt
    plt.plot(x, times)
    plt.show()


def get_schedule_jump(t_T, n_sample, jump_length, jump_n_sample,
                      jump2_length=1, jump2_n_sample=1,
                      jump3_length=1, jump3_n_sample=1,
                      start_resampling=100000000):

    jumps = {}
    for j in range(0, t_T - jump_length, jump_length):
        jumps[j] = jump_n_sample - 1

    jumps2 = {}
    for j in range(0, t_T - jump2_length, jump2_length):
        jumps2[j] = jump2_n_sample - 1

    jumps3 = {}
    for j in range(0, t_T - jump3_length, jump3_length):
        jumps3[j] = jump3_n_sample - 1

    t = t_T
    ts = []

    while t >= 1:
        t = t-1
        ts.append(t)

        if (
            t + 1 < t_T - 1 and
            t <= start_resampling
        ):
            for _ in range(n_sample - 1):
                t = t + 1
                ts.append(t)

                if t >= 0:
                    t = t - 1
                    ts.append(t)

        if (
            jumps3.get(t, 0) > 0 and
            t <= start_resampling - jump3_length
        ):
            jumps3[t] = jumps3[t] - 1
            for _ in range(jump3_length):
                t = t + 1
                ts.append(t)

        if (
            jumps2.get(t, 0) > 0 and
            t <= start_resampling - jump2_length
        ):
            jumps2[t] = jumps2[t] - 1
            for _ in range(jump2_length):
                t = t + 1
                ts.append(t)
            jumps3 = {}
            for j in range(0, t_T - jump3_length, jump3_length):
                jumps3[j] = jump3_n_sample - 1

        if (
            jumps.get(t, 0) > 0 and
            t <= start_resampling - jump_length
        ):
            jumps[t] = jumps[t] - 1
            for _ in range(jump_length):
                t = t + 1
                ts.append(t)
            jumps2 = {}
            for j in range(0, t_T - jump2_length, jump2_length):
                jumps2[j] = jump2_n_sample - 1

            jumps3 = {}
            for j in range(0, t_T - jump3_length, jump3_length):
                jumps3[j] = jump3_n_sample - 1

    ts.append(-1)

    _check_times(ts, -1, t_T)

    return ts


def get_schedule_jump_paper():
    t_T = 250
    jump_length = 10
    jump_n_sample = 10

    jumps = {}
    for j in range(0, t_T - jump_length, jump_length):
        jumps[j] = jump_n_sample - 1

    t = t_T
    ts = []

    while t >= 1:
        t = t-1
        ts.append(t)

        if jumps.get(t, 0) > 0:
            jumps[t] = jumps[t] - 1
            for _ in range(jump_length):
                t = t + 1
                ts.append(t)

    ts.append(-1)

    _check_times(ts, -1, t_T)

    return ts


def get_schedule_jump_test(to_supplement=False):
    ts = get_schedule_jump(t_T=250, n_sample=1,
                           jump_length=10, jump_n_sample=10,
                           jump2_length=1, jump2_n_sample=1,
                           jump3_length=1, jump3_n_sample=1,
                           start_resampling=250)

    import matplotlib.pyplot as plt
    SMALL_SIZE = 8*3
    MEDIUM_SIZE = 10*3
    BIGGER_SIZE = 12*3

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    plt.plot(ts)

    fig = plt.gcf()
    fig.set_size_inches(20, 10)

    ax = plt.gca()
    ax.set_xlabel('Number of Transitions')
    ax.set_ylabel('Diffusion time $t$')

    fig.tight_layout()

    if to_supplement:
        out_path = "/cluster/home/alugmayr/gdiff/paper/supplement/figures/jump_sched.pdf"
        plt.savefig(out_path)

    out_path = "./schedule.png"
    plt.savefig(out_path)
    print(out_path)


def main():
    get_schedule_jump_test()


if __name__ == "__main__":
    main()
