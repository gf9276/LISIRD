import os
import pathlib
from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt
import matplotlib
import re

# matplotlib.rc("font", family='SimSun', weight='normal')


plt.rcParams['font.sans-serif'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False


def find_file(file, suffix):
    return list(pathlib.Path(file).rglob('*.' + suffix))


if __name__ == '__main__':
    # pathlib 挺好用的 主要是路径/和\统一了 好看。。。
    tensorboard_files = find_file(r'E:\fuckpython\kp2d\records\complete_data\tensorboardX', 'amax')  # 不是字符串
    # tensorboard_files = list(map(str, tensorboard_files))
    ea_dict = {}
    for file_name in tensorboard_files:
        ea = event_accumulator.EventAccumulator(str(file_name))
        ea.Reload()
        ea_dict[file_name.parent.name] = ea
        del ea
    for key in ea_dict.keys():
        scalars_keys = ea_dict[key].scalars.Keys()
        break

    for scalars_key in scalars_keys:
        fig = plt.figure(figsize=(6, 4))
        ax1 = fig.add_subplot(111)
        for key in ea_dict.keys():
            data = ea_dict[key].scalars.Items(scalars_key)
            ax1.plot([i.step for i in data], [i.value for i in data], label=key)

        tmp_name = re.sub("_", " ", str(scalars_key)).split()
        tmp_name = ' '.join(tmp_name)
        ax1.set_xlabel("训练轮次")
        ax1.set_ylabel(tmp_name)
        ax1.set_title("训练过程中 " + tmp_name + " 的变化")

        # 还是分开画比较好看
        plt.legend(loc='lower right')
        plt.show()
        tmp_name = re.sub("[^\w]", " ", str(scalars_key)).split()
        tmp_name = '_'.join(tmp_name) + '.svg'
        fig.savefig(r"C:\Users\guof\Desktop\本科毕设\picture\\" + tmp_name)

    print(ea_dict.keys())
