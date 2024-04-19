import matplotlib.pyplot as plt

font1 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 36,
         }


def read_txt(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    splitlines = [x.strip().split(' ') for x in lines]
    return splitlines


def smooth_loss(weightPath, weight=0.85):
    iters = []
    loss = []
    data = read_txt(weightPath)
    for value in data:
        iters.append(int(value[0]))
        loss.append(float(value[1]))
    # Note a str like ‘3.552’ can not be changed to int type directly
    # You need to change it to float first, can then you can change the float type ton int type
    last = loss[0]
    smoothed = []
    for point in loss:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return iters, smoothed


if __name__ == '__main__':
    print("Weight Line:")
    v = 4
    for i in range(v):
        path = './weight_3Source_' + str(i) + '.txt'
        iteration, loss = smooth_loss(path)
        plt.plot(iteration, loss, linewidth=2)

    plt.title("Iterations", fontsize=24)
    plt.xlabel("Iteration", fontsize=14, fontproperties=font1)
    plt.ylabel("Weight", fontsize=14, fontproperties=font1)
    plt.tick_params(axis='both', labelsize=14)
    plt.savefig('./weight_line.svg')
    plt.show()
