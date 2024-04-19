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


def smooth_loss(path, weight=0):
    iter = []
    loss = []
    data = read_txt(path)
    for value in data:
        iter.append(int(value[0]))
        loss.append(float(value[1]))
    # Note a str like ‘3.552’ can not be changed to int type directly
    # You need to change it to float first, can then you can change the float type ton int type
    last = loss[0]
    smoothed = []
    for point in loss:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return iter, smoothed


if __name__ == '__main__':
    print("Loss Line:")
    dataset = "3Source"
    path = './loss_' + dataset + '.txt'
    iter, loss = smooth_loss(path)
    plt.plot(iter, loss, linewidth=2)
    plt.title("Loss-iters", fontsize=24)
    # plt.xlabel("iters", fontsize=14, fontproperties=font1)
    # plt.ylabel("loss", fontsize=14, fontproperties=font1)
    plt.tick_params(axis='both', labelsize=14)
    plt.savefig('./' + dataset + '_loss_func.svg')
    plt.show()
