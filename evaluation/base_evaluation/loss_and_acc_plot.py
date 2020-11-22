import matplotlib.pyplot as plt
import evaluation.base_evaluation.from_file_get_list as getfile

mode = 0

if mode == 0:  # train的loss和test的loss在一张图中。acc同理
    y1 = getfile.get_acc(r'./model/model--87/acc.txt')[:24]
    y2 = getfile.get_loss(r'./model/model--87/loss.txt')[:24]
    y3 = getfile.get_acc(r'./model/model--87/acc_test.txt')[:24]
    y4 = getfile.get_loss(r'loss_test.txt')[:24]

    plt.figure(figsize=(6*1.2, 6))
    # plt.subplot(1, 2, 1)
    plt.plot(range(len(y1)), y1, 'o-', label="Training Accuracy", color='b')
    plt.plot(range(len(y3)), y3, 'o-', label="Testing Accuracy", color='r')
    plt.xlim(0, len(y1))
    plt.xlabel('Epoches')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.savefig('Accuracy.pdf')
    plt.show()
    # plt.subplot(1, 2, 2)
    plt.figure(figsize=(6*1.2, 6))
    plt.plot(range(len(y2)), y2, '.-', label="Training Loss", color='b')
    plt.plot(range(len(y4)), y4, '.-', label="Testing Loss", color='r')
    plt.xlim(0, len(y1))
    plt.xlabel('Epoches')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.savefig('Loss.pdf')
    plt.show()

if mode == 1:  # train的loss和acc放在一张图中。test同理
    def loss_and_acc(loss_list, acc_list, n_epochs, path):
        x1 = range(len(loss_list))
        x2 = range(len(acc_list))

        fig = plt.figure(figsize=(6 * 1.2, 6 * 1.2))

        ax1 = fig.add_subplot(111)
        ax1.plot(x1, loss_list, label='loss')
        ax1.set_ylabel('test loss')
        # ax1.legend(loc='best')
        ax1.set_title("loss and acc")

        ax2 = ax1.twinx()  # this is the important function
        ax2.plot(x2, acc_list, 'r', label='acc')
        ax2.set_xlim([0, n_epochs])
        ax2.set_ylabel('test acc')
        ax2.legend(loc='best')
        plt.savefig(path)
        plt.show()

    acc = getfile.get_test_labels('./model/model--87/acc.txt')[:24]
    loss = getfile.get_loss('./model/model--87/loss.txt')[:24]
    acc_test = getfile.get_test_labels('./model/model--87/acc_test.txt')[:24]
    loss_test = getfile.get_loss('./model/model--87/loss_test.txt')[:24]
    loss_and_acc(loss, acc, len(acc), './model/acc-87/evaluation_and_plot/acc_and_loss_plot_train.pdf')
    loss_and_acc(loss_test, acc_test, len(acc), './model/acc-87/evaluation_and_plot/acc_and_loss_plot_test.pdf')


