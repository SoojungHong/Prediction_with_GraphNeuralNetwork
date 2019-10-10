import matplotlib.pyplot as plt
from pylab import *

def test():
    #t = arange(0.0, 2.0, 0.01)
    #s = sin(2.5 * pi * t)
    t = [1, 2, 3, 4, 5,]
    s = [0.1, 0.3, 0.15, 0.6, 0.3]
    plot(t, s)

    xlabel('epoch')
    ylabel('loss')
    title('Loss')
    grid(True)
    show()
    plt.savefig('foo.pdf')


def visualize_loss(epoch_index, loss_vals):
    plot(epoch_index, loss_vals)

    xlabel('epoch')
    ylabel('loss')
    title('Loss')
    grid(True)
    show()
    plt.savefig('foo.pdf')

