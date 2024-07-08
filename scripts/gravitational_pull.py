import matplotlib.pyplot as plt
import numpy as np


def plot_pull():
    x, y = np.meshgrid(np.linspace(-7, 7, 50), np.linspace(-7, 7, 50))
    z = (
        3 / np.sqrt(x**2 + y**2)
        + 1.5 / np.sqrt((x - 5) ** 2 + (y - 4) ** 2)
        + 0.5 / np.sqrt((x + 3) ** 2 + (y + 4) ** 2)
    )
    # 3d plot without axes and grid
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, -z, cmap='magma', edgecolor='none', alpha=0.8)
    ax.set_axis_off()
    plt.savefig('gravitational_pull_3d.png', transparent=True, dpi=300)


if __name__ == '__main__':
    plot_pull()
