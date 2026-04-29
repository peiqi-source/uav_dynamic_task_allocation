import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm


def set_chinese_font():
    font_path = None
    for font in fm.findSystemFonts(fontpaths=None, fontext='ttf'):
        if 'SimHei' in font:
            font_path = font
            break

    if font_path is None:
        raise RuntimeError("没有找到合适的中文字体，请安装 SimHei 字体或其他支持中文的字体")

    font_properties = fm.FontProperties(fname=font_path)
    plt.rcParams['font.family'] = font_properties.get_name()
    plt.rcParams['axes.unicode_minus'] = False

def plot_clusters_and_routes(cluster):
    """
    Function to plot clusters and their attack routes in 2D space.

    Parameters:
    cluster : list of numpy arrays
        Each numpy array represents a cluster of targets. The columns are:
        - Target ID
        - x-coordinate
        - y-coordinate
        - Angle (not used in 2D plotting)
    """
    City = []
    City1 = []
    bestroute = []
    bestroute1 = []

    for c in cluster:
        A = c[:, [1, 2, 0]]  # Extract x, y, and ID
        B = c[:, 0].astype(int)  # Extract target IDs
        A1 = np.vstack(([0, 0, 0], A))  # Add starting point (0,0,0)
        B1 = np.hstack(([0], B))  # Add starting point ID (0)
        City.append(A)
        City1.append(A1)
        bestroute.extend(B)
        bestroute1.extend(B1)

    City1.append([0, 0, 0])
    bestroute1.append(0)

    City = np.vstack(City)
    City1 = np.vstack(City1)

    plot_path(bestroute, City, bestroute1, City1)


def plot_path(route, City, route1, City1):
    """
    Function to draw path in 2D space.

    Parameters:
    route : list of int
        The sequence of target IDs in the optimal path.
    City : numpy array
        Coordinates of each target (x, y, target ID).
    route1 : list of int
        The sequence of target IDs in the optimal path including starting point.
    City1 : numpy array
        Coordinates of each target (x, y, target ID) including starting point.
    """
    plt.figure()
    plt.scatter(0, 0, c='r', marker='*', s=100, label='base')

    plt.scatter(City[:, 0], City[:, 1], c='g', marker='o')

    for i in range(len(City)):
        plt.text(City[i, 0], City[i, 1], f'{int(City[i, 2])}', color='black')

    arrcolor = np.random.rand(3, )
    for i in range(1, len(City1)):
        plt.plot([City1[i - 1, 0], City1[i, 0]], [City1[i - 1, 1], City1[i, 1]], color=arrcolor)
        if route1[i] == 0:
            arrcolor = np.random.rand(3, )  # Change color for new route

    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('attack_order')
    plt.legend()
    plt.show()


'''
# Example usage:
cluster = [
    np.array([[2, 1, 74, 0.44234463], [15, -9, 51, 0.36286906]]),
    np.array([[9, 29, 7, 1], [6, 38, 21, 0.88681102], [3, 35, 42, 0.73004462], [18, 33, 46, 0.69945319],
              [4, 24, 46, 0.63972956], [19, 9, 32, 0.55242619]]),
    np.array([[5, -18, 38, 0.249811251], [11, -12, 15, 0.151676499], [1, -45, 28, 0.008315018], [17, -42, 25, 0]])
]
plot_clusters_and_routes(cluster)
'''
