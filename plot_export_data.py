import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import ast
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def draw_frame(box_c):
    fig = plt.figure()
    #     plt.rcParams['font.size'] = '16'
    ax = plt.axes(xlim=(0, 500), ylim=(0, 500))
    plt.axis('square')

    marker_size = 5
    ax.plot(
        box_c[0],
        box_c[1],
        's',
        color="b",
        #         fillstyle = 'none',
        markersize=marker_size)
    #         linewidth=2)

    plt.xticks([0, 250, 500])
    plt.yticks([0, 250, 500])


def proc_data(data):
    c = {}
    for col in data.columns:
        c_0 = []
        c_1 = []
        for r in data[col]:
            c__ = ast.literal_eval(r)
            c_0.append(c__[0])
            c_1.append(c__[1])
        c[int(col)] = [c_0, c_1]
    return c


def draw_multiple_frames(data):
    keys = sorted(data.keys())
    if not keys:
        return  # No data to draw

    first_key = keys[0]
    last_key = keys[-1]

    print(f"Drawing first frame (key={first_key})")
    draw_frame(data[first_key])
    plt.show()  # Show the first frame

    if first_key != last_key:
        print(f"Drawing last frame (key={last_key})")
        draw_frame(data[last_key])
        plt.show()  # Show the last frame

def animate_frames(data, time_step=1.0):
    keys = sorted(data.keys())
    fig, ax = plt.subplots()
    ax.set_xlim(0, 500)
    ax.set_ylim(0, 500)
    ax.set_aspect('equal')
    plt.xticks([0, 250, 500])
    plt.yticks([0, 250, 500])

    scatter, = ax.plot([], [], 's', color="b", markersize=5)
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

    def init():
        scatter.set_data([], [])
        time_text.set_text('')
        return scatter, time_text

    def update(frame_key):
        box_c = data[frame_key]
        scatter.set_data(box_c[0], box_c[1])
        current_time = frame_key * time_step
        time_text.set_text(f'Time: {current_time:.1f}s')
        return scatter, time_text

    anim = FuncAnimation(
        fig,
        update,
        frames=keys,
        init_func=init,
        blit=True,
        interval=100  # 100 ms between frames
    )

    #plt.show()
    anim.save('1746885691_animation.mp4', writer='ffmpeg', fps=10)

coords_1 = proc_data(pd.read_csv("data/e_1/1746920587/boxes.csv"))
draw_multiple_frames(coords_1)


#df = pd.read_csv('data/e_1/1746831890/boxes.csv')
#cleaned_data = proc_data(df)
#animate_frames(cleaned_data)