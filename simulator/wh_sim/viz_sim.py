from . import Simulator
from matplotlib import pyplot as plt, animation
import time
import numpy as np
import sys
from os.path import dirname, realpath
import os

class VizSim(Simulator):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.snapshot_s = []#[1]

    def generate_dot_positional_data(self):
        agent_range = range(self.cfg.get('warehouse', 'number_of_agents'))
        x_data = [
            [self.warehouse.rob_c[i,0] for i in agent_range]
        ]
        y_data = [
            [self.warehouse.rob_c[i,1] for i in agent_range]
        ]
        marker = ['ko']

        return (x_data, y_data, marker)

    def generate_dot_heading_arrow(self):
        length = 20
        steps = 20
        agents = self.swarm.number_of_agents
        x_vec = []
        y_vec = []
        for i in range(agents):
            start_x = self.warehouse.rob_c[i,0]
            end_x = start_x + length * -np.cos(self.swarm.computed_heading[i])
            start_y = self.warehouse.rob_c[i,1]
            end_y = start_y + length * -np.sin(self.swarm.computed_heading[i])
            x_vec.append(np.linspace(start_x, end_x, steps).tolist())
            y_vec.append(np.linspace(start_y, end_y, steps).tolist())
        
        return x_vec, y_vec

    # iterate method called once per timestep
    def iterate(self, i, dot=None, box=None, h_line=None, cam_range=None):
        self.warehouse.iterate(self.cfg.get('heading_bias'), self.cfg.get('box_attraction'))
        counter = self.warehouse.counter

        self.animate(i, counter, dot, box, h_line, cam_range)
        self.take_snapshot(counter)
        # time.sleep(self.sim_delay)

        if self.verbose:
            if self.warehouse.counter == 1:
                print("Progress |", end="", flush=True)
            if self.warehouse.counter%100 == 0:
                print("=", end="", flush=True)

        self.exit_sim(counter)

    def animate(self, i, counter, dot=None, box=None, h_line=None, cam_range=None):
        cam_range.set_data(
            [self.warehouse.rob_c[i,0] for i in range(self.cfg.get('warehouse', 'number_of_agents'))],
            [self.warehouse.rob_c[i,1] for i in range(self.cfg.get('warehouse', 'number_of_agents'))]
        )

        x_data, y_data, _ = self.generate_dot_positional_data()
        for i in range(len(dot)):
            dot[i].set_data(x_data[i], y_data[i])
            
        box.set_data(
            [self.warehouse.box_c[n,0] for n in range(self.cfg.get('warehouse', 'number_of_boxes'))], 
            [self.warehouse.box_c[n,1] for n in range(self.cfg.get('warehouse', 'number_of_boxes'))])

        h_x_vec, h_y_vec = self.generate_dot_heading_arrow()
        for i in range(self.swarm.number_of_agents):
            h_line[i].set_data(h_x_vec[i], h_y_vec[i])

        realtime = int(np.ceil(counter/50))
        plt.title("Time is "+str(realtime)+"s")

    def take_snapshot(self, counter):
        if counter not in self.snapshot_s:
            return

        dir_path = dirname(dirname(dirname(realpath(__file__))))
        save_dir = os.path.join(dir_path, "animation")
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        plt.rcParams.update({'font.family':'serif', 'text.usetex': True, 'pdf.fonttype': 42})

        form="pdf"
        save_path = os.path.join(save_dir, "%d.%s"%(counter,form))
        fig = plt.gcf()
        fig.savefig(save_path, format=form, dpi=1200, bbox_inches="tight")        

    def exit_sim(self, counter):
        if counter > self.cfg.get('time_limit'):
            self.exit_threads = True
            try:
                self.save_anim_t.join()
            except:
                pass

            print("Exiting...")
            if self.cfg.get('animate'):
                exit()

    def run(self):
        if self.verbose:
            print("Running with seed: %d"%self.random_seed)

        self.init_animate()
        if self.cfg.get('save_animation'):
            try:
                self.save_anim_t = threading.Thread(target=self.save_animation)
                self.save_anim_t.start()
            except Exception as e:
                print(e)
        else:
            plt.show()
        
        if self.verbose:
            print("\n")

    def init_animate(self):
        fig = plt.figure()
        plt.rcParams['font.size'] = '16'
        ax = plt.axes(xlim=(0, self.cfg.get('warehouse', 'width')), ylim=(0, self.cfg.get('warehouse', 'height')))

        # assume all swarm radius same
        marker_size = 12.5
        cam_range_marker_size = marker_size/self.swarm.robot_r[0]*self.swarm.camera_sensor_range_V[0]
        cam_range, = ax.plot(
            [self.warehouse.rob_c[i,0] for i in range(self.cfg.get('warehouse', 'number_of_agents'))],
            [self.warehouse.rob_c[i,1] for i in range(self.cfg.get('warehouse', 'number_of_agents'))], 
            'ko', 
            markersize = cam_range_marker_size,
            # linestyle=":",
            color="#f2f2f2",
            fillstyle='none'
        )

        x_data, y_data, marker = self.generate_dot_positional_data()
        dot = {}
        for i in range(len(x_data)):
            dot[i], = ax.plot(x_data[i], y_data[i], marker[i],
                markersize = marker_size, fillstyle = 'none')
                    
        box, = ax.plot(
            [self.warehouse.box_c[i,0] for i in range(self.cfg.get('warehouse', 'number_of_boxes'))],
            [self.warehouse.box_c[i,1] for i in range(self.cfg.get('warehouse', 'number_of_boxes'))], 
            'bs', 
            markersize = marker_size-5)

        h_x_vec, h_y_vec = self.generate_dot_heading_arrow()
        h_line = {}
        for i in range(self.swarm.number_of_agents):
            h_line[i], = ax.plot(h_x_vec[i], h_y_vec[i], linestyle="dashed", color="#4CB580")

        plt.axis('square')
        plt.axis([0, self.cfg.get('warehouse', 'width'), 0, self.cfg.get('warehouse', 'height')])

        self.anim = animation.FuncAnimation(fig, self.iterate, 
            frames=10000, 
            interval=0.1, 
            save_count=sys.maxsize,
            fargs=(dot, box, h_line, cam_range))

        # ex = [self.cfg.get('warehouse', 'width')-self.cfg.get('warehouse', 'exit_width'), self.cfg.get('warehouse', 'width')-self.cfg.get('warehouse', 'exit_width')]
        # ey = [0, self.cfg.get('warehouse', 'height')]
        # plt.plot(ex, ey, ':')
