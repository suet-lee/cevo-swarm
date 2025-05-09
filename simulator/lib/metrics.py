import math

# Metric evaluation class
class EvalMetric:

    def compute_metrics(self, ap_c, box_c, box_t):
        self.compute_symmetry(box_c, box_t, 0)
        self.compute_symmetry(box_c, box_t, 1)
        self.compute_spread(box_c) # count numbers in grid cells
        self.compute_coherence(box_c, ap_c) # check how many boxes within varying radii of AP (if one is much higher, then assume ring has formed i.e. coherence)

    def compute_symmetry(self, box_c, box_t, axis=0):
        c1 = 0
        c2 = 0
        t1 = {}
        t2 = {}

        # 0-axis: (0,0) to (500,500)
        if axis == 0:
            for idx, c in enumerate(box_c):
                t = box_t[idx]
                if t not in t1:
                    t1[t] = 0
                if t not in t2:
                    t2[t] = 0

                if c[0] >= c[1]:
                    c1 += 1
                    t1[t] += 1
                else:
                    c2 += 1
                    t2[t] += 1

        # 1-axis: (0,500) to (500,0)
        elif axis == 1:
            for idx, c in enumerate(box_c):
                t = box_t[idx]
                if t not in t1:
                    t1[t] = 0
                if t not in t2:
                    t2[t] = 0

                if c[0] + c[1] <= 500: # TODO remove hardcode
                    c1 += 1
                    t1[t] += 1
                else:
                    c2 += 1
                    t2[t] += 1

        # compute box type ratio
        t_r = 0
        for t in t1:
            r += (t1[t]/t2[t])/len(t1)

        if t_r > 1:
            t_r = 1/t_r

        if c1 > c2:
            c_r = c2/c1
        else:
            c_r = c1/c2
        
        return c_r, t_r

    # TODO compute box type spread
    def compute_spread(self, box_c, cell_size=25):
        counts = {}

        for idx, c in enumerate(box_c):
            x = math.floor(c[0]/cell_size)
            y = math.floor(c[1]/cell_size)
            key = "f{x},{y}"
            if key in counts:
                counts[key] += 1
            else:
                counts[key] = 1

        no_cells = (500/cell_size)*(500/cell_size)
        no_box = len(box_c)
        diff = 0
        perf_dist = int(no_box/no_cells)
        # perfect distribution would be: box in cell = no_box/no_cells
        # count how many boxes away from perfect distribution each cell is
        for x in range(0,500,cell_size):
            for y in range(0,500,cell_size):
                key = "f{x},{y}"
                if key in counts:
                    diff += math.abs(counts[key] - perf_dist)
                else:
                    diff += perf_dist

        return diff/no_box # closer to 0 is more even distribution

    def compute_coherence(self, box_c, ap_c):
        coh = []
        for c in ap_c:
            box_in_r = {}
            for bc in box_c:
                d0 = bc[0] - c[0]
                d1 = bc[1] - c[1]
                r = math.sqrt(d0*d0 + d1*d1)
                r_band = math.floor(r/10)
                if r_band in box_in_r:
                    box_in_r[r_band] += 1
                else:
                    box_in_r[r_band] = 1

            coh_ = max(box_in_r.values()) - min(box_in_r.values()) # higher means more coherence
            coh.append(coh_)
        
        return coh
                        