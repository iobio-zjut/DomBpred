#!/usr/bin/env /usr/bin/python
import os
import numpy as np
import matplotlib
import sys

docstring = """
options:
    -in_path=XXX.npz
        TrRosetta predicts the original output of the distance file, the dimension is L*L*37.
    -out_path=XXX.npz (optional)
        The output file after the conversion of the predicted distance file has a dimension of L*L*1.
        If this parameter is missing, the converted file will be generated in the current folder by default.
"""


def weighted_avg_and_std(values, weights):
    if len(values) == 1:
        return values[0], 0

    average = np.average(values, weights=weights)
    variance = np.array(values).std()
    return average, variance


def dist(npz):
    dat = npz['dist']
    nres = int(dat.shape[0])

    pcut = 0.05
    bin_p = 0.01

    mat = np.zeros((nres, nres))
    cont = np.zeros((nres, nres))

    for i in range(0, nres):
        for j in range(0, nres):
            if j == i:
                mat[i][i] = 4
                cont[i][j] = 1
                continue

            if j < i:
                mat[i][j] = mat[j][i]
                cont[i][j] = cont[j][i]
                continue

            # check probability
            Praw = dat[i][j]
            first_bin = 5
            first_d = 4.25  # 4-->3.75, 5-->4.25
            weight = 0

            pcont = 0
            for ii in range(first_bin, 13):
                pcont += Praw[ii]
            cont[i][j] = pcont

            for P in Praw[first_bin:]:
                if P > bin_p:
                    weight += P

            if weight < pcut:
                mat[i][j] = 20
                continue

            Pnorm = [P for P in Praw[first_bin:]]

            probs = []
            dists = []
            xs = []
            ys = []
            dmax = 0
            pmax = -1

            for k, P in enumerate(Pnorm):
                d = first_d + k * 0.5
                if P > pmax:
                    dmax = d
                    pmax = P

                if P > bin_p:
                    probs.append(P)
                    dists.append(d)
                xs.append(d)

            e_dis = 8
            e_std = 0
            if len(probs) == 0:
                e_dis = dmax
                e_std = 0
            else:
                probs = [P / sum(probs) for P in probs]
                e_dis, e_std = weighted_avg_and_std(dists, probs)

            mat[i][j] = e_dis

    return mat, cont


if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.stderr.write("Sorry, too few parameters")
        sys.stderr.write(docstring)
        exit()

    NPZ = OUT = None
    for arg in sys.argv[1:]:
        if arg.startswith("-in_path="):
            NPZ = os.path.abspath(arg[len("-in_path="):])
        elif arg.startswith("-out_path="):
            OUT = os.path.abspath(arg[len("-out_path="):])
        elif arg.startswith("-"):
            sys.stderr.write("Sorry, Unknown option %s\n" % arg)
            exit()
        else:
            sys.stderr.write("Sorry, Too many input parameters\n")
            exit()

    npz = np.load(NPZ)

    matplotlib.use('Agg')
    matplotlib.rcParams['image.cmap'] = 'gist_stern'

    img, cont = dist(npz)

    for i in range(0, img.shape[0]):
        img[i, i] = 0

    if OUT is None:
        nowfilepath = sys.path[0] + '/'
        filename = os.path.basename(NPZ)
        OUT = nowfilepath + "LL1_" + filename
        np.savez_compressed(OUT, dist=img, contact=cont)
    else:
        np.savez_compressed(OUT, dist=img, contact=cont)

    pass
